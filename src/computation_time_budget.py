"""
Computation Time Budget Manager
Implements time limits and progress tracking for quality evaluations
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import subprocess
import signal
import os

logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """Status of time budget."""
    ACTIVE = "active"
    EXCEEDED = "exceeded"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TimeBudget:
    """Time budget configuration for an operation."""
    total_budget: float
    vmaf_budget: float
    ssim_budget: float
    progress_interval: float
    graceful_degradation: bool


@dataclass
class ProgressUpdate:
    """Progress update information."""
    timestamp: float
    elapsed_time: float
    estimated_remaining: Optional[float]
    progress_percentage: Optional[float]
    operation_type: str
    details: Dict[str, Any]


class ComputationTimeBudget:
    """Manages time budgets and progress tracking for quality evaluations."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Budget configuration
        self.budget = self._load_time_budget()
        
        # Current operation tracking
        self.current_operation: Optional[str] = None
        self.operation_start_time: Optional[float] = None
        self.budget_status = BudgetStatus.ACTIVE
        self.progress_updates: List[ProgressUpdate] = []
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
        self.progress_thread: Optional[threading.Thread] = None
        self.stop_progress_tracking = threading.Event()
        
        # Process management for timeout handling
        self.current_process: Optional[subprocess.Popen] = None
        self.process_lock = threading.Lock()
        
        logger.info(f"Computation time budget initialized: "
                   f"total={self.budget.total_budget}s, "
                   f"vmaf={self.budget.vmaf_budget}s, "
                   f"ssim={self.budget.ssim_budget}s")
    
    def _load_time_budget(self) -> TimeBudget:
        """Load time budget configuration."""
        if self.config:
            base_path = 'quality_evaluation.time_budget'
            return TimeBudget(
                total_budget=self.config.get(f'{base_path}.total_budget_seconds', 300.0),
                vmaf_budget=self.config.get(f'{base_path}.vmaf_budget_seconds', 180.0),
                ssim_budget=self.config.get(f'{base_path}.ssim_budget_seconds', 90.0),
                progress_interval=self.config.get(f'{base_path}.progress_interval_seconds', 5.0),
                graceful_degradation=self.config.get(f'{base_path}.graceful_degradation', True)
            )
        
        # Default budget
        return TimeBudget(
            total_budget=300.0,
            vmaf_budget=180.0,
            ssim_budget=90.0,
            progress_interval=5.0,
            graceful_degradation=True
        )
    
    def get_operation_budget(self, operation_type: str) -> float:
        """Get time budget for specific operation type.
        
        Args:
            operation_type: Type of operation ('vmaf', 'ssim', 'combined')
            
        Returns:
            Time budget in seconds
        """
        if operation_type == 'vmaf':
            return self.budget.vmaf_budget
        elif operation_type == 'ssim':
            return self.budget.ssim_budget
        elif operation_type == 'combined':
            return self.budget.total_budget
        else:
            return self.budget.total_budget * 0.5  # Default for unknown operations
    
    def start_operation(self, operation_type: str, progress_callback: Optional[Callable] = None) -> bool:
        """Start tracking time budget for an operation.
        
        Args:
            operation_type: Type of operation to track
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if operation can start, False if budget exceeded
        """
        if self.budget_status == BudgetStatus.EXCEEDED:
            logger.warning(f"Cannot start {operation_type}: time budget already exceeded")
            return False
        
        self.current_operation = operation_type
        self.operation_start_time = time.time()
        self.budget_status = BudgetStatus.ACTIVE
        self.progress_callback = progress_callback
        self.progress_updates.clear()
        
        # Start progress tracking thread
        if progress_callback:
            self.stop_progress_tracking.clear()
            self.progress_thread = threading.Thread(
                target=self._progress_tracking_loop,
                daemon=True
            )
            self.progress_thread.start()
        
        operation_budget = self.get_operation_budget(operation_type)
        logger.info(f"Started operation '{operation_type}' with {operation_budget:.1f}s budget")
        
        return True
    
    def _progress_tracking_loop(self) -> None:
        """Background thread for progress tracking."""
        while not self.stop_progress_tracking.is_set():
            if self.current_operation and self.operation_start_time:
                elapsed = time.time() - self.operation_start_time
                operation_budget = self.get_operation_budget(self.current_operation)
                
                # Create progress update
                progress = ProgressUpdate(
                    timestamp=time.time(),
                    elapsed_time=elapsed,
                    estimated_remaining=max(0, operation_budget - elapsed),
                    progress_percentage=min(100.0, (elapsed / operation_budget) * 100),
                    operation_type=self.current_operation,
                    details={
                        'budget_seconds': operation_budget,
                        'budget_exceeded': elapsed > operation_budget
                    }
                )
                
                self.progress_updates.append(progress)
                
                # Call progress callback
                if self.progress_callback:
                    try:
                        self.progress_callback(progress)
                    except Exception as e:
                        logger.debug(f"Progress callback error: {e}")
                
                # Check if budget exceeded
                if elapsed > operation_budget:
                    self.budget_status = BudgetStatus.EXCEEDED
                    logger.warning(f"Time budget exceeded for {self.current_operation}: "
                                 f"{elapsed:.1f}s > {operation_budget:.1f}s")
                    break
            
            # Wait for next progress update
            self.stop_progress_tracking.wait(self.budget.progress_interval)
    
    def check_budget_status(self) -> tuple[BudgetStatus, float, float]:
        """Check current budget status.
        
        Returns:
            Tuple of (status, elapsed_time, remaining_time)
        """
        if not self.current_operation or not self.operation_start_time:
            return BudgetStatus.ACTIVE, 0.0, self.budget.total_budget
        
        elapsed = time.time() - self.operation_start_time
        operation_budget = self.get_operation_budget(self.current_operation)
        remaining = max(0, operation_budget - elapsed)
        
        if elapsed > operation_budget:
            status = BudgetStatus.EXCEEDED
        else:
            status = self.budget_status
        
        return status, elapsed, remaining
    
    def execute_with_timeout(
        self, 
        command: List[str], 
        operation_type: str,
        timeout_override: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute command with time budget enforcement.
        
        Args:
            command: Command to execute
            operation_type: Type of operation for budget calculation
            timeout_override: Optional timeout override
            
        Returns:
            Dictionary with execution results
        """
        timeout = timeout_override or self.get_operation_budget(operation_type)
        
        result = {
            'success': False,
            'stdout': '',
            'stderr': '',
            'returncode': -1,
            'execution_time': 0.0,
            'timeout_exceeded': False,
            'budget_exceeded': False,
            'graceful_degradation': False
        }
        
        start_time = time.time()
        
        try:
            logger.debug(f"Executing with {timeout:.1f}s timeout: {' '.join(command[:3])}...")
            
            # Start process
            with self.process_lock:
                self.current_process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
            
            # Wait for completion with timeout
            try:
                stdout, stderr = self.current_process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                
                result.update({
                    'success': self.current_process.returncode == 0,
                    'stdout': stdout,
                    'stderr': stderr,
                    'returncode': self.current_process.returncode,
                    'execution_time': execution_time
                })
                
                logger.debug(f"Command completed in {execution_time:.2f}s")
                
            except subprocess.TimeoutExpired:
                execution_time = time.time() - start_time
                
                # Handle timeout with graceful degradation
                if self.budget.graceful_degradation:
                    partial_result = self._handle_graceful_timeout(operation_type, execution_time)
                    result.update(partial_result)
                    result['graceful_degradation'] = True
                else:
                    # Terminate process
                    self._terminate_process()
                    result.update({
                        'timeout_exceeded': True,
                        'execution_time': execution_time,
                        'stderr': f'Operation timed out after {timeout:.1f} seconds'
                    })
                
                logger.warning(f"Command timed out after {execution_time:.1f}s")
        
        except Exception as e:
            execution_time = time.time() - start_time
            result.update({
                'execution_time': execution_time,
                'stderr': f'Execution error: {str(e)}'
            })
            logger.error(f"Command execution error: {e}")
        
        finally:
            with self.process_lock:
                self.current_process = None
        
        # Check if budget was exceeded
        operation_budget = self.get_operation_budget(operation_type)
        if result['execution_time'] > operation_budget:
            result['budget_exceeded'] = True
        
        return result
    
    def _handle_graceful_timeout(self, operation_type: str, elapsed_time: float) -> Dict[str, Any]:
        """Handle timeout with graceful degradation.
        
        Args:
            operation_type: Type of operation that timed out
            elapsed_time: Time elapsed before timeout
            
        Returns:
            Dictionary with partial results
        """
        logger.info(f"Attempting graceful degradation for {operation_type} timeout")
        
        # Try to get partial output from process
        partial_stdout = ""
        partial_stderr = ""
        
        try:
            if self.current_process:
                # Send SIGTERM first for graceful shutdown
                self.current_process.terminate()
                
                # Wait briefly for graceful shutdown
                try:
                    stdout, stderr = self.current_process.communicate(timeout=5.0)
                    partial_stdout = stdout or ""
                    partial_stderr = stderr or ""
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    self.current_process.kill()
                    try:
                        stdout, stderr = self.current_process.communicate(timeout=2.0)
                        partial_stdout = stdout or ""
                        partial_stderr = stderr or ""
                    except:
                        pass
        
        except Exception as e:
            logger.debug(f"Error during graceful timeout handling: {e}")
        
        return {
            'success': False,
            'stdout': partial_stdout,
            'stderr': partial_stderr + f'\nGraceful timeout after {elapsed_time:.1f}s',
            'returncode': -1,
            'timeout_exceeded': True,
            'partial_results_available': len(partial_stdout) > 0 or len(partial_stderr) > 0
        }
    
    def _terminate_process(self) -> None:
        """Terminate the current process."""
        try:
            if self.current_process:
                # Try graceful termination first
                self.current_process.terminate()
                
                # Wait briefly for graceful shutdown
                try:
                    self.current_process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    self.current_process.kill()
                    self.current_process.wait(timeout=2.0)
        
        except Exception as e:
            logger.debug(f"Error terminating process: {e}")
    
    def finish_operation(self, success: bool = True) -> Dict[str, Any]:
        """Finish the current operation and get summary.
        
        Args:
            success: Whether the operation completed successfully
            
        Returns:
            Dictionary with operation summary
        """
        if not self.current_operation or not self.operation_start_time:
            return {}
        
        elapsed = time.time() - self.operation_start_time
        operation_budget = self.get_operation_budget(self.current_operation)
        
        # Stop progress tracking
        if self.progress_thread:
            self.stop_progress_tracking.set()
            self.progress_thread.join(timeout=1.0)
            self.progress_thread = None
        
        # Update status
        if success:
            self.budget_status = BudgetStatus.COMPLETED
        elif elapsed > operation_budget:
            self.budget_status = BudgetStatus.EXCEEDED
        
        summary = {
            'operation_type': self.current_operation,
            'elapsed_time': elapsed,
            'budget_seconds': operation_budget,
            'budget_exceeded': elapsed > operation_budget,
            'success': success,
            'status': self.budget_status.value,
            'progress_updates_count': len(self.progress_updates),
            'efficiency': min(1.0, operation_budget / elapsed) if elapsed > 0 else 1.0
        }
        
        logger.info(f"Operation '{self.current_operation}' finished: "
                   f"{elapsed:.2f}s (budget: {operation_budget:.1f}s), "
                   f"success={success}")
        
        # Reset current operation
        self.current_operation = None
        self.operation_start_time = None
        
        return summary
    
    def cancel_operation(self, reason: str = "user_cancelled") -> None:
        """Cancel the current operation.
        
        Args:
            reason: Reason for cancellation
        """
        if self.current_operation:
            logger.info(f"Cancelling operation '{self.current_operation}': {reason}")
            
            # Terminate any running process
            self._terminate_process()
            
            # Stop progress tracking
            if self.progress_thread:
                self.stop_progress_tracking.set()
                self.progress_thread.join(timeout=1.0)
                self.progress_thread = None
            
            self.budget_status = BudgetStatus.CANCELLED
            self.current_operation = None
            self.operation_start_time = None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of progress updates.
        
        Returns:
            Dictionary with progress summary
        """
        if not self.progress_updates:
            return {
                'total_updates': 0,
                'current_progress': None,
                'average_progress_interval': 0.0
            }
        
        latest_update = self.progress_updates[-1]
        
        # Calculate average interval between updates
        if len(self.progress_updates) > 1:
            intervals = []
            for i in range(1, len(self.progress_updates)):
                interval = self.progress_updates[i].timestamp - self.progress_updates[i-1].timestamp
                intervals.append(interval)
            avg_interval = sum(intervals) / len(intervals)
        else:
            avg_interval = 0.0
        
        return {
            'total_updates': len(self.progress_updates),
            'current_progress': {
                'elapsed_time': latest_update.elapsed_time,
                'estimated_remaining': latest_update.estimated_remaining,
                'progress_percentage': latest_update.progress_percentage,
                'operation_type': latest_update.operation_type
            },
            'average_progress_interval': avg_interval,
            'budget_status': self.budget_status.value
        }
    
    def adjust_budget(self, operation_type: str, new_budget: float) -> None:
        """Adjust budget for specific operation type.
        
        Args:
            operation_type: Type of operation to adjust
            new_budget: New budget in seconds
        """
        if operation_type == 'vmaf':
            self.budget.vmaf_budget = new_budget
        elif operation_type == 'ssim':
            self.budget.ssim_budget = new_budget
        elif operation_type == 'total':
            self.budget.total_budget = new_budget
        
        logger.info(f"Adjusted {operation_type} budget to {new_budget:.1f}s")
    
    def get_budget_configuration(self) -> Dict[str, Any]:
        """Get current budget configuration.
        
        Returns:
            Dictionary with budget configuration
        """
        return {
            'total_budget': self.budget.total_budget,
            'vmaf_budget': self.budget.vmaf_budget,
            'ssim_budget': self.budget.ssim_budget,
            'progress_interval': self.budget.progress_interval,
            'graceful_degradation': self.budget.graceful_degradation,
            'current_operation': self.current_operation,
            'budget_status': self.budget_status.value
        }