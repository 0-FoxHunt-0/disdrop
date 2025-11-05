"""
Quality-Size Optimization Engine
Implements multi-dimensional parameter optimization with constraint satisfaction
for size limits while maximizing quality improvement
"""

import logging
import math
import time
from typing import Dict, Any, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    MULTI_DIMENSIONAL = "multi_dimensional"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    ITERATIVE_IMPROVEMENT = "iterative_improvement"
    FALLBACK_STRATEGY = "fallback_strategy"


class ConvergenceStatus(Enum):
    """Convergence detection status."""
    CONVERGED = "converged"
    IMPROVING = "improving"
    STAGNANT = "stagnant"
    DIVERGING = "diverging"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class OptimizationConstraints:
    """Constraints for quality-size optimization."""
    max_size_mb: float
    min_quality_vmaf: float
    min_quality_ssim: float
    max_iterations: int
    convergence_threshold: float
    size_tolerance_mb: float
    quality_improvement_threshold: float
    time_budget_seconds: Optional[float] = None


@dataclass
class ParameterSpace:
    """Multi-dimensional parameter space for optimization."""
    bitrate_range: Tuple[int, int]  # (min, max) in kbps
    crf_range: Tuple[int, int]      # (min, max) CRF values
    preset_options: List[str]       # Available presets
    encoder_options: List[str]      # Available encoders
    resolution_factors: List[float] # Resolution scaling factors
    fps_factors: List[float]        # FPS scaling factors


@dataclass
class OptimizationPoint:
    """A point in the optimization space."""
    bitrate: int
    crf: Optional[int]
    preset: str
    encoder: str
    resolution_factor: float
    fps_factor: float
    quality_score: Optional[float] = None
    size_mb: Optional[float] = None
    evaluation_time: Optional[float] = None
    feasible: bool = True
    constraint_violations: List[str] = None


@dataclass
class OptimizationResult:
    """Result of quality-size optimization."""
    best_point: OptimizationPoint
    optimization_history: List[OptimizationPoint]
    convergence_status: ConvergenceStatus
    total_iterations: int
    total_time_seconds: float
    constraint_satisfaction: Dict[str, bool]
    improvement_achieved: float
    fallback_used: bool
    optimization_summary: Dict[str, Any]


class QualitySizeOptimizationEngine:
    """
    Multi-dimensional optimization engine for quality-size trade-offs.
    
    Implements advanced optimization algorithms including:
    - Multi-dimensional parameter space exploration
    - Constraint satisfaction with quality improvement focus
    - Iterative improvement with convergence detection
    - Intelligent fallback strategies
    """
    
    def __init__(self, config_manager=None, quality_improvement_engine=None):
        self.config = config_manager
        self.quality_engine = quality_improvement_engine
        
        # Load optimization configuration
        self.max_parallel_evaluations = self._get_config('max_parallel_evaluations', 3)
        self.convergence_patience = self._get_config('convergence_patience', 3)
        self.exploration_factor = self._get_config('exploration_factor', 0.2)
        self.exploitation_factor = self._get_config('exploitation_factor', 0.8)
        self.adaptive_step_size = self._get_config('adaptive_step_size', True)
        
        # Optimization state
        self.optimization_cache = {}
        self.pareto_frontier = []
        
        logger.info("Quality-Size Optimization Engine initialized")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'quality_size_optimization.{key}', default)
        return default
    
    def optimize_quality_size_tradeoff(
        self,
        input_path: str,
        constraints: OptimizationConstraints,
        parameter_space: ParameterSpace,
        initial_point: Optional[OptimizationPoint] = None,
        video_characteristics: Optional[Dict[str, Any]] = None
    ) -> OptimizationResult:
        """
        Perform multi-dimensional quality-size optimization.
        
        Args:
            input_path: Path to input video
            constraints: Optimization constraints
            parameter_space: Available parameter space
            initial_point: Starting point for optimization
            video_characteristics: Video content characteristics
            
        Returns:
            OptimizationResult with best parameters and optimization history
        """
        logger.info("Starting multi-dimensional quality-size optimization")
        start_time = time.time()
        
        # Initialize optimization state
        optimization_history = []
        best_point = None
        convergence_status = ConvergenceStatus.IMPROVING
        iteration = 0
        
        # Generate initial population if no starting point provided
        if initial_point is None:
            initial_population = self._generate_initial_population(
                parameter_space, constraints, video_characteristics
            )
        else:
            initial_population = [initial_point]
        
        # Evaluate initial population
        logger.info(f"Evaluating initial population of {len(initial_population)} points")
        evaluated_population = self._evaluate_population_parallel(
            input_path, initial_population, constraints
        )
        
        optimization_history.extend(evaluated_population)
        best_point = self._select_best_point(evaluated_population, constraints)
        
        if best_point:
            logger.info(f"Initial best point: quality={best_point.quality_score:.1f}, "
                       f"size={best_point.size_mb:.2f}MB")
        
        # Main optimization loop
        while (iteration < constraints.max_iterations and 
               convergence_status == ConvergenceStatus.IMPROVING and
               (constraints.time_budget_seconds is None or 
                time.time() - start_time < constraints.time_budget_seconds)):
            
            iteration += 1
            logger.info(f"Optimization iteration {iteration}/{constraints.max_iterations}")
            
            # Generate candidate points using multiple strategies
            candidate_points = self._generate_candidate_points(
                best_point, parameter_space, constraints, optimization_history, iteration
            )
            
            # Evaluate candidates
            if candidate_points:
                evaluated_candidates = self._evaluate_population_parallel(
                    input_path, candidate_points, constraints
                )
                optimization_history.extend(evaluated_candidates)
                
                # Update best point
                new_best = self._select_best_point(evaluated_candidates + [best_point], constraints)
                
                # Check for improvement and convergence
                improvement = self._calculate_improvement(best_point, new_best, constraints)
                convergence_status = self._check_convergence(
                    optimization_history, constraints, iteration
                )
                
                if improvement > constraints.quality_improvement_threshold:
                    best_point = new_best
                    logger.info(f"Improvement found: quality={best_point.quality_score:.1f}, "
                               f"size={best_point.size_mb:.2f}MB, improvement={improvement:.2f}")
                else:
                    logger.info(f"No significant improvement: {improvement:.2f}")
            else:
                logger.warning("No candidate points generated")
                convergence_status = ConvergenceStatus.STAGNANT
        
        # Handle convergence or termination
        if iteration >= constraints.max_iterations:
            convergence_status = ConvergenceStatus.MAX_ITERATIONS
        
        total_time = time.time() - start_time
        
        # Apply fallback strategy if needed
        fallback_used = False
        if not best_point or not self._satisfies_constraints(best_point, constraints):
            logger.warning("Optimization failed to find satisfactory solution, applying fallback")
            best_point = self._apply_fallback_strategy(
                optimization_history, constraints, parameter_space
            )
            fallback_used = True
        
        # Calculate final metrics
        constraint_satisfaction = self._evaluate_constraint_satisfaction(best_point, constraints)
        improvement_achieved = self._calculate_total_improvement(
            optimization_history[0] if optimization_history else None, best_point, constraints
        )
        
        # Generate optimization summary
        optimization_summary = self._generate_optimization_summary(
            optimization_history, best_point, constraints, total_time
        )
        
        result = OptimizationResult(
            best_point=best_point,
            optimization_history=optimization_history,
            convergence_status=convergence_status,
            total_iterations=iteration,
            total_time_seconds=total_time,
            constraint_satisfaction=constraint_satisfaction,
            improvement_achieved=improvement_achieved,
            fallback_used=fallback_used,
            optimization_summary=optimization_summary
        )
        
        logger.info(f"Optimization completed: {convergence_status.value}, "
                   f"{iteration} iterations, {total_time:.1f}s, "
                   f"improvement={improvement_achieved:.2f}")
        
        return result
    
    def _generate_initial_population(
        self,
        parameter_space: ParameterSpace,
        constraints: OptimizationConstraints,
        video_characteristics: Optional[Dict[str, Any]]
    ) -> List[OptimizationPoint]:
        """Generate initial population for optimization."""
        
        population = []
        
        # Strategy 1: Conservative baseline
        conservative_point = OptimizationPoint(
            bitrate=parameter_space.bitrate_range[0],
            crf=parameter_space.crf_range[1],  # Higher CRF = lower quality, smaller size
            preset=parameter_space.preset_options[0] if parameter_space.preset_options else 'medium',
            encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
            resolution_factor=min(parameter_space.resolution_factors),
            fps_factor=min(parameter_space.fps_factors)
        )
        population.append(conservative_point)
        
        # Strategy 2: Quality-focused
        quality_point = OptimizationPoint(
            bitrate=parameter_space.bitrate_range[1],
            crf=parameter_space.crf_range[0],  # Lower CRF = higher quality
            preset=parameter_space.preset_options[-1] if parameter_space.preset_options else 'slow',
            encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
            resolution_factor=max(parameter_space.resolution_factors),
            fps_factor=max(parameter_space.fps_factors)
        )
        population.append(quality_point)
        
        # Strategy 3: Balanced approach
        mid_bitrate = (parameter_space.bitrate_range[0] + parameter_space.bitrate_range[1]) // 2
        mid_crf = (parameter_space.crf_range[0] + parameter_space.crf_range[1]) // 2
        mid_preset_idx = len(parameter_space.preset_options) // 2 if parameter_space.preset_options else 0
        
        balanced_point = OptimizationPoint(
            bitrate=mid_bitrate,
            crf=mid_crf,
            preset=parameter_space.preset_options[mid_preset_idx] if parameter_space.preset_options else 'medium',
            encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
            resolution_factor=1.0,
            fps_factor=1.0
        )
        population.append(balanced_point)
        
        # Strategy 4: Content-adaptive points based on video characteristics
        if video_characteristics:
            adaptive_points = self._generate_content_adaptive_points(
                parameter_space, video_characteristics
            )
            population.extend(adaptive_points)
        
        # Strategy 5: Random exploration points
        random_points = self._generate_random_points(parameter_space, 2)
        population.extend(random_points)
        
        return population
    
    def _generate_content_adaptive_points(
        self,
        parameter_space: ParameterSpace,
        video_characteristics: Dict[str, Any]
    ) -> List[OptimizationPoint]:
        """Generate optimization points adapted to video content."""
        
        points = []
        
        # Extract content characteristics
        motion_level = video_characteristics.get('motion_level', 'medium')
        complexity = video_characteristics.get('complexity', 'medium')
        duration = video_characteristics.get('duration', 60)
        
        # High motion content - prioritize temporal quality
        if motion_level == 'high':
            high_motion_point = OptimizationPoint(
                bitrate=int(parameter_space.bitrate_range[1] * 0.8),
                crf=parameter_space.crf_range[0] + 2,
                preset='fast',  # Faster preset for high motion
                encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
                resolution_factor=0.9,  # Slight resolution reduction for motion
                fps_factor=1.0
            )
            points.append(high_motion_point)
        
        # High complexity content - prioritize spatial quality
        if complexity == 'high':
            high_complexity_point = OptimizationPoint(
                bitrate=int(parameter_space.bitrate_range[1] * 0.9),
                crf=parameter_space.crf_range[0],
                preset='slow',  # Slower preset for complex content
                encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
                resolution_factor=1.0,
                fps_factor=0.95  # Slight FPS reduction for complexity
            )
            points.append(high_complexity_point)
        
        # Long duration content - prioritize compression efficiency
        if duration > 300:  # 5 minutes
            long_duration_point = OptimizationPoint(
                bitrate=int(parameter_space.bitrate_range[0] * 1.5),
                crf=parameter_space.crf_range[0] + 3,
                preset='medium',
                encoder='libx265' if 'libx265' in parameter_space.encoder_options else parameter_space.encoder_options[0],
                resolution_factor=0.85,
                fps_factor=0.9
            )
            points.append(long_duration_point)
        
        return points
    
    def _generate_random_points(
        self,
        parameter_space: ParameterSpace,
        count: int
    ) -> List[OptimizationPoint]:
        """Generate random points for exploration."""
        
        points = []
        
        for _ in range(count):
            # Random bitrate within range
            bitrate = np.random.randint(
                parameter_space.bitrate_range[0],
                parameter_space.bitrate_range[1] + 1
            )
            
            # Random CRF within range
            crf = np.random.randint(
                parameter_space.crf_range[0],
                parameter_space.crf_range[1] + 1
            )
            
            # Random preset
            preset = np.random.choice(parameter_space.preset_options) if parameter_space.preset_options else 'medium'
            
            # Random encoder
            encoder = np.random.choice(parameter_space.encoder_options) if parameter_space.encoder_options else 'libx264'
            
            # Random resolution factor
            resolution_factor = np.random.choice(parameter_space.resolution_factors)
            
            # Random FPS factor
            fps_factor = np.random.choice(parameter_space.fps_factors)
            
            point = OptimizationPoint(
                bitrate=bitrate,
                crf=crf,
                preset=preset,
                encoder=encoder,
                resolution_factor=resolution_factor,
                fps_factor=fps_factor
            )
            points.append(point)
        
        return points
    
    def _generate_candidate_points(
        self,
        best_point: OptimizationPoint,
        parameter_space: ParameterSpace,
        constraints: OptimizationConstraints,
        optimization_history: List[OptimizationPoint],
        iteration: int
    ) -> List[OptimizationPoint]:
        """Generate candidate points for the next iteration."""
        
        candidates = []
        
        # Strategy 1: Local search around best point
        local_candidates = self._generate_local_search_candidates(
            best_point, parameter_space, constraints
        )
        candidates.extend(local_candidates)
        
        # Strategy 2: Gradient-based improvement
        gradient_candidates = self._generate_gradient_based_candidates(
            best_point, optimization_history, parameter_space, constraints
        )
        candidates.extend(gradient_candidates)
        
        # Strategy 3: Constraint-guided search
        constraint_candidates = self._generate_constraint_guided_candidates(
            best_point, parameter_space, constraints
        )
        candidates.extend(constraint_candidates)
        
        # Strategy 4: Exploration vs exploitation balance
        exploration_candidates = self._generate_exploration_candidates(
            parameter_space, optimization_history, iteration, constraints.max_iterations
        )
        candidates.extend(exploration_candidates)
        
        # Remove duplicates and invalid points
        candidates = self._filter_and_deduplicate_candidates(
            candidates, optimization_history, parameter_space
        )
        
        return candidates
    
    def _generate_local_search_candidates(
        self,
        best_point: OptimizationPoint,
        parameter_space: ParameterSpace,
        constraints: OptimizationConstraints
    ) -> List[OptimizationPoint]:
        """Generate candidates through local search around the best point."""
        
        candidates = []
        
        # Bitrate variations
        bitrate_steps = [0.9, 1.1, 1.2]
        for step in bitrate_steps:
            new_bitrate = int(best_point.bitrate * step)
            if parameter_space.bitrate_range[0] <= new_bitrate <= parameter_space.bitrate_range[1]:
                candidate = OptimizationPoint(
                    bitrate=new_bitrate,
                    crf=best_point.crf,
                    preset=best_point.preset,
                    encoder=best_point.encoder,
                    resolution_factor=best_point.resolution_factor,
                    fps_factor=best_point.fps_factor
                )
                candidates.append(candidate)
        
        # CRF variations
        if best_point.crf is not None:
            crf_steps = [-2, -1, 1, 2]
            for step in crf_steps:
                new_crf = best_point.crf + step
                if parameter_space.crf_range[0] <= new_crf <= parameter_space.crf_range[1]:
                    candidate = OptimizationPoint(
                        bitrate=best_point.bitrate,
                        crf=new_crf,
                        preset=best_point.preset,
                        encoder=best_point.encoder,
                        resolution_factor=best_point.resolution_factor,
                        fps_factor=best_point.fps_factor
                    )
                    candidates.append(candidate)
        
        # Preset variations
        if parameter_space.preset_options:
            current_idx = parameter_space.preset_options.index(best_point.preset) if best_point.preset in parameter_space.preset_options else 0
            for offset in [-1, 1]:
                new_idx = current_idx + offset
                if 0 <= new_idx < len(parameter_space.preset_options):
                    candidate = OptimizationPoint(
                        bitrate=best_point.bitrate,
                        crf=best_point.crf,
                        preset=parameter_space.preset_options[new_idx],
                        encoder=best_point.encoder,
                        resolution_factor=best_point.resolution_factor,
                        fps_factor=best_point.fps_factor
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_gradient_based_candidates(
        self,
        best_point: OptimizationPoint,
        optimization_history: List[OptimizationPoint],
        parameter_space: ParameterSpace,
        constraints: OptimizationConstraints
    ) -> List[OptimizationPoint]:
        """Generate candidates using gradient-based optimization."""
        
        candidates = []
        
        if len(optimization_history) < 3:
            return candidates  # Need sufficient history for gradient estimation
        
        # Estimate gradients from recent history
        recent_points = optimization_history[-5:]  # Use last 5 points
        
        # Calculate quality gradient with respect to bitrate
        bitrate_gradient = self._estimate_quality_bitrate_gradient(recent_points)
        
        # Calculate size gradient with respect to bitrate
        size_gradient = self._estimate_size_bitrate_gradient(recent_points)
        
        # Generate candidates based on gradients
        if bitrate_gradient > 0:  # Quality improves with bitrate
            # Try higher bitrate if size allows
            if best_point.size_mb and best_point.size_mb < constraints.max_size_mb * 0.9:
                new_bitrate = min(
                    int(best_point.bitrate * 1.15),
                    parameter_space.bitrate_range[1]
                )
                candidate = OptimizationPoint(
                    bitrate=new_bitrate,
                    crf=best_point.crf,
                    preset=best_point.preset,
                    encoder=best_point.encoder,
                    resolution_factor=best_point.resolution_factor,
                    fps_factor=best_point.fps_factor
                )
                candidates.append(candidate)
        
        # Try CRF adjustment based on quality gradient
        if best_point.crf is not None and bitrate_gradient < 0.5:
            # If bitrate gradient is low, try CRF adjustment
            new_crf = max(
                best_point.crf - 1,
                parameter_space.crf_range[0]
            )
            candidate = OptimizationPoint(
                bitrate=best_point.bitrate,
                crf=new_crf,
                preset=best_point.preset,
                encoder=best_point.encoder,
                resolution_factor=best_point.resolution_factor,
                fps_factor=best_point.fps_factor
            )
            candidates.append(candidate)
        
        return candidates
    
    def _estimate_quality_bitrate_gradient(self, points: List[OptimizationPoint]) -> float:
        """Estimate the gradient of quality with respect to bitrate."""
        
        valid_points = [p for p in points if p.quality_score is not None and p.bitrate is not None]
        
        if len(valid_points) < 2:
            return 0.0
        
        # Simple linear regression to estimate gradient
        bitrates = np.array([p.bitrate for p in valid_points])
        qualities = np.array([p.quality_score for p in valid_points])
        
        if np.std(bitrates) == 0:
            return 0.0
        
        # Calculate correlation coefficient as gradient estimate
        correlation = np.corrcoef(bitrates, qualities)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.0
    
    def _estimate_size_bitrate_gradient(self, points: List[OptimizationPoint]) -> float:
        """Estimate the gradient of size with respect to bitrate."""
        
        valid_points = [p for p in points if p.size_mb is not None and p.bitrate is not None]
        
        if len(valid_points) < 2:
            return 1.0  # Default assumption: size increases with bitrate
        
        bitrates = np.array([p.bitrate for p in valid_points])
        sizes = np.array([p.size_mb for p in valid_points])
        
        if np.std(bitrates) == 0:
            return 1.0
        
        correlation = np.corrcoef(bitrates, sizes)[0, 1]
        
        return correlation if not np.isnan(correlation) else 1.0
    
    def _generate_constraint_guided_candidates(
        self,
        best_point: OptimizationPoint,
        parameter_space: ParameterSpace,
        constraints: OptimizationConstraints
    ) -> List[OptimizationPoint]:
        """Generate candidates guided by constraint satisfaction."""
        
        candidates = []
        
        # If size constraint is violated, generate size-reducing candidates
        if best_point.size_mb and best_point.size_mb > constraints.max_size_mb:
            # Reduce bitrate
            reduced_bitrate = max(
                int(best_point.bitrate * 0.8),
                parameter_space.bitrate_range[0]
            )
            candidate = OptimizationPoint(
                bitrate=reduced_bitrate,
                crf=best_point.crf,
                preset=best_point.preset,
                encoder=best_point.encoder,
                resolution_factor=best_point.resolution_factor,
                fps_factor=best_point.fps_factor
            )
            candidates.append(candidate)
            
            # Increase CRF (reduce quality for smaller size)
            if best_point.crf is not None:
                increased_crf = min(
                    best_point.crf + 2,
                    parameter_space.crf_range[1]
                )
                candidate = OptimizationPoint(
                    bitrate=best_point.bitrate,
                    crf=increased_crf,
                    preset=best_point.preset,
                    encoder=best_point.encoder,
                    resolution_factor=best_point.resolution_factor,
                    fps_factor=best_point.fps_factor
                )
                candidates.append(candidate)
            
            # Reduce resolution
            if best_point.resolution_factor > min(parameter_space.resolution_factors):
                available_factors = [f for f in parameter_space.resolution_factors if f < best_point.resolution_factor]
                if available_factors:
                    new_resolution_factor = max(available_factors)
                    candidate = OptimizationPoint(
                        bitrate=best_point.bitrate,
                        crf=best_point.crf,
                        preset=best_point.preset,
                        encoder=best_point.encoder,
                        resolution_factor=new_resolution_factor,
                        fps_factor=best_point.fps_factor
                    )
                    candidates.append(candidate)
        
        # If quality constraint is violated, generate quality-improving candidates
        if (best_point.quality_score and 
            best_point.quality_score < constraints.min_quality_vmaf):
            
            # Increase bitrate if size allows
            if not best_point.size_mb or best_point.size_mb < constraints.max_size_mb * 0.9:
                increased_bitrate = min(
                    int(best_point.bitrate * 1.2),
                    parameter_space.bitrate_range[1]
                )
                candidate = OptimizationPoint(
                    bitrate=increased_bitrate,
                    crf=best_point.crf,
                    preset=best_point.preset,
                    encoder=best_point.encoder,
                    resolution_factor=best_point.resolution_factor,
                    fps_factor=best_point.fps_factor
                )
                candidates.append(candidate)
            
            # Decrease CRF (increase quality)
            if best_point.crf is not None:
                decreased_crf = max(
                    best_point.crf - 2,
                    parameter_space.crf_range[0]
                )
                candidate = OptimizationPoint(
                    bitrate=best_point.bitrate,
                    crf=decreased_crf,
                    preset=best_point.preset,
                    encoder=best_point.encoder,
                    resolution_factor=best_point.resolution_factor,
                    fps_factor=best_point.fps_factor
                )
                candidates.append(candidate)
        
        return candidates
    
    def _generate_exploration_candidates(
        self,
        parameter_space: ParameterSpace,
        optimization_history: List[OptimizationPoint],
        current_iteration: int,
        max_iterations: int
    ) -> List[OptimizationPoint]:
        """Generate exploration candidates based on exploration vs exploitation balance."""
        
        candidates = []
        
        # Calculate exploration probability (higher early in optimization)
        exploration_prob = self.exploration_factor * (1.0 - current_iteration / max_iterations)
        
        if np.random.random() < exploration_prob:
            # Generate random exploration points
            num_exploration = max(1, int(3 * exploration_prob))
            exploration_points = self._generate_random_points(parameter_space, num_exploration)
            candidates.extend(exploration_points)
        
        # Diversification: explore underexplored regions
        if len(optimization_history) > 5:
            diverse_candidates = self._generate_diverse_candidates(
                parameter_space, optimization_history
            )
            candidates.extend(diverse_candidates)
        
        return candidates
    
    def _generate_diverse_candidates(
        self,
        parameter_space: ParameterSpace,
        optimization_history: List[OptimizationPoint]
    ) -> List[OptimizationPoint]:
        """Generate candidates in underexplored regions of parameter space."""
        
        candidates = []
        
        # Analyze explored bitrate ranges
        explored_bitrates = [p.bitrate for p in optimization_history if p.bitrate is not None]
        
        if explored_bitrates:
            min_explored = min(explored_bitrates)
            max_explored = max(explored_bitrates)
            
            # Explore outside current range if possible
            if min_explored > parameter_space.bitrate_range[0]:
                # Explore lower bitrates
                low_bitrate = max(
                    parameter_space.bitrate_range[0],
                    min_explored - (max_explored - min_explored) * 0.2
                )
                candidate = OptimizationPoint(
                    bitrate=int(low_bitrate),
                    crf=parameter_space.crf_range[0] + 2,
                    preset=parameter_space.preset_options[0] if parameter_space.preset_options else 'medium',
                    encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
                    resolution_factor=0.8,
                    fps_factor=0.9
                )
                candidates.append(candidate)
            
            if max_explored < parameter_space.bitrate_range[1]:
                # Explore higher bitrates
                high_bitrate = min(
                    parameter_space.bitrate_range[1],
                    max_explored + (max_explored - min_explored) * 0.2
                )
                candidate = OptimizationPoint(
                    bitrate=int(high_bitrate),
                    crf=parameter_space.crf_range[0],
                    preset=parameter_space.preset_options[-1] if parameter_space.preset_options else 'slow',
                    encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
                    resolution_factor=1.0,
                    fps_factor=1.0
                )
                candidates.append(candidate)
        
        return candidates
    
    def _filter_and_deduplicate_candidates(
        self,
        candidates: List[OptimizationPoint],
        optimization_history: List[OptimizationPoint],
        parameter_space: ParameterSpace
    ) -> List[OptimizationPoint]:
        """Filter invalid candidates and remove duplicates."""
        
        filtered_candidates = []
        seen_signatures = set()
        
        # Add signatures of already evaluated points
        for point in optimization_history:
            signature = self._get_point_signature(point)
            seen_signatures.add(signature)
        
        for candidate in candidates:
            # Check if candidate is valid
            if not self._is_valid_point(candidate, parameter_space):
                continue
            
            # Check for duplicates
            signature = self._get_point_signature(candidate)
            if signature in seen_signatures:
                continue
            
            seen_signatures.add(signature)
            filtered_candidates.append(candidate)
        
        return filtered_candidates
    
    def _get_point_signature(self, point: OptimizationPoint) -> str:
        """Get a unique signature for an optimization point."""
        return f"{point.bitrate}_{point.crf}_{point.preset}_{point.encoder}_{point.resolution_factor}_{point.fps_factor}"
    
    def _is_valid_point(self, point: OptimizationPoint, parameter_space: ParameterSpace) -> bool:
        """Check if an optimization point is valid within the parameter space."""
        
        # Check bitrate range
        if not (parameter_space.bitrate_range[0] <= point.bitrate <= parameter_space.bitrate_range[1]):
            return False
        
        # Check CRF range
        if point.crf is not None:
            if not (parameter_space.crf_range[0] <= point.crf <= parameter_space.crf_range[1]):
                return False
        
        # Check preset validity
        if parameter_space.preset_options and point.preset not in parameter_space.preset_options:
            return False
        
        # Check encoder validity
        if parameter_space.encoder_options and point.encoder not in parameter_space.encoder_options:
            return False
        
        # Check resolution factor
        if point.resolution_factor not in parameter_space.resolution_factors:
            return False
        
        # Check FPS factor
        if point.fps_factor not in parameter_space.fps_factors:
            return False
        
        return True
    
    def _evaluate_population_parallel(
        self,
        input_path: str,
        population: List[OptimizationPoint],
        constraints: OptimizationConstraints
    ) -> List[OptimizationPoint]:
        """Evaluate a population of optimization points in parallel."""
        
        if not population:
            return []
        
        logger.info(f"Evaluating population of {len(population)} points in parallel")
        
        evaluated_points = []
        
        # Use thread pool for parallel evaluation
        with ThreadPoolExecutor(max_workers=self.max_parallel_evaluations) as executor:
            
            # Submit evaluation tasks
            future_to_point = {
                executor.submit(self._evaluate_single_point, input_path, point, constraints): point
                for point in population
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_point):
                point = future_to_point[future]
                try:
                    evaluated_point = future.result(timeout=300)  # 5 minute timeout
                    if evaluated_point:
                        evaluated_points.append(evaluated_point)
                        logger.debug(f"Point evaluated: bitrate={evaluated_point.bitrate}, "
                                   f"quality={evaluated_point.quality_score}, "
                                   f"size={evaluated_point.size_mb}MB")
                except Exception as e:
                    logger.warning(f"Point evaluation failed: {e}")
                    # Mark point as infeasible
                    point.feasible = False
                    point.constraint_violations = [str(e)]
                    evaluated_points.append(point)
        
        return evaluated_points
    
    def _evaluate_single_point(
        self,
        input_path: str,
        point: OptimizationPoint,
        constraints: OptimizationConstraints
    ) -> OptimizationPoint:
        """Evaluate a single optimization point."""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_point_signature(point)
            if cache_key in self.optimization_cache:
                cached_result = self.optimization_cache[cache_key]
                logger.debug(f"Using cached result for point: {cache_key}")
                return cached_result
            
            # Simulate compression and quality evaluation
            # In a real implementation, this would call the actual compression and quality evaluation
            quality_score, size_mb = self._simulate_compression_evaluation(point)
            
            # Update point with results
            point.quality_score = quality_score
            point.size_mb = size_mb
            point.evaluation_time = time.time() - start_time
            
            # Check constraint satisfaction
            violations = []
            if size_mb > constraints.max_size_mb:
                violations.append(f"size_violation: {size_mb:.2f}MB > {constraints.max_size_mb:.2f}MB")
            
            if quality_score < constraints.min_quality_vmaf:
                violations.append(f"quality_violation: {quality_score:.1f} < {constraints.min_quality_vmaf:.1f}")
            
            point.constraint_violations = violations
            point.feasible = len(violations) == 0
            
            # Cache the result
            self.optimization_cache[cache_key] = point
            
            return point
            
        except Exception as e:
            logger.error(f"Error evaluating point: {e}")
            point.feasible = False
            point.constraint_violations = [str(e)]
            point.evaluation_time = time.time() - start_time
            return point
    
    def _simulate_compression_evaluation(self, point: OptimizationPoint) -> Tuple[float, float]:
        """
        Simulate compression and quality evaluation.
        
        In a real implementation, this would:
        1. Generate FFmpeg command with the point's parameters
        2. Execute compression
        3. Evaluate quality using VMAF/SSIM
        4. Measure output file size
        
        For now, we simulate with a realistic model.
        """
        
        # Simulate quality based on parameters
        base_quality = 75.0  # Base VMAF score
        
        # Bitrate impact on quality (logarithmic relationship)
        bitrate_factor = math.log(point.bitrate / 1000.0) * 8.0
        
        # CRF impact on quality (inverse relationship)
        crf_factor = 0.0
        if point.crf is not None:
            crf_factor = (28 - point.crf) * 2.0  # Lower CRF = higher quality
        
        # Preset impact on quality
        preset_quality_map = {
            'ultrafast': -5.0, 'superfast': -3.0, 'veryfast': -2.0,
            'faster': -1.0, 'fast': 0.0, 'medium': 1.0,
            'slow': 2.0, 'slower': 3.0, 'veryslow': 4.0
        }
        preset_factor = preset_quality_map.get(point.preset, 0.0)
        
        # Encoder impact on quality
        encoder_factor = 2.0 if point.encoder == 'libx265' else 0.0
        
        # Resolution impact on quality
        resolution_factor = (point.resolution_factor - 1.0) * 10.0
        
        # Calculate final quality score
        quality_score = (base_quality + bitrate_factor + crf_factor + 
                        preset_factor + encoder_factor + resolution_factor)
        quality_score = max(20.0, min(100.0, quality_score))  # Clamp to valid range
        
        # Simulate size based on parameters
        base_size_mb = 10.0  # Base size for reference
        
        # Bitrate is the primary size factor
        size_factor = point.bitrate / 2000.0  # Normalize to ~2Mbps baseline
        
        # CRF impact on size (inverse relationship)
        if point.crf is not None:
            size_factor *= math.exp((point.crf - 23) * -0.1)
        
        # Resolution impact on size
        size_factor *= (point.resolution_factor ** 2)  # Quadratic relationship
        
        # FPS impact on size
        size_factor *= point.fps_factor
        
        # Encoder efficiency
        if point.encoder == 'libx265':
            size_factor *= 0.7  # HEVC is more efficient
        
        size_mb = base_size_mb * size_factor
        
        # Add some realistic noise
        quality_score += np.random.normal(0, 1.0)
        size_mb += np.random.normal(0, size_mb * 0.05)
        
        return max(0, quality_score), max(0.1, size_mb)
    
    def _select_best_point(
        self,
        points: List[OptimizationPoint],
        constraints: OptimizationConstraints
    ) -> Optional[OptimizationPoint]:
        """Select the best point from a list based on multi-objective optimization."""
        
        if not points:
            return None
        
        # Filter feasible points first
        feasible_points = [p for p in points if p.feasible and p.quality_score is not None and p.size_mb is not None]
        
        if not feasible_points:
            # If no feasible points, select the least violating one
            return min(points, key=lambda p: len(p.constraint_violations or []))
        
        # Multi-objective optimization: maximize quality while respecting size constraint
        def objective_function(point: OptimizationPoint) -> float:
            quality_score = point.quality_score or 0.0
            size_mb = point.size_mb or float('inf')
            
            # Primary objective: quality
            score = quality_score
            
            # Penalty for size constraint violation
            if size_mb > constraints.max_size_mb:
                size_penalty = (size_mb - constraints.max_size_mb) * 10.0
                score -= size_penalty
            
            # Bonus for efficient size utilization
            size_utilization = size_mb / constraints.max_size_mb
            if 0.8 <= size_utilization <= 1.0:
                score += 5.0  # Bonus for good size utilization
            
            return score
        
        best_point = max(feasible_points, key=objective_function)
        return best_point
    
    def _calculate_improvement(
        self,
        old_point: Optional[OptimizationPoint],
        new_point: Optional[OptimizationPoint],
        constraints: OptimizationConstraints
    ) -> float:
        """Calculate improvement between two points."""
        
        if not old_point or not new_point:
            return 0.0
        
        if not old_point.quality_score or not new_point.quality_score:
            return 0.0
        
        quality_improvement = new_point.quality_score - old_point.quality_score
        
        # Consider size efficiency in improvement calculation
        if old_point.size_mb and new_point.size_mb:
            old_efficiency = old_point.quality_score / old_point.size_mb
            new_efficiency = new_point.quality_score / new_point.size_mb
            efficiency_improvement = (new_efficiency - old_efficiency) * 10.0
            
            return quality_improvement + efficiency_improvement
        
        return quality_improvement
    
    def _check_convergence(
        self,
        optimization_history: List[OptimizationPoint],
        constraints: OptimizationConstraints,
        current_iteration: int
    ) -> ConvergenceStatus:
        """Check convergence status of the optimization."""
        
        if current_iteration >= constraints.max_iterations:
            return ConvergenceStatus.MAX_ITERATIONS
        
        if len(optimization_history) < self.convergence_patience + 1:
            return ConvergenceStatus.IMPROVING
        
        # Check recent improvements
        recent_points = optimization_history[-self.convergence_patience:]
        quality_scores = [p.quality_score for p in recent_points if p.quality_score is not None]
        
        if len(quality_scores) < self.convergence_patience:
            return ConvergenceStatus.IMPROVING
        
        # Check if improvements are below threshold
        max_recent_improvement = 0.0
        for i in range(1, len(quality_scores)):
            improvement = quality_scores[i] - quality_scores[i-1]
            max_recent_improvement = max(max_recent_improvement, improvement)
        
        if max_recent_improvement < constraints.convergence_threshold:
            # Check if we're stagnant or converged
            quality_variance = np.var(quality_scores)
            if quality_variance < 0.1:
                return ConvergenceStatus.CONVERGED
            else:
                return ConvergenceStatus.STAGNANT
        
        # Check for divergence (quality getting worse)
        if len(quality_scores) >= 3:
            recent_trend = quality_scores[-1] - quality_scores[-3]
            if recent_trend < -constraints.quality_improvement_threshold:
                return ConvergenceStatus.DIVERGING
        
        return ConvergenceStatus.IMPROVING
    
    def _satisfies_constraints(
        self,
        point: OptimizationPoint,
        constraints: OptimizationConstraints
    ) -> bool:
        """Check if a point satisfies all constraints."""
        
        if not point or not point.feasible:
            return False
        
        if point.size_mb and point.size_mb > constraints.max_size_mb:
            return False
        
        if point.quality_score and point.quality_score < constraints.min_quality_vmaf:
            return False
        
        return True
    
    def _apply_fallback_strategy(
        self,
        optimization_history: List[OptimizationPoint],
        constraints: OptimizationConstraints,
        parameter_space: ParameterSpace
    ) -> OptimizationPoint:
        """Apply fallback strategy when optimization fails to find satisfactory solution."""
        
        logger.info("Applying fallback strategy")
        
        # Strategy 1: Find best compromise from history
        if optimization_history:
            # Find point with best quality that satisfies size constraint
            size_compliant_points = [
                p for p in optimization_history 
                if p.size_mb and p.size_mb <= constraints.max_size_mb and p.quality_score is not None
            ]
            
            if size_compliant_points:
                best_compromise = max(size_compliant_points, key=lambda p: p.quality_score)
                logger.info(f"Fallback: Using best size-compliant point with quality {best_compromise.quality_score:.1f}")
                return best_compromise
            
            # If no size-compliant points, find point closest to size constraint
            points_with_size = [p for p in optimization_history if p.size_mb is not None]
            if points_with_size:
                closest_to_constraint = min(
                    points_with_size,
                    key=lambda p: abs(p.size_mb - constraints.max_size_mb)
                )
                logger.info(f"Fallback: Using point closest to size constraint ({closest_to_constraint.size_mb:.2f}MB)")
                return closest_to_constraint
        
        # Strategy 2: Generate conservative fallback point
        logger.info("Fallback: Generating conservative point")
        fallback_point = OptimizationPoint(
            bitrate=parameter_space.bitrate_range[0],
            crf=parameter_space.crf_range[1] - 2,  # Conservative CRF
            preset=parameter_space.preset_options[0] if parameter_space.preset_options else 'fast',
            encoder=parameter_space.encoder_options[0] if parameter_space.encoder_options else 'libx264',
            resolution_factor=min(parameter_space.resolution_factors),
            fps_factor=min(parameter_space.fps_factors)
        )
        
        # Simulate evaluation for fallback point
        quality_score, size_mb = self._simulate_compression_evaluation(fallback_point)
        fallback_point.quality_score = quality_score
        fallback_point.size_mb = size_mb
        fallback_point.feasible = size_mb <= constraints.max_size_mb
        
        return fallback_point
    
    def _evaluate_constraint_satisfaction(
        self,
        point: OptimizationPoint,
        constraints: OptimizationConstraints
    ) -> Dict[str, bool]:
        """Evaluate constraint satisfaction for a point."""
        
        satisfaction = {}
        
        if point.size_mb is not None:
            satisfaction['size_constraint'] = point.size_mb <= constraints.max_size_mb
        else:
            satisfaction['size_constraint'] = False
        
        if point.quality_score is not None:
            satisfaction['quality_constraint'] = point.quality_score >= constraints.min_quality_vmaf
        else:
            satisfaction['quality_constraint'] = False
        
        satisfaction['overall_feasible'] = point.feasible
        
        return satisfaction
    
    def _calculate_total_improvement(
        self,
        initial_point: Optional[OptimizationPoint],
        final_point: Optional[OptimizationPoint],
        constraints: OptimizationConstraints
    ) -> float:
        """Calculate total improvement achieved during optimization."""
        
        if not initial_point or not final_point:
            return 0.0
        
        if not initial_point.quality_score or not final_point.quality_score:
            return 0.0
        
        return final_point.quality_score - initial_point.quality_score
    
    def _generate_optimization_summary(
        self,
        optimization_history: List[OptimizationPoint],
        best_point: OptimizationPoint,
        constraints: OptimizationConstraints,
        total_time: float
    ) -> Dict[str, Any]:
        """Generate comprehensive optimization summary."""
        
        summary = {
            'total_evaluations': len(optimization_history),
            'successful_evaluations': len([p for p in optimization_history if p.feasible]),
            'best_quality_achieved': best_point.quality_score if best_point else None,
            'best_size_achieved': best_point.size_mb if best_point else None,
            'constraint_satisfaction_rate': 0.0,
            'average_evaluation_time': 0.0,
            'quality_progression': [],
            'size_progression': [],
            'pareto_frontier_size': len(self.pareto_frontier)
        }
        
        if optimization_history:
            feasible_points = [p for p in optimization_history if p.feasible]
            summary['constraint_satisfaction_rate'] = len(feasible_points) / len(optimization_history)
            
            evaluation_times = [p.evaluation_time for p in optimization_history if p.evaluation_time is not None]
            if evaluation_times:
                summary['average_evaluation_time'] = np.mean(evaluation_times)
            
            # Track quality and size progression
            for point in optimization_history:
                if point.quality_score is not None:
                    summary['quality_progression'].append(point.quality_score)
                if point.size_mb is not None:
                    summary['size_progression'].append(point.size_mb)
        
        return summary