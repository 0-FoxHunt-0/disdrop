"""
Command Line Interface for Video Compressor
Main entry point with argument parsing and command execution
"""

import argparse
import atexit
import os
import sys
import shutil
import signal
import traceback
import threading
import subprocess
from typing import Dict, Any, Optional, List, Set
import logging
import time

from .logger_setup import setup_logging, get_logger, _cleanup_old_logs, get_package_base_dir
from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .video_processing.video_compressor import DynamicVideoCompressor
from .gif_processing.gif_generator import GifGenerator
from .gif_processing.gif_optimizer import GifOptimizer
from .automated_workflow import AutomatedWorkflow
from .file_validator import FileValidator
from .video_processing.bitrate_validator import BitrateValidationError

logger = None  # Will be initialized after logging setup

class VideoCompressorCLI:
    def __init__(self):
        self.config = None
        self.hardware = None
        self.video_compressor = None
        self.gif_generator = None
        self.automated_workflow = None
        # Shutdown tracking
        self.shutdown_requested = False
        self.shutdown_lock = threading.Lock()
        self.graceful_shutdown_timeout = 5.0  # seconds - reduced for faster response
        self.graceful_shutdown_start_time = None
        
    def main(self):
        """Main entry point"""
        try:
            # Parse arguments first to get log level
            args = self._parse_arguments()
            
            # Clean up old logs at startup (keep only last 5 executions)
            _cleanup_old_logs("logs", keep_count=5)
            
            # Setup logging (quiet console by default; enable verbose console when --debug)
            global logger
            effective_level = 'DEBUG' if getattr(args, 'debug', False) else args.log_level
            logger = setup_logging(log_level=effective_level)

            # Clear failures directory at startup to avoid mix-ups/clutter
            self._clear_failures_directory()
            
            # Setup signal handlers for graceful cleanup
            self._setup_signal_handlers()
            
            # Register atexit handler for cleanup
            atexit.register(self._cleanup_temp_files_on_exit)
            
            # Initialize components
            self._initialize_components(args)
            
            # Execute command
            self._execute_command(args)
            
        except KeyboardInterrupt:
            if logger:
                logger.info("Operation cancelled by user")
            sys.exit(1)
        except Exception as e:
            if logger:
                logger.error(f"Unexpected error: {e}")
                logger.debug(traceback.format_exc())
            else:
                print(f"Error: {e}")
            sys.exit(1)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful cleanup"""
        # Track signal count for force exit
        signal_count = {'count': 0}
        # Track if monitoring thread is running
        monitoring_active = {'active': False}
        
        def signal_handler(signum, frame):
            signal_count['count'] += 1
            
            signal_name = None
            try:
                signal_name = signal.Signals(signum).name
            except Exception:
                signal_name = str(signum)
            
            # Force kill all processes on second signal (or if already shutting down)
            if signal_count['count'] >= 2:
                print(f"\n{signal_name} received {signal_count['count']} times. Force killing all processes and exiting...")
                if logger:
                    logger.warning(f"{signal_name} received {signal_count['count']} times. Force killing all processes and exiting.")
                # Kill all processes instantly
                self._kill_all_processes_instantly()
                sys.exit(1)
            
            # On first signal, initiate graceful shutdown with immediate process termination
            with self.shutdown_lock:
                if not self.shutdown_requested:
                    self.shutdown_requested = True
                    self.graceful_shutdown_start_time = time.time()
                    
                    if logger:
                        logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
                    else:
                        print(f"\nReceived {signal_name} signal, cleaning up... (Press Ctrl+C again to force quit)")

                    # Immediately terminate all processes for faster shutdown
                    try:
                        self._kill_all_processes_instantly()
                    except Exception as e:
                        if logger:
                            logger.debug(f"Error during immediate process termination: {e}")

                    # Request shutdown for all components (idempotent)
                    try:
                        if hasattr(self, 'gif_generator') and self.gif_generator:
                            self.gif_generator.request_shutdown()
                            if hasattr(self.gif_generator, 'optimizer') and hasattr(self.gif_generator.optimizer, 'request_shutdown'):
                                self.gif_generator.optimizer.request_shutdown()
                    except Exception:
                        pass
                    try:
                        if hasattr(self, 'video_compressor') and self.video_compressor:
                            self.video_compressor.request_shutdown()
                            # Also request shutdown for video_segmenter if it exists
                            if hasattr(self.video_compressor, 'video_segmenter') and self.video_compressor.video_segmenter:
                                if hasattr(self.video_compressor.video_segmenter, 'request_shutdown'):
                                    self.video_compressor.video_segmenter.request_shutdown()
                    except Exception:
                        pass

                    # Set shutdown flag for automated workflow if it exists
                    if hasattr(self, 'automated_workflow') and self.automated_workflow:
                        self.automated_workflow.shutdown_requested = True

                    if logger:
                        logger.info("Graceful shutdown sequence initiated. Waiting for components to finish...")
                    
                    # Start graceful shutdown timeout monitoring thread
                    def monitor_graceful_shutdown():
                        """Monitor graceful shutdown and exit when complete or timeout exceeded"""
                        monitoring_active['active'] = True
                        try:
                            while True:
                                time.sleep(0.2)  # Check every 200ms for faster response
                                
                                # Check shutdown state and timeout
                                with self.shutdown_lock:
                                    if not self.shutdown_requested:
                                        break
                                    if self.graceful_shutdown_start_time is None:
                                        break
                                    elapsed = time.time() - self.graceful_shutdown_start_time
                                    if elapsed > self.graceful_shutdown_timeout:
                                        # Timeout exceeded, force kill all processes
                                        if logger:
                                            logger.warning(f"Graceful shutdown timeout ({self.graceful_shutdown_timeout}s) exceeded. Force killing all processes...")
                                        print(f"\nGraceful shutdown timeout exceeded. Force killing all processes...")
                                        self._kill_all_processes_instantly()
                                        # Use os._exit to force exit from background thread
                                        os._exit(1)
                                
                                # Check if all processes have terminated (outside lock for thread safety)
                                # This is safe because _collect_all_processes() is read-only
                                processes = self._collect_all_processes()
                                if not processes:
                                    # All processes terminated, graceful shutdown complete
                                    # Exit the program now
                                    if logger:
                                        logger.info("All processes terminated, graceful shutdown complete. Exiting...")
                                    print("\n✅ Graceful shutdown complete. Exiting...")
                                    # Use os._exit to ensure immediate exit
                                    os._exit(0)
                        finally:
                            monitoring_active['active'] = False
                    
                    # Start monitoring thread
                    if not monitoring_active['active']:
                        monitor_thread = threading.Thread(target=monitor_graceful_shutdown, daemon=False)
                        monitor_thread.start()
                else:
                    # Already shutting down, second signal means force quit
                    print(f"\n{signal_name} received again. Force killing all processes and exiting...")
                    if logger:
                        logger.warning(f"{signal_name} received again during shutdown. Force killing all processes...")
                    self._kill_all_processes_instantly()
                    sys.exit(1)

        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    def _collect_all_processes(self) -> Set[subprocess.Popen]:
        """Collect all subprocess.Popen objects from all components"""
        processes: Set[subprocess.Popen] = set()
        
        try:
            # Collect from video_compressor
            if hasattr(self, 'video_compressor') and self.video_compressor:
                # Main video compressor processes
                if hasattr(self.video_compressor, '_ffmpeg_processes'):
                    for proc in self.video_compressor._ffmpeg_processes:
                        if proc and proc.poll() is None:
                            processes.add(proc)
                if hasattr(self.video_compressor, 'current_ffmpeg_process'):
                    proc = self.video_compressor.current_ffmpeg_process
                    if proc and proc.poll() is None:
                        processes.add(proc)
                
                # Video segmenter processes
                if hasattr(self.video_compressor, 'video_segmenter') and self.video_compressor.video_segmenter:
                    segmenter = self.video_compressor.video_segmenter
                    if hasattr(segmenter, '_active_processes'):
                        for proc in segmenter._active_processes:
                            if proc and proc.poll() is None:
                                processes.add(proc)
                    if hasattr(segmenter, 'current_ffmpeg_process'):
                        proc = segmenter.current_ffmpeg_process
                        if proc and proc.poll() is None:
                            processes.add(proc)
                
                # Advanced optimizer processes (if accessible)
                if hasattr(self.video_compressor, 'advanced_optimizer') and self.video_compressor.advanced_optimizer:
                    optimizer = self.video_compressor.advanced_optimizer
                    if hasattr(optimizer, 'time_budget') and optimizer.time_budget:
                        time_budget = optimizer.time_budget
                        if hasattr(time_budget, 'current_process'):
                            proc = time_budget.current_process
                            if proc and proc.poll() is None:
                                processes.add(proc)
        except Exception as e:
            if logger:
                logger.debug(f"Error collecting processes from video_compressor: {e}")
        
        try:
            # Collect from gif_generator
            if hasattr(self, 'gif_generator') and self.gif_generator:
                # Main GIF generator processes
                if hasattr(self.gif_generator, '_ffmpeg_processes'):
                    for proc in self.gif_generator._ffmpeg_processes:
                        if proc and proc.poll() is None:
                            processes.add(proc)
                if hasattr(self.gif_generator, 'current_ffmpeg_process'):
                    proc = self.gif_generator.current_ffmpeg_process
                    if proc and proc.poll() is None:
                        processes.add(proc)
                
                # GIF optimizer processes
                if hasattr(self.gif_generator, 'optimizer') and self.gif_generator.optimizer:
                    optimizer = self.gif_generator.optimizer
                    if hasattr(optimizer, 'current_ffmpeg_process'):
                        proc = optimizer.current_ffmpeg_process
                        if proc and proc.poll() is None:
                            processes.add(proc)
        except Exception as e:
            if logger:
                logger.debug(f"Error collecting processes from gif_generator: {e}")
        
        try:
            # Collect from automated_workflow (if it has its own instances)
            if hasattr(self, 'automated_workflow') and self.automated_workflow:
                # Workflow's video_compressor
                if hasattr(self.automated_workflow, 'video_compressor') and self.automated_workflow.video_compressor:
                    vc = self.automated_workflow.video_compressor
                    if hasattr(vc, '_ffmpeg_processes'):
                        for proc in vc._ffmpeg_processes:
                            if proc and proc.poll() is None:
                                processes.add(proc)
                    if hasattr(vc, 'current_ffmpeg_process'):
                        proc = vc.current_ffmpeg_process
                        if proc and proc.poll() is None:
                            processes.add(proc)
                    # Workflow's video_compressor's video_segmenter
                    if hasattr(vc, 'video_segmenter') and vc.video_segmenter:
                        segmenter = vc.video_segmenter
                        if hasattr(segmenter, '_active_processes'):
                            for proc in segmenter._active_processes:
                                if proc and proc.poll() is None:
                                    processes.add(proc)
                        if hasattr(segmenter, 'current_ffmpeg_process'):
                            proc = segmenter.current_ffmpeg_process
                            if proc and proc.poll() is None:
                                processes.add(proc)
                
                # Workflow's gif_generator
                if hasattr(self.automated_workflow, 'gif_generator') and self.automated_workflow.gif_generator:
                    gg = self.automated_workflow.gif_generator
                    if hasattr(gg, '_ffmpeg_processes'):
                        for proc in gg._ffmpeg_processes:
                            if proc and proc.poll() is None:
                                processes.add(proc)
                    if hasattr(gg, 'current_ffmpeg_process'):
                        proc = gg.current_ffmpeg_process
                        if proc and proc.poll() is None:
                            processes.add(proc)
                    if hasattr(gg, 'optimizer') and gg.optimizer:
                        optimizer = gg.optimizer
                        if hasattr(optimizer, 'current_ffmpeg_process'):
                            proc = optimizer.current_ffmpeg_process
                            if proc and proc.poll() is None:
                                processes.add(proc)
        except Exception as e:
            if logger:
                logger.debug(f"Error collecting processes from automated_workflow: {e}")
        
        return processes
    
    def _kill_all_processes_instantly(self):
        """Kill all collected processes instantly without graceful termination"""
        processes = self._collect_all_processes()
        
        if not processes:
            if logger:
                logger.debug("No active processes to kill")
            return
        
        if logger:
            logger.warning(f"Force killing {len(processes)} process(es) instantly...")
        
        killed_count = 0
        for proc in processes:
            try:
                if proc and proc.poll() is None:
                    proc.kill()
                    killed_count += 1
                    # Wait briefly to ensure kill signal is sent
                    try:
                        proc.wait(timeout=0.5)
                    except (subprocess.TimeoutExpired, OSError):
                        pass
            except (OSError, subprocess.SubprocessError) as e:
                if logger:
                    logger.debug(f"Error killing process {proc}: {e}")
        
        if logger:
            logger.warning(f"Force killed {killed_count} process(es)")
    
    def _cleanup_temp_files_on_exit(self):
        """Clean up temporary files on program exit"""
        try:
            if hasattr(self, 'config') and self.config:
                temp_dir = self.config.get_temp_dir()
                if temp_dir and os.path.exists(temp_dir):
                    # Look for all types of temp files and folders
                    for item in os.listdir(temp_dir):
                        item_path = os.path.join(temp_dir, item)
                        try:
                            # Clean up segment temp folders
                            if os.path.isdir(item_path) and ('_segments_temp' in item or '_segments' in item):
                                shutil.rmtree(item_path)
                                if logger:
                                    logger.info(f"Cleaned up temp segment folder: {item}")
                            # Clean up temp files
                            elif os.path.isfile(item_path) and (
                                'temp_' in item or
                                item.startswith('candidate_') or
                                item.endswith('.tmp') or
                                item.startswith('temp') or
                                'temp' in item.lower()
                            ):
                                os.remove(item_path)
                                if logger:
                                    logger.debug(f"Cleaned up temp file: {item}")
                            # Clean up palette cache files
                            elif os.path.isfile(item_path) and 'palette' in item.lower():
                                os.remove(item_path)
                                if logger:
                                    logger.debug(f"Cleaned up palette file: {item}")
                        except Exception as e:
                            if logger:
                                logger.warning(f"Could not clean up {item}: {e}")

                    # Also check for any .tmp files that might be missed
                    import time as time_module
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('.tmp'):
                                file_path = os.path.join(root, file)
                                try:
                                    # On Windows, files may be locked briefly after subprocess closes
                                    # Retry with small delays to handle file locking
                                    max_retries = 3
                                    retry_delay = 0.1  # 100ms
                                    
                                    for attempt in range(max_retries):
                                        try:
                                            os.remove(file_path)
                                            if logger:
                                                logger.debug(f"Cleaned up .tmp file: {file}")
                                            break
                                        except (OSError, PermissionError) as e:
                                            if attempt < max_retries - 1:
                                                time_module.sleep(retry_delay * (attempt + 1))
                                            else:
                                                # Last attempt failed, log but don't raise
                                                if logger:
                                                    logger.warning(f"Could not clean up .tmp file {file}: {e}")
                                except Exception as e:
                                    if logger:
                                        logger.warning(f"Could not clean up .tmp file {file}: {e}")
        except Exception as e:
            if logger:
                logger.warning(f"Error during temp file cleanup: {e}")
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Video Compressor - Compress videos and create GIFs for social media platforms",
            epilog="Examples:\n"
                   "  %(prog)s c input.mp4 output.mp4 -p instagram\n"
                   "  %(prog)s c \"*.mp4\" -o compressed/ -p tiktok -j 4\n"
                   "  %(prog)s g input.mp4 output.gif -p twitter -d 10\n"
                   "  %(prog)s g \"*.mp4\" -o gifs/ -p discord -j 3\n"
                   "  %(prog)s w --check-interval 10 -s 8\n"
                   "  %(prog)s hw\n\n"
                   "Smart Features:\n"
                   "  - Auto-detects batch mode from glob patterns (*.mp4)\n"
                   "  - Unified 'gif' command auto-detects optimization type\n"
                   "  - Short aliases: c/v (compress), g/a (gif), w/m (watch/auto), hw/i (info)\n",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Global options
        # Default config-dir to package base config folder
        # Default config dir now points to packaged defaults inside the installed package
        default_config_dir = os.path.join(get_package_base_dir(), 'config')
        parser.add_argument('--config-dir', default=default_config_dir,
                          help='Configuration directory (default: config)')
        # Default to quieter console; use --debug to enable verbose output
        parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default=None, help='Override logging level (default: WARNING to console, DEBUG to file)')
        parser.add_argument('-v', '--debug', action='store_true',
                          help='Enable verbose debug output in console and logs')
        parser.add_argument('--temp-dir', help='Temporary directory for processing')
        parser.add_argument('-s', '--max-size', type=float, metavar='MB',
                          help='Maximum output file size in MB (overrides platform defaults)')
        parser.add_argument('--max-files', type=int, metavar='N',
                          help='Maximum number of files to process before exiting')
        parser.add_argument('-i', '--input-dir', help='Input directory for automated/other modes (default: ./input)')
        parser.add_argument('-o', '--output-dir', help='Output directory for generated files (default varies by mode, typically ./output)')
        parser.add_argument('--force-software', action='store_true',
                          help='Force software encoding (bypass hardware acceleration)')
        parser.add_argument('--no-cache', action='store_true',
                          help='Do not use success cache; verify and process all files even if previously successful')
        parser.add_argument('--max-input-size', dest='max_input_size', metavar='SIZE',
                          help='Maximum input file size to process (e.g., 500, 750MB, 1.2GB, 2TB). Bare numbers are MB.')
        # Segmentation preference: prefer a fixed number of segments (1-10)
        parser.add_argument('--prefer-segments', type=int, choices=list(range(1, 11)), metavar='N',
                          help='Prefer N segments (1-10). If impossible, fall back to normal operations')
        
        # Bitrate validation options
        parser.add_argument('--bitrate-floor', type=int, metavar='KBPS',
                          help='Override minimum bitrate floor for all encoders (in kbps)')
        parser.add_argument('--encoder-bitrate-floor', action='append', nargs=2, 
                          metavar=('ENCODER', 'KBPS'),
                          help='Override minimum bitrate floor for specific encoder (e.g., --encoder-bitrate-floor libx264 5)')
        parser.add_argument('--disable-bitrate-validation', action='store_true',
                          help='Disable bitrate validation and allow encoding below encoder minimums')
        parser.add_argument('--bitrate-safety-margin', type=float, metavar='MULTIPLIER',
                          help='Safety margin multiplier for bitrate calculations (default: 1.1)')
        parser.add_argument('--min-resolution', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                          help='Override minimum resolution constraints (e.g., --min-resolution 320 180)')
        parser.add_argument('--min-fps', type=float, metavar='FPS',
                          help='Override minimum FPS constraint (default: 10)')
        
        # Quality evaluation debugging options
        parser.add_argument('--quality-fallback-mode', choices=['conservative', 'permissive', 'strict'],
                          help='Quality evaluation fallback mode when evaluation fails')
        parser.add_argument('--quality-debug', action='store_true',
                          help='Enable detailed quality evaluation debugging (parsing attempts, performance metrics)')
        parser.add_argument('--quality-timeout', type=int, metavar='SECONDS',
                          help='Timeout for quality evaluation operations (default: 300)')
        parser.add_argument('--quality-retry-attempts', type=int, metavar='N',
                          help='Number of retry attempts for failed quality operations (default: 3)')
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Compress video command (with aliases: c, v, video)
        compress_parser = subparsers.add_parser('compress', aliases=['c', 'v', 'video'],
                                               help='Compress video file(s) - auto-detects batch mode from glob patterns')
        compress_parser.add_argument('input', help='Input video file or glob pattern (e.g., *.mp4 for batch)')
        compress_parser.add_argument('output', nargs='?', help='Output video file (optional for batch mode)')
        compress_parser.add_argument('-p', '--platform', choices=['instagram', 'twitter', 'tiktok', 'youtube_shorts', 'facebook'],
                                   help='Target social media platform')
        compress_parser.add_argument('--encoder', help='Force specific encoder (overrides hardware detection)')
        compress_parser.add_argument('-q', '--quality', type=int, metavar='CRF', help='Quality setting (CRF value, lower = better)')
        compress_parser.add_argument('--bitrate', help='Target bitrate (e.g., 1000k)')
        compress_parser.add_argument('--resolution', help='Target resolution (e.g., 1080x1080)')
        compress_parser.add_argument('-f', '--fps', type=int, help='Target frame rate')
        # Batch mode options (used when glob pattern detected)
        compress_parser.add_argument('--suffix', default='_compressed', help='Suffix for batch output files (default: _compressed)')
        compress_parser.add_argument('-j', '--parallel', type=int, metavar='N', help='Number of parallel processes for batch mode')
        
        # Unified GIF command (with aliases: g, a, anim) - merges gif, quality-gif, optimize-gif
        gif_parser = subparsers.add_parser('gif', aliases=['g', 'a', 'anim'],
                                          help='Create/optimize GIF(s) - auto-detects operation type and batch mode')
        gif_parser.add_argument('input', help='Input video/GIF file or glob pattern (e.g., *.mp4 for batch)')
        gif_parser.add_argument('output', nargs='?', help='Output GIF file (optional for batch mode)')
        gif_parser.add_argument('-p', '--platform', choices=['twitter', 'discord', 'slack'],
                               help='Target platform for GIF optimization')
        gif_parser.add_argument('--start', type=float, default=0, metavar='SECONDS',
                               help='Start time in seconds (default: 0)')
        gif_parser.add_argument('-d', '--duration', type=float, metavar='SECONDS',
                               help='Duration in seconds (default: platform/config limit)')
        gif_parser.add_argument('-s', '--max-size', type=float, metavar='MB',
                               help='Maximum file size in MB (enables quality optimization)')
        gif_parser.add_argument('--target-size', type=float, metavar='MB',
                               help='Target file size in MB for quality-optimized GIF (triggers advanced optimization)')
        gif_parser.add_argument('--quality-preference', choices=['quality', 'balanced', 'size'],
                               default='balanced', help='Optimization strategy when using --target-size (default: balanced)')
        gif_parser.add_argument('-f', '--fps', type=int, help='Frame rate for GIF')
        gif_parser.add_argument('--width', type=int, help='GIF width in pixels')
        gif_parser.add_argument('--height', type=int, help='GIF height in pixels')
        gif_parser.add_argument('--colors', type=int, help='Number of colors in GIF palette')
        # Batch mode options (used when glob pattern detected)
        gif_parser.add_argument('-j', '--parallel', type=int, metavar='N', help='Number of parallel processes for batch mode')
        
        # Batch compress command (kept for backward compatibility, hidden)
        batch_parser = subparsers.add_parser('batch-compress', help=argparse.SUPPRESS)
        batch_parser.add_argument('input_pattern', help='Input files pattern (e.g., *.mp4)')
        batch_parser.add_argument('--output-dir', help='Output directory (overrides global)')
        batch_parser.add_argument('--platform', choices=['instagram', 'twitter', 'tiktok', 'youtube_shorts', 'facebook'],
                                 help='Target social media platform')
        batch_parser.add_argument('--suffix', default='_compressed', help='Suffix for output files')
        batch_parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel processes')
        
        # Batch GIF command (kept for backward compatibility, hidden)
        batch_gif_parser = subparsers.add_parser('batch-gif', help=argparse.SUPPRESS)
        batch_gif_parser.add_argument('input_pattern', help='Input files pattern (e.g., *.mp4)')
        batch_gif_parser.add_argument('--output-dir', help='Output directory (overrides global)')
        batch_gif_parser.add_argument('--platform', choices=['twitter', 'discord', 'slack'],
                                     help='Target platform for GIF optimization')
        batch_gif_parser.add_argument('--duration', type=float, metavar='SECONDS',
                                     help='Duration for each GIF')
        batch_gif_parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel processes')
        
        # Optimize existing GIF (kept for backward compatibility, hidden - now handled by unified gif command)
        optimize_parser = subparsers.add_parser('optimize-gif', help=argparse.SUPPRESS)
        optimize_parser.add_argument('input', help='Input GIF file')
        optimize_parser.add_argument('output', help='Output GIF file')
        
        # Quality-optimized GIF command (kept for backward compatibility, hidden - now handled by unified gif command)
        quality_gif_parser = subparsers.add_parser('quality-gif', help=argparse.SUPPRESS)
        quality_gif_parser.add_argument('input', help='Input video file')
        quality_gif_parser.add_argument('output', help='Output GIF file')
        quality_gif_parser.add_argument('--platform', choices=['twitter', 'discord', 'slack'],
                                       help='Target platform for GIF optimization')
        quality_gif_parser.add_argument('--start', type=float, default=0, metavar='SECONDS',
                                       help='Start time in seconds (default: 0)')
        quality_gif_parser.add_argument('--duration', type=float, metavar='SECONDS',
                                       help='Duration in seconds (default: platform/config limit)')
        quality_gif_parser.add_argument('--target-size', type=float, required=True, metavar='MB',
                                       help='Target file size in MB (will optimize to get close to this)')
        quality_gif_parser.add_argument('--quality-preference', choices=['quality', 'balanced', 'size'],
                                       default='balanced', help='Optimization strategy (default: balanced)')
        
        # Hardware info command (with aliases: hw, i, info)
        subparsers.add_parser('hardware-info', aliases=['hw', 'i', 'info'],
                             help='Display hardware acceleration information')
        
        # Config command (with alias: cfg)
        config_parser = subparsers.add_parser('config', aliases=['cfg'],
                                             help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        config_subparsers.add_parser('show', help='Show current configuration')
        config_subparsers.add_parser('validate', help='Validate configuration files')
        
        # Automated workflow command (with aliases: w, watch, m, monitor)
        auto_parser = subparsers.add_parser('auto', aliases=['w', 'watch', 'm', 'monitor'],
                                           help='Run automated workflow (watch input folder and process new files)')
        auto_parser.add_argument('--check-interval', type=int, default=5, metavar='SECONDS',
                                help='How often to check for new files (default: 5 seconds)')
        auto_parser.add_argument('-s', '--max-size', type=float, default=10.0, metavar='MB',
                                help='Maximum output file size in MB (default: 10.0)')
        auto_parser.add_argument('--no-cache', action='store_true',
                                help='Do not use success cache in automated workflow')
        # Accept global-style flags after subcommand for convenience
        auto_parser.add_argument('-i', '--input-dir', help='Input directory to watch (default: package input folder)')
        auto_parser.add_argument('-o', '--output-dir', help='Output directory (default: package output folder)')
        # Allow specifying temp directory after the subcommand as well as globally
        auto_parser.add_argument('--temp-dir', help='Temporary directory for processing')
        auto_parser.add_argument('--max-input-size', dest='max_input_size', metavar='SIZE',
                                help='Maximum input file size to process (e.g., 500, 750MB, 1.2GB, 2TB). Bare numbers are MB.')
        # Segmentation options for automated workflow
        auto_parser.add_argument('--no-segmentation', '-ns', action='store_true',
                                help='Disable segmentation and force single-file processing with aggressive optimization')
        
        # Cache management command (with alias: ch)
        cache_parser = subparsers.add_parser('cache', aliases=['ch'],
                                            help='Cache management operations')
        cache_subparsers = cache_parser.add_subparsers(dest='cache_action', help='Cache operations')
        cache_subparsers.add_parser('clear', help='Clear all cache entries')
        cache_subparsers.add_parser('stats', help='Show cache statistics')
        cache_subparsers.add_parser('validate', help='Validate cache entries and remove invalid ones')
        
        # Performance monitoring command (with alias: perf)
        perf_parser = subparsers.add_parser('performance', aliases=['perf', 'p'],
                                           help='Performance monitoring and analysis')
        perf_subparsers = perf_parser.add_subparsers(dest='perf_action', help='Performance operations')
        perf_subparsers.add_parser('summary', help='Show performance summary')
        perf_subparsers.add_parser('bottlenecks', help='Analyze performance bottlenecks')
        perf_subparsers.add_parser('trends', help='Show performance trends')
        perf_subparsers.add_parser('report', help='Generate comprehensive performance report')
        
        # Performance profile management
        profile_parser = perf_subparsers.add_parser('profile', help='Manage performance profiles')
        profile_subparsers = profile_parser.add_subparsers(dest='profile_action', help='Profile operations')
        profile_subparsers.add_parser('list', help='List available performance profiles')
        profile_subparsers.add_parser('show', help='Show current performance profile')
        set_profile_parser = profile_subparsers.add_parser('set', help='Set performance profile')
        set_profile_parser.add_argument('profile_name', help='Profile name (fast, balanced, quality)')
        
        # Performance metrics management
        metrics_parser = perf_subparsers.add_parser('metrics', help='Manage performance metrics')
        metrics_subparsers = metrics_parser.add_subparsers(dest='metrics_action', help='Metrics operations')
        metrics_subparsers.add_parser('export', help='Export performance metrics')
        clear_metrics_parser = metrics_subparsers.add_parser('clear', help='Clear old performance metrics')
        clear_metrics_parser.add_argument('--days', type=int, default=30, help='Clear metrics older than N days')
        
        # Open folder command
        open_parser = subparsers.add_parser('open',
                                           help='Open application folders in file explorer')
        open_parser.add_argument('-i', '--input', action='store_true',
                                help='Open input folder')
        open_parser.add_argument('-o', '--output', action='store_true',
                                help='Open output folder')
        open_parser.add_argument('-l', '--logs', action='store_true',
                                help='Open logs folder')
        open_parser.add_argument('-c', '--config', action='store_true',
                                help='Open config folder')
        
        # Set default command if none provided
        args = parser.parse_args()
        if not args.command:
            # Default to automated workflow if no command specified
            args.command = 'auto'
            args.check_interval = 5
            args.max_size = 10.0
            
        return args
    
    def _clear_failures_directory(self):
        """Remove all contents of the failures directory at program start.

        Keeps the directory itself present for later use.
        """
        try:
            failures_dir = os.path.join(os.getcwd(), 'failures')
            # Ensure the folder exists
            os.makedirs(failures_dir, exist_ok=True)

            removed_items = 0
            for name in os.listdir(failures_dir):
                path = os.path.join(failures_dir, name)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
                        removed_items += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        removed_items += 1
                except Exception as e:
                    if logger:
                        logger.debug(f"Could not remove item in failures dir '{name}': {e}")
            if logger:
                logger.info(f"Failures directory cleaned ({removed_items} item(s) removed)")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to clean failures directory: {e}")
    
    def _initialize_components(self, args: argparse.Namespace):
        """Initialize all components with configuration"""
        try:
            # Load configuration
            self.config = ConfigManager(args.config_dir)
            # Apply CLI overrides to configuration
            try:
                overrides = self._extract_config_overrides(args)
                if overrides:
                    self.config.update_from_args(overrides)
                    if logger:
                        logger.debug(f"Applied CLI config overrides: {overrides}")
                
                # Apply bitrate validation specific overrides using dedicated methods
                if hasattr(args, 'bitrate_floor') and args.bitrate_floor:
                    self.config.set_all_encoder_bitrate_floors(args.bitrate_floor)
                
                if hasattr(args, 'encoder_bitrate_floor') and args.encoder_bitrate_floor:
                    for encoder, bitrate_str in args.encoder_bitrate_floor:
                        try:
                            bitrate = int(bitrate_str)
                            if self.config.validate_encoder_name(encoder):
                                self.config.set_encoder_bitrate_floor(encoder, bitrate)
                            else:
                                logger.warning(f"Unsupported encoder for bitrate override: {encoder}")
                        except ValueError:
                            logger.warning(f"Invalid bitrate value for {encoder}: {bitrate_str}")
                
                # Apply quality evaluation specific overrides
                self.config.apply_quality_evaluation_overrides(args)
                
                # Apply debug logging overrides
                debug_enabled = getattr(args, 'debug', False)
                quality_debug = getattr(args, 'quality_debug', False)
                self.config.apply_debug_logging_overrides(debug_enabled, quality_debug)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to apply CLI config overrides: {e}")
            
            # Initialize hardware detector
            self.hardware = HardwareDetector()
            
            # Force software encoding if requested
            if args.force_software:
                self.hardware.force_software_encoding()
            
            # Initialize video compressor
            self.video_compressor = DynamicVideoCompressor(self.config, self.hardware)
            
            # Initialize GIF components
            self.gif_generator = GifGenerator(self.config)
            # Note: GifGenerator now has optimizer built-in, but we keep a separate instance for direct optimization
            self.gif_optimizer = GifOptimizer(self.config)
            self.file_validator = FileValidator()
            # Pass shared video_compressor and gif_generator instances to avoid duplicate initialization
            self.automated_workflow = AutomatedWorkflow(self.config, self.hardware, 
                                                       video_compressor=self.video_compressor,
                                                       gif_generator=self.gif_generator)
            # Thread max input size through to workflow
            try:
                self.automated_workflow.max_input_size_bytes = self._parse_max_input_size_to_bytes(getattr(args, 'max_input_size', None))
            except Exception:
                self.automated_workflow.max_input_size_bytes = None
            # Propagate preferred segments (legacy single-segment retained via value 1)
            try:
                preferred = getattr(args, 'prefer_segments', None)
                if preferred is not None:
                    self.automated_workflow.preferred_segments = int(preferred)
                else:
                    self.automated_workflow.preferred_segments = None
            except Exception:
                self.automated_workflow.preferred_segments = None

            # Propagate no-segmentation flag to automated workflow
            try:
                self.automated_workflow.force_single_file = bool(getattr(args, 'no_segmentation', False))
            except Exception:
                self.automated_workflow.force_single_file = False

            # Clean up duplicate segment summaries at startup
            try:
                self.automated_workflow.cleanup_segments_summary_files()
            except Exception as e:
                logger.debug(f"Segments summary cleanup skipped: {e}")
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _parse_max_input_size_to_bytes(self, size_str: Optional[str]) -> Optional[int]:
        """Parse a human-readable size string into bytes.

        Rules: bare numbers => MB; supports KB/MB/GB/TB (case-insensitive). Examples: 500, 750MB, 1.2GB, 2TB.
        Returns integer bytes or None if not provided.
        """
        try:
            if not size_str:
                return None
            s = str(size_str).strip()
            # If just digits (and optional decimal), treat as MB
            import re
            if re.fullmatch(r"\d+(?:\.\d+)?", s):
                value_mb = float(s)
                return int(value_mb * 1024 * 1024)
            # Parse with unit
            m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([Kk][Bb]?|[Mm][Bb]?|[Gg][Bb]?|[Tt][Bb]?)\s*", s)
            if not m:
                raise ValueError(f"Invalid size format: {size_str}")
            value = float(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith('k'):
                mult = 1024
            elif unit.startswith('m'):
                mult = 1024 ** 2
            elif unit.startswith('g'):
                mult = 1024 ** 3
            elif unit.startswith('t'):
                mult = 1024 ** 4
            else:
                mult = 1024 ** 2
            return int(value * mult)
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse max input size '{size_str}': {e}")
            return None
    
    def _extract_config_overrides(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract configuration overrides from CLI arguments"""
        overrides = {}
        
        # Temporary directory override is intentionally ignored as of v0.1.2.
        # The application always uses <package_root>/temp regardless of CLI input.
        
        # File size limit
        if hasattr(args, 'max_size') and args.max_size:
            overrides['video_compression.max_file_size_mb'] = args.max_size
            overrides['gif_settings.max_file_size_mb'] = args.max_size
        
        # Video compression specific
        if hasattr(args, 'quality') and args.quality:
            overrides['video_compression.quality.crf'] = args.quality
        
        if hasattr(args, 'bitrate') and args.bitrate:
            # Convert bitrate format if needed
            bitrate = args.bitrate
            if not bitrate.endswith('k'):
                bitrate += 'k'
            overrides['video_compression.platforms.custom.bitrate'] = bitrate
        
        if hasattr(args, 'fps') and args.fps:
            overrides['video_compression.platforms.custom.fps'] = args.fps
            overrides['gif_settings.fps'] = args.fps
        
        # GIF specific
        if hasattr(args, 'width') and args.width:
            overrides['gif_settings.width'] = args.width
        
        if hasattr(args, 'height') and args.height:
            overrides['gif_settings.height'] = args.height
        
        if hasattr(args, 'colors') and args.colors:
            overrides['gif_settings.colors'] = args.colors
        
        # Bitrate validation overrides
        if hasattr(args, 'disable_bitrate_validation') and args.disable_bitrate_validation:
            overrides['video_compression.bitrate_validation.enabled'] = False
        
        if hasattr(args, 'bitrate_safety_margin') and args.bitrate_safety_margin:
            overrides['video_compression.bitrate_validation.safety_margin'] = args.bitrate_safety_margin
        
        # Handle minimum resolution override
        if hasattr(args, 'min_resolution') and args.min_resolution:
            overrides['video_compression.bitrate_validation.min_resolution'] = {
                'width': args.min_resolution[0],
                'height': args.min_resolution[1]
            }
        
        # Handle minimum FPS override
        if hasattr(args, 'min_fps') and args.min_fps:
            overrides['video_compression.bitrate_validation.min_fps'] = args.min_fps
        
        # Handle bitrate floor override - this needs special handling since it affects all encoders
        if hasattr(args, 'bitrate_floor') and args.bitrate_floor:
            # Override all encoder minimums with the specified floor
            encoder_minimums = {
                'libx264': args.bitrate_floor,
                'libx265': args.bitrate_floor,
                'h264_nvenc': args.bitrate_floor,
                'h264_amf': args.bitrate_floor,
                'h264_qsv': args.bitrate_floor,
                'h264_videotoolbox': args.bitrate_floor
            }
            overrides['video_compression.bitrate_validation.encoder_minimums'] = encoder_minimums
        
        # Handle encoder-specific bitrate floor overrides
        if hasattr(args, 'encoder_bitrate_floor') and args.encoder_bitrate_floor:
            # Get current encoder minimums from config or use defaults
            current_minimums = self.config.get('video_compression.bitrate_validation.encoder_minimums', {
                'libx264': 10,
                'libx265': 15,
                'h264_nvenc': 8,
                'h264_amf': 8,
                'h264_qsv': 8,
                'h264_videotoolbox': 8
            }) if hasattr(self, 'config') and self.config else {
                'libx264': 10,
                'libx265': 15,
                'h264_nvenc': 8,
                'h264_amf': 8,
                'h264_qsv': 8,
                'h264_videotoolbox': 8
            }
            
            # Apply encoder-specific overrides
            for encoder, bitrate_str in args.encoder_bitrate_floor:
                try:
                    bitrate = int(bitrate_str)
                    if bitrate <= 0:
                        logger.warning(f"Invalid bitrate floor for {encoder}: {bitrate} (must be positive)")
                        continue
                    
                    # Validate encoder name
                    supported_encoders = ['libx264', 'libx265', 'h264_nvenc', 'h264_amf', 'h264_qsv', 'h264_videotoolbox']
                    if encoder not in supported_encoders:
                        logger.warning(f"Unsupported encoder: {encoder}. Supported: {', '.join(supported_encoders)}")
                        continue
                    
                    current_minimums[encoder] = bitrate
                    logger.debug(f"Set bitrate floor override for {encoder}: {bitrate}kbps")
                except ValueError:
                    logger.warning(f"Invalid bitrate value for {encoder}: {bitrate_str} (must be integer)")
                    continue
            
            overrides['video_compression.bitrate_validation.encoder_minimums'] = current_minimums
        
        return overrides
    
    def _execute_command(self, args: argparse.Namespace):
        """Execute the requested command"""
        
        # Normalize command aliases to base commands
        command = args.command
        if command in ['c', 'v', 'video']:
            command = 'compress'
        elif command in ['g', 'a', 'anim']:
            command = 'gif'
        elif command in ['w', 'watch', 'm', 'monitor']:
            command = 'auto'
        elif command in ['hw', 'i', 'info']:
            command = 'hardware-info'
        elif command in ['cfg']:
            command = 'config'
        elif command in ['ch']:
            command = 'cache'
        elif command in ['perf', 'p']:
            command = 'performance'
        
        if command == 'compress':
            self._compress_video_smart(args)
        
        elif command == 'gif':
            self._create_gif_smart(args)
        
        elif command == 'batch-compress':
            self._batch_compress(args)
        
        elif command == 'batch-gif':
            self._batch_gif(args)
        
        elif command == 'optimize-gif':
            self._optimize_gif(args)
        
        elif command == 'quality-gif':
            self._create_quality_gif(args)
        
        elif command == 'hardware-info':
            self._show_hardware_info()
        
        elif command == 'config':
            self._handle_config_command(args)
        
        elif command == 'auto':
            self._run_automated_workflow(args)
        
        elif command == 'cache':
            self._handle_cache_command(args)
        
        elif command == 'performance':
            self._handle_performance_command(args)
        
        elif command == 'open':
            self._handle_open_command(args)
        
        else:
            logger.error(f"Unknown command: {command}")
            sys.exit(1)
    
    def _is_glob_pattern(self, path: str) -> bool:
        """Check if a path contains glob pattern characters"""
        return any(char in path for char in ['*', '?', '['])
    
    def _compress_video_smart(self, args: argparse.Namespace):
        """Smart compress handler that auto-detects batch mode from glob patterns"""
        from .logger_setup import get_app_base_dir
        base_dir = get_app_base_dir()
        # Check if input is a glob pattern
        if self._is_glob_pattern(args.input):
            logger.info(f"Detected glob pattern, switching to batch mode: {args.input}")
            # Convert to batch mode arguments
            batch_args = argparse.Namespace(
                input_pattern=args.input,
                output_dir=args.output if args.output else (getattr(args, 'output_dir', None) or os.path.join(base_dir, 'output')),
                platform=getattr(args, 'platform', None),
                suffix=getattr(args, 'suffix', '_compressed'),
                parallel=getattr(args, 'parallel', None),
                max_size=getattr(args, 'max_size', None),
                max_files=getattr(args, 'max_files', None),
                max_input_size=getattr(args, 'max_input_size', None),
                encoder=getattr(args, 'encoder', None),
                quality=getattr(args, 'quality', None),
                bitrate=getattr(args, 'bitrate', None),
                resolution=getattr(args, 'resolution', None),
                fps=getattr(args, 'fps', None)
            )
            self._batch_compress(batch_args)
        else:
            # Single file mode
            if not args.output:
                # Auto-generate output path
                import os
                output_dir = getattr(args, 'output_dir', None) or os.path.join(base_dir, 'output')
                os.makedirs(output_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(args.input))[0]
                suffix = getattr(args, 'suffix', '_compressed')
                args.output = os.path.join(output_dir, f"{basename}{suffix}.mp4")
                logger.info(f"Auto-generated output path: {args.output}")
            self._compress_video(args)
    
    def _create_gif_smart(self, args: argparse.Namespace):
        """Smart GIF handler that auto-detects operation type and batch mode"""
        from .logger_setup import get_app_base_dir
        base_dir = get_app_base_dir()
        # Check if input is a glob pattern (batch mode)
        if self._is_glob_pattern(args.input):
            logger.info(f"Detected glob pattern, switching to batch GIF mode: {args.input}")
            # Convert to batch mode arguments
            batch_args = argparse.Namespace(
                input_pattern=args.input,
                output_dir=args.output if args.output else (getattr(args, 'output_dir', None) or os.path.join(base_dir, 'output')),
                platform=getattr(args, 'platform', None),
                duration=getattr(args, 'duration', None),
                parallel=getattr(args, 'parallel', None),
                max_size=getattr(args, 'max_size', None),
                max_files=getattr(args, 'max_files', None),
                max_input_size=getattr(args, 'max_input_size', None),
                start=getattr(args, 'start', 0),
                fps=getattr(args, 'fps', None),
                width=getattr(args, 'width', None),
                height=getattr(args, 'height', None),
                colors=getattr(args, 'colors', None)
            )
            self._batch_gif(batch_args)
        else:
            # Single file mode - detect operation type
            if not args.output:
                # Auto-generate output path
                import os
                output_dir = getattr(args, 'output_dir', None) or os.path.join(base_dir, 'output')
                os.makedirs(output_dir, exist_ok=True)
                basename = os.path.splitext(os.path.basename(args.input))[0]
                args.output = os.path.join(output_dir, f"{basename}.gif")
                logger.info(f"Auto-generated output path: {args.output}")
            
            # Detect operation type based on input/output and flags
            input_is_gif = args.input.lower().endswith('.gif')
            output_is_gif = args.output.lower().endswith('.gif')
            
            if input_is_gif and output_is_gif:
                # GIF to GIF = optimize existing GIF
                logger.info("Detected GIF optimization mode (input and output are both GIFs)")
                self._optimize_gif(args)
            elif hasattr(args, 'target_size') and args.target_size:
                # target-size specified = quality-optimized GIF
                logger.info(f"Detected quality-optimized GIF mode (target size: {args.target_size}MB)")
                self._create_quality_gif(args)
            else:
                # Standard GIF creation (may include iterative optimization if max-size specified)
                self._create_gif(args)
    
    def _compress_video(self, args: argparse.Namespace):
        """Handle video compression command"""
        logger.info(f"Compressing video: {args.input} -> {args.output}")
        
        try:
            # Validate input file
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
            # Enforce max input size if provided
            max_input_bytes = self._parse_max_input_size_to_bytes(getattr(args, 'max_input_size', None))
            if max_input_bytes is not None:
                try:
                    sz = os.path.getsize(args.input)
                    if sz > max_input_bytes:
                        logger.info(f"Skipping {os.path.basename(args.input)}: {sz/1024/1024:.2f}MB exceeds max input size")
                        return
                except Exception:
                    pass
            
            # Create output directory if needed
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Perform compression
            results = self.video_compressor.compress_video(
                input_path=args.input,
                output_path=args.output,
                platform=args.platform,
                max_size_mb=args.max_size
            )
            
            # Display results
            self._display_compression_results(results)
            
        except Exception as e:
            logger.error(f"Video compression failed: {e}")
            raise
    
    def _create_gif(self, args: argparse.Namespace):
        """Handle GIF creation command"""
        logger.info(f"Creating GIF: {args.input} -> {args.output}")
        
        # Determine max size (command-specific takes precedence over global)
        max_size_mb = getattr(args, 'max_size', None)
        if max_size_mb:
            logger.info(f"Quality optimization enabled with max size: {max_size_mb}MB")
        else:
            logger.info("Using standard GIF generation (no size limit specified)")
        
        try:
            # Validate input file
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
            # Enforce max input size if provided
            max_input_bytes = self._parse_max_input_size_to_bytes(getattr(args, 'max_input_size', None))
            if max_input_bytes is not None:
                try:
                    sz = os.path.getsize(args.input)
                    if sz > max_input_bytes:
                        logger.info(f"Skipping {os.path.basename(args.input)}: {sz/1024/1024:.2f}MB exceeds max input size")
                        return
                except Exception:
                    pass
            
            # Create output directory if needed
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create GIF
            results = self.gif_generator.create_gif(
                input_video=args.input,
                output_path=args.output,
                platform=args.platform,
                max_size_mb=max_size_mb,
                start_time=args.start,
                duration=args.duration
            )
            
            # Handle segmentation results by moving files from temp to output
            if results.get('method') == 'Video Segmentation':
                if results.get('success', False):
                    # For successful segmentation, move temp files to final location
                    temp_segments_folder = results.get('temp_segments_folder')
                    base_name = results.get('base_name')
                    segments = results.get('segments', [])
                    
                    if temp_segments_folder and base_name and segments:
                        # Create final segments folder
                        output_dir = os.path.dirname(args.output)
                        final_segments_folder = os.path.join(output_dir, f"{base_name}_segments")
                        os.makedirs(final_segments_folder, exist_ok=True)
                        
                        # Move segments from temp to final location
                        moved_segments = 0
                        for segment in segments:
                            temp_path = segment.get('temp_path')
                            segment_name = segment.get('name')
                            
                            if temp_path and segment_name and os.path.exists(temp_path):
                                # Validate size before moving to final output
                                temp_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                                if temp_size_mb > args.max_size:
                                    logger.error(f"Segment {segment_name} exceeds size limit: {temp_size_mb:.2f}MB > {args.max_size:.2f}MB")
                                    # Remove the oversized temp file instead of moving it
                                    try:
                                        os.remove(temp_path)
                                        logger.debug(f"Removed oversized segment: {segment_name}")
                                    except Exception:
                                        pass
                                    continue
                                
                                final_path = os.path.join(final_segments_folder, segment_name)
                                try:
                                    shutil.move(temp_path, final_path)
                                    
                                    # Final validation after move
                                    final_size_mb = os.path.getsize(final_path) / (1024 * 1024)
                                    if final_size_mb > args.max_size:
                                        logger.error(f"Final segment {segment_name} exceeds size limit: {final_size_mb:.2f}MB > {args.max_size:.2f}MB")
                                        os.remove(final_path)  # Remove the invalid file
                                        continue
                                    
                                    moved_segments += 1
                                    logger.debug(f"Moved segment: {segment_name} -> {final_path}")
                                except Exception as e:
                                    logger.error(f"Failed to move segment {segment_name}: {e}")
                        
                        # Move summary file if it exists
                        summary_file = os.path.join(temp_segments_folder, f"{base_name}_segments_info.txt")
                        if os.path.exists(summary_file):
                            final_summary = os.path.join(final_segments_folder, f"zzz_{base_name}_segments_info.txt")
                            try:
                                shutil.move(summary_file, final_summary)
                            except Exception as e:
                                logger.warning(f"Failed to move summary file: {e}")
                        
                        # Ensure MP4 is moved to segments folder for user convenience
                        try:
                            # Find the source MP4 file
                            source_mp4 = None
                            for segment in segments:
                                if segment.get('temp_path') and segment.get('temp_path').endswith('.mp4'):
                                    source_mp4 = segment.get('temp_path')
                                    break
                            
                            if source_mp4 and os.path.exists(source_mp4):
                                # Check if there's an MP4 in the final segments folder
                                mp4_files = [f for f in os.listdir(final_segments_folder) if f.endswith('.mp4')]
                                if not mp4_files:
                                    # Move the source MP4 to segments folder
                                    mp4_name = os.path.basename(source_mp4)
                                    final_mp4_path = os.path.join(final_segments_folder, mp4_name)
                                    shutil.move(source_mp4, final_mp4_path)
                                    logger.info(f"Moved source MP4 to segments folder: {mp4_name}")
                        except Exception as e:
                            logger.warning(f"Failed to move MP4 to segments folder: {e}")
                        
                        # Clean up temp folder
                        try:
                            if os.path.exists(temp_segments_folder):
                                shutil.rmtree(temp_segments_folder)
                                logger.info(f"Cleaned up temp segments folder: {temp_segments_folder}")
                        except Exception as e:
                            logger.warning(f"Could not clean up temp folder: {e}")
                        
                        # Update results to show the final segments folder
                        results['segments_folder'] = final_segments_folder
                        results['output_file'] = final_segments_folder
                    else:
                        # Segmentation data incomplete, clean up temp files
                        temp_folder = results.get('temp_segments_folder')
                        temp_files = results.get('temp_files_to_cleanup', [])
                        
                        if temp_files:
                            for temp_file in temp_files:
                                try:
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                except Exception:
                                    pass
                        
                        if temp_folder and os.path.exists(temp_folder):
                            try:
                                shutil.rmtree(temp_folder)
                            except Exception:
                                pass
                else:
                    # Failed segmentation - clean up temp files
                    temp_folder = results.get('temp_segments_folder')
                    temp_files = results.get('temp_files_to_cleanup', [])
                    
                    if temp_files:
                        for temp_file in temp_files:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                            except Exception:
                                pass
                    
                    if temp_folder and os.path.exists(temp_folder):
                        try:
                            shutil.rmtree(temp_folder)
                        except Exception:
                            pass
                    
                    raise Exception(f"Segmentation failed: {results.get('error', 'Unknown error')}")
            
            # Display results based on method used
            # Normalize segmentation method values across modules
            method = results.get('method')
            if method in ('Video Segmentation', 'segmentation'):
                self._display_segmentation_results(results)
            elif method in ('Single Segment Conversion', 'single'):
                self._display_single_segment_conversion_results(results)
            else:
                self._display_quality_gif_results(results)
            
        except Exception as e:
            logger.error(f"GIF creation failed: {e}")
            raise
    
    def _batch_compress(self, args: argparse.Namespace):
        """Handle batch video compression"""
        import glob
        import concurrent.futures
        from .logger_setup import get_app_base_dir
        
        # Find input files
        input_files = glob.glob(args.input_pattern)
        if not input_files:
            logger.error(f"No files found matching pattern: {args.input_pattern}")
            return

        # Apply processing limit if specified
        if getattr(args, 'max_files', None):
            input_files = input_files[: int(args.max_files)]
        
        logger.info(f"Found {len(input_files)} files to compress")
        
        # Resolve output directory: prefer provided --output-dir, else default to package output
        effective_output_dir = args.output_dir if getattr(args, 'output_dir', None) else os.path.join(get_app_base_dir(), 'output')
        os.makedirs(effective_output_dir, exist_ok=True)
        
        # Determine number of parallel processes
        max_workers = args.parallel or min(4, len(input_files))
        
        # Parse max input size once
        max_input_bytes = self._parse_max_input_size_to_bytes(getattr(args, 'max_input_size', None))

        def compress_single(input_file):
            try:
                # Enforce max input size if provided
                if max_input_bytes is not None:
                    try:
                        sz = os.path.getsize(input_file)
                        if sz > max_input_bytes:
                            logger.info(f"Skipping {os.path.basename(input_file)}: {sz/1024/1024:.2f}MB exceeds max input size")
                            return {'success': True, 'input': input_file, 'skipped': True, 'reason': 'max_input_size'}
                    except Exception:
                        pass
                # Generate output filename
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(effective_output_dir, f"{basename}{args.suffix}.mp4")
                
                # NEW: Check if output already exists and is valid
                if os.path.exists(output_file):
                    is_valid, _ = self.file_validator.is_valid_video(output_file)
                    if is_valid:
                        logger.info(f"Skipping {basename} - valid compressed output already exists")
                        return {
                            'success': True, 
                            'input': input_file, 
                            'output': output_file, 
                            'result': {'method': 'Skipped - Already Compressed'},
                            'skipped': True
                        }
                    else:
                        logger.info(f"Recompressing {basename} - existing output is invalid")
                
                # Compress video
                result = self.video_compressor.compress_video(
                    input_path=input_file,
                    output_path=output_file,
                    platform=args.platform,
                    max_size_mb=args.max_size
                )
                
                return {'success': True, 'input': input_file, 'output': output_file, 'result': result}
                
            except Exception as e:
                # Enhanced error handling for bitrate validation failures
                error_msg = str(e)
                error_type = 'general'
                basename = os.path.basename(input_file)
                
                # Detailed error categorization for better batch processing
                if isinstance(e, BitrateValidationError):
                    error_type = 'bitrate_validation'
                    logger.error(f"Bitrate validation failure for {basename}: {e.get_short_message()}")
                    logger.info(f"Detailed error for {basename}:\n{e.get_detailed_message()}")
                    
                    # Context-specific suggestions
                    if e.severity == 'critical':
                        logger.error(f"Critical bitrate failure for {basename} - segmentation strongly recommended")
                        logger.info(f"Suggestions: --enable-segmentation or increase target size")
                    else:
                        logger.info(f"Suggestions: --bitrate-floor {max(1, e.bitrate_kbps)} or --enable-auto-adjustment")
                        
                elif 'bitrate' in error_msg.lower() and ('minimum' in error_msg.lower() or 'floor' in error_msg.lower()):
                    error_type = 'bitrate_validation'
                    logger.error(f"Bitrate validation failure for {basename}: {error_msg}")
                    logger.info(f"Suggestion: Try using --bitrate-floor to override minimum requirements or enable segmentation")
                    
                elif 'segmentation' in error_msg.lower():
                    error_type = 'segmentation'
                    logger.error(f"Segmentation failure for {basename}: {error_msg}")
                    logger.info(f"Suggestion: Check video duration and size limits, or try --force-single-file")
                    
                elif 'encoder' in error_msg.lower() or 'ffmpeg' in error_msg.lower():
                    error_type = 'encoder'
                    logger.error(f"Encoder failure for {basename}: {error_msg}")
                    logger.info(f"Suggestion: Try --force-software or different encoder settings")
                    
                elif 'timeout' in error_msg.lower():
                    error_type = 'timeout'
                    logger.error(f"Processing timeout for {basename}: {error_msg}")
                    logger.info(f"Suggestion: Try smaller files or increase timeout settings")
                    
                elif 'memory' in error_msg.lower() or 'out of memory' in error_msg.lower():
                    error_type = 'memory'
                    logger.error(f"Memory error for {basename}: {error_msg}")
                    logger.info(f"Suggestion: Reduce resolution or enable segmentation")
                    
                else:
                    logger.error(f"Failed to compress {basename}: {error_msg}")
                    logger.debug(f"Full error details for {basename}: {repr(e)}")
                
                # Continue processing other files in batch mode
                logger.info(f"Continuing batch processing despite failure of {basename}")
                
                return {
                    'success': False, 
                    'input': input_file, 
                    'error': error_msg,
                    'error_type': error_type,
                    'basename': basename,
                    'exception_type': type(e).__name__
                }
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(compress_single, input_files))
        
        # Enhanced summary with error categorization
        successful = sum(1 for r in results if r['success'])
        skipped = sum(1 for r in results if r.get('skipped', False))
        failed = len(results) - successful
        
        # Enhanced failure categorization and reporting
        bitrate_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'bitrate_validation')
        segmentation_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'segmentation')
        encoder_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'encoder')
        timeout_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'timeout')
        memory_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'memory')
        other_failures = failed - bitrate_failures - segmentation_failures - encoder_failures - timeout_failures - memory_failures
        
        logger.info(f"Batch compression completed: {successful} successful, {skipped} skipped, {failed} failed")
        
        if failed > 0:
            logger.error("=== BATCH PROCESSING FAILURE ANALYSIS ===")
            
            if bitrate_failures > 0:
                logger.error(f"Bitrate validation failures: {bitrate_failures}")
                logger.error("  Root cause: Calculated bitrate below encoder minimum requirements")
                logger.error("  Solutions:")
                logger.error("    • Enable segmentation: --enable-segmentation")
                logger.error("    • Lower bitrate floor: --bitrate-floor 1")
                logger.error("    • Use hardware encoders: --force-hardware")
                logger.error("    • Increase target size: --max-size 15")
                
            if segmentation_failures > 0:
                logger.error(f"Segmentation failures: {segmentation_failures}")
                logger.error("  Root cause: Unable to split video into acceptable segments")
                logger.error("  Solutions:")
                logger.error("    • Check video duration and complexity")
                logger.error("    • Try smaller segment duration: --segment-duration 30")
                logger.error("    • Force single file: --force-single-file")
                
            if encoder_failures > 0:
                logger.error(f"Encoder failures: {encoder_failures}")
                logger.error("  Root cause: FFmpeg encoding process failed")
                logger.error("  Solutions:")
                logger.error("    • Try software encoding: --force-software")
                logger.error("    • Check video file integrity")
                logger.error("    • Update FFmpeg installation")
                
            if timeout_failures > 0:
                logger.error(f"Timeout failures: {timeout_failures}")
                logger.error("  Root cause: Processing took too long")
                logger.error("  Solutions:")
                logger.error("    • Reduce video resolution or duration")
                logger.error("    • Enable segmentation for large files")
                
            if memory_failures > 0:
                logger.error(f"Memory failures: {memory_failures}")
                logger.error("  Root cause: Insufficient system memory")
                logger.error("  Solutions:")
                logger.error("    • Reduce video resolution")
                logger.error("    • Enable segmentation")
                logger.error("    • Close other applications")
                
            if other_failures > 0:
                logger.error(f"Other failures: {other_failures}")
                logger.error("  Check individual file logs for specific error details")
        
        # Enhanced failed files logging with error categorization
        failed_files = [r for r in results if not r['success']]
        if failed_files:
            logger.error("=== FAILED FILES DETAILS ===")
            
            # Group by error type for better organization
            error_groups = {}
            for failure in failed_files:
                error_type = failure.get('error_type', 'unknown')
                if error_type not in error_groups:
                    error_groups[error_type] = []
                error_groups[error_type].append(failure)
            
            for error_type, failures in error_groups.items():
                logger.error(f"{error_type.upper()} FAILURES ({len(failures)} files):")
                for failure in failures:
                    basename = failure.get('basename', 'unknown')
                    error_msg = failure.get('error', 'unknown error')
                    exception_type = failure.get('exception_type', 'Exception')
                    logger.error(f"  • {basename}: {exception_type} - {error_msg}")
        
        # Log batch processing resilience message
        if failed > 0 and successful > 0:
            success_rate = (successful / (successful + failed)) * 100
            logger.info(f"Batch processing resilience: {success_rate:.1f}% success rate - processing continued despite {failed} failures")
    
    def _batch_gif(self, args: argparse.Namespace):
        """Handle batch GIF creation"""
        import glob
        import concurrent.futures
        
        # Find input files
        input_files = glob.glob(args.input_pattern)
        if not input_files:
            logger.error(f"No files found matching pattern: {args.input_pattern}")
            return

        # Apply processing limit if specified
        if getattr(args, 'max_files', None):
            input_files = input_files[: int(args.max_files)]
        
        logger.info(f"Found {len(input_files)} files for GIF creation")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Determine number of parallel processes
        max_workers = args.parallel or min(4, len(input_files))
        
        # Parse max input size once
        max_input_bytes = self._parse_max_input_size_to_bytes(getattr(args, 'max_input_size', None))

        def create_single_gif(input_file):
            try:
                # Enforce max input size if provided
                if max_input_bytes is not None:
                    try:
                        sz = os.path.getsize(input_file)
                        if sz > max_input_bytes:
                            logger.info(f"Skipping {os.path.basename(input_file)}: {sz/1024/1024:.2f}MB exceeds max input size")
                            return {'success': True, 'input': input_file, 'skipped': True, 'reason': 'max_input_size'}
                    except Exception:
                        pass
                # Generate output filename
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(args.output_dir, f"{basename}.gif")
                
                # NEW: Comprehensive check for existing outputs before processing
                if self._has_existing_output_cli(input_file, effective_output_dir):
                    logger.info(f"Skipping {basename} - output already exists")
                    return {
                        'success': True,
                        'input_file': input_file,
                        'output_info': 'Existing output found',
                        'method': 'Skipped - Already Processed',
                        'skipped': True
                    }
                
                # Create GIF
                result = self.gif_generator.create_gif(
                    input_video=input_file,
                    output_path=output_file,
                    platform=args.platform,
                    max_size_mb=args.max_size,
                    duration=args.duration
                )
                
                # Handle both single GIF and segmented results
                output_info = output_file
                if result.get('method') == 'Video Segmentation':
                    if result.get('success', False):
                        # For successful segmentation, move temp files to final location
                        temp_segments_folder = result.get('temp_segments_folder')
                        base_name = result.get('base_name')
                        segments = result.get('segments', [])
                        
                        if temp_segments_folder and base_name and segments:
                            # Create final segments folder
                            final_segments_folder = os.path.join(args.output_dir, f"{base_name}_segments")
                            try:
                                os.makedirs(args.output_dir, exist_ok=True)
                            except Exception:
                                pass
                            os.makedirs(final_segments_folder, exist_ok=True)
                            
                            # Move segments from temp to final location
                            moved_segments = 0
                            for segment in segments:
                                temp_path = segment.get('temp_path')
                                segment_name = segment.get('name')
                                
                                if temp_path and segment_name and os.path.exists(temp_path):
                                    # Validate size before moving to final output
                                    temp_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                                    if temp_size_mb > args.max_size:
                                        logger.error(f"Batch segment {segment_name} exceeds size limit: {temp_size_mb:.2f}MB > {args.max_size:.2f}MB")
                                        # Remove the oversized temp file instead of moving it
                                        try:
                                            os.remove(temp_path)
                                            logger.debug(f"Removed oversized batch segment: {segment_name}")
                                        except Exception:
                                            pass
                                        continue
                                    
                                    final_path = os.path.join(final_segments_folder, segment_name)
                                    try:
                                        shutil.move(temp_path, final_path)
                                        
                                        # Final validation after move
                                        final_size_mb = os.path.getsize(final_path) / (1024 * 1024)
                                        if final_size_mb > args.max_size:
                                            logger.error(f"Final batch segment {segment_name} exceeds size limit: {final_size_mb:.2f}MB > {args.max_size:.2f}MB")
                                            os.remove(final_path)  # Remove the invalid file
                                            continue
                                        
                                        moved_segments += 1
                                        logger.debug(f"Moved segment: {segment_name} -> {final_path}")
                                    except Exception as e:
                                        logger.error(f"Failed to move segment {segment_name}: {e}")
                            
                            # Move summary file if it exists
                            summary_file = os.path.join(temp_segments_folder, f"{base_name}_segments_info.txt")
                            if os.path.exists(summary_file):
                                final_summary = os.path.join(final_segments_folder, f"{base_name}_segments_info.txt")
                                try:
                                    shutil.move(summary_file, final_summary)
                                except Exception as e:
                                    logger.warning(f"Failed to move summary file: {e}")
                            
                            # Ensure MP4 is moved to segments folder for user convenience
                            try:
                                # Find the source MP4 file
                                source_mp4 = None
                                for segment in segments:
                                    if segment.get('temp_path') and segment.get('temp_path').endswith('.mp4'):
                                        source_mp4 = segment.get('temp_path')
                                        break
                                
                                if source_mp4 and os.path.exists(source_mp4):
                                    # Check if there's an MP4 in the final segments folder
                                    mp4_files = [f for f in os.listdir(final_segments_folder) if f.endswith('.mp4')]
                                    if not mp4_files:
                                        # Move the source MP4 to segments folder
                                        mp4_name = os.path.basename(source_mp4)
                                        final_mp4_path = os.path.join(final_segments_folder, mp4_name)
                                        shutil.move(source_mp4, final_mp4_path)
                                        logger.info(f"Moved source MP4 to segments folder: {mp4_name}")
                            except Exception as e:
                                logger.warning(f"Failed to move MP4 to segments folder: {e}")
                            
                            # Clean up temp folder
                            try:
                                if os.path.exists(temp_segments_folder):
                                    shutil.rmtree(temp_segments_folder)
                                    logger.info(f"Cleaned up temp segments folder: {temp_segments_folder}")
                            except Exception as e:
                                logger.warning(f"Could not clean up temp folder: {e}")
                            
                            output_info = final_segments_folder
                        else:
                            # Segmentation data incomplete, clean up temp files
                            temp_folder = result.get('temp_segments_folder')
                            temp_files = result.get('temp_files_to_cleanup', [])
                            
                            if temp_files:
                                for temp_file in temp_files:
                                    try:
                                        if os.path.exists(temp_file):
                                            os.remove(temp_file)
                                    except Exception:
                                        pass
                            
                            if temp_folder and os.path.exists(temp_folder):
                                try:
                                    shutil.rmtree(temp_folder)
                                except Exception:
                                    pass
                    else:
                        # Failed segmentation - clean up temp files
                        temp_folder = result.get('temp_segments_folder')
                        temp_files = result.get('temp_files_to_cleanup', [])
                        
                        if temp_files:
                            for temp_file in temp_files:
                                try:
                                    if os.path.exists(temp_file):
                                        os.remove(temp_file)
                                except Exception:
                                    pass
                        
                        if temp_folder and os.path.exists(temp_folder):
                            try:
                                shutil.rmtree(temp_folder)
                            except Exception:
                                pass
                        
                        # Return failure for failed segmentation
                        return {'success': False, 'input': input_file, 'error': result.get('error', 'Segmentation failed')}
                elif result.get('method') == 'Single Segment Conversion':
                    # For single segment conversion, the file is already at the correct location
                    # Just update the output info to point to the correct file
                    output_info = result.get('output_file', output_file)
                    logger.info(f"Single segment conversion completed: {output_info}")
                
                return {'success': True, 'input': input_file, 'output': output_info, 'result': result}
                
            except Exception as e:
                # Enhanced error handling for GIF creation failures
                error_msg = str(e)
                error_type = 'general'
                
                # Check if this is a bitrate validation related error
                if 'bitrate' in error_msg.lower() and ('minimum' in error_msg.lower() or 'floor' in error_msg.lower()):
                    error_type = 'bitrate_validation'
                    logger.error(f"Bitrate validation failure for GIF {os.path.basename(input_file)}: {error_msg}")
                    logger.info(f"Suggestion: Try using --bitrate-floor to override minimum requirements")
                elif 'segmentation' in error_msg.lower():
                    error_type = 'segmentation'
                    logger.error(f"GIF segmentation failure for {os.path.basename(input_file)}: {error_msg}")
                elif 'size limit' in error_msg.lower() or 'target size' in error_msg.lower():
                    error_type = 'size_limit'
                    logger.error(f"Size limit failure for GIF {os.path.basename(input_file)}: {error_msg}")
                    logger.info(f"Suggestion: Increase --max-size or reduce video duration")
                else:
                    logger.error(f"Failed to create GIF from {os.path.basename(input_file)}: {error_msg}")
                
                return {
                    'success': False, 
                    'input': input_file, 
                    'error': error_msg,
                    'error_type': error_type,
                    'basename': os.path.basename(input_file)
                }
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(create_single_gif, input_files))
        
        # Enhanced summary with error categorization
        successful = sum(1 for r in results if r['success'])
        skipped = sum(1 for r in results if r.get('skipped', False))
        failed = len(results) - successful
        
        # Categorize failures
        bitrate_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'bitrate_validation')
        segmentation_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'segmentation')
        size_limit_failures = sum(1 for r in results if not r['success'] and r.get('error_type') == 'size_limit')
        other_failures = failed - bitrate_failures - segmentation_failures - size_limit_failures
        
        logger.info(f"Batch GIF creation completed: {successful} successful, {skipped} skipped, {failed} failed")
        
        if failed > 0:
            logger.info("Failure breakdown:")
            if bitrate_failures > 0:
                logger.info(f"  • Bitrate validation failures: {bitrate_failures}")
                logger.info(f"    Suggestion: Use --bitrate-floor or --disable-bitrate-validation")
            if segmentation_failures > 0:
                logger.info(f"  • Segmentation failures: {segmentation_failures}")
                logger.info(f"    Suggestion: Check video duration and size limits")
            if size_limit_failures > 0:
                logger.info(f"  • Size limit failures: {size_limit_failures}")
                logger.info(f"    Suggestion: Increase --max-size or reduce video duration")
            if other_failures > 0:
                logger.info(f"  • Other failures: {other_failures}")
        
        # Log specific failed files for debugging
        failed_files = [r for r in results if not r['success']]
        if failed_files:
            logger.debug("Failed GIF files details:")
            for failure in failed_files:
                logger.debug(f"  {failure.get('basename', 'unknown')}: {failure.get('error', 'unknown error')}")
    
    def _optimize_gif(self, args: argparse.Namespace):
        """Handle GIF optimization command"""
        logger.info(f"Optimizing GIF: {args.input} -> {args.output}")
        
        try:
            # Validate input file
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
            
            # Create output directory if needed
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Optimize GIF
            max_size_mb = args.max_size or self.config.get('gif_settings.max_file_size_mb', 10)
            results = self.gif_generator.optimize_existing_gif(
                input_gif=args.input,
                output_gif=args.output,
                max_size_mb=max_size_mb
            )
            
            # Display results
            self._display_gif_optimization_results(results)
            
        except Exception as e:
            logger.error(f"GIF optimization failed: {e}")
            raise
    
    def _create_quality_gif(self, args: argparse.Namespace):
        """Handle quality-optimized GIF creation command"""
        logger.info(f"Creating quality-optimized GIF: {args.input} -> {args.output}")
        logger.info(f"Target size: {args.target_size}MB, Quality preference: {args.quality_preference}")
        
        try:
            # Validate input file
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
            
            # Create output directory if needed
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Create quality-optimized GIF using GifGenerator (which includes optimization)
            # Note: quality_preference is not directly supported, but GifGenerator will optimize to target size
            results = self.gif_generator.create_gif(
                input_video=args.input,
                output_path=args.output,
                platform=args.platform,
                max_size_mb=args.target_size,
                start_time=args.start,
                duration=args.duration
            )
            
            # Display results
            self._display_quality_gif_results(results)
            
        except Exception as e:
            logger.error(f"Quality GIF creation failed: {e}")
            raise
    
    def _show_hardware_info(self):
        """Display hardware acceleration information"""
        print(self.hardware.get_system_report())
    
    def _handle_config_command(self, args: argparse.Namespace):
        """Handle configuration commands"""
        if args.config_action == 'show':
            self._show_config()
        elif args.config_action == 'validate':
            self._validate_config()
        else:
            logger.error("Config command requires an action (show|validate)")
    
    def _show_config(self):
        """Show current configuration"""
        import yaml
        print("Current Configuration:")
        print("=" * 50)
        print(yaml.dump(self.config.config, default_flow_style=False, indent=2))
        
        # Show bitrate validation configuration summary
        print("\n" + "=" * 50)
        print("BITRATE VALIDATION SUMMARY:")
        print("=" * 50)
        bitrate_config = self.config.get_bitrate_validation_config()
        
        print(f"Enabled: {bitrate_config['enabled']}")
        print(f"Safety Margin: {bitrate_config['safety_margin']}")
        print(f"Segmentation Threshold: {bitrate_config['segmentation_threshold_mb']} MB")
        
        print("\nEncoder Minimum Bitrates (kbps):")
        for encoder, minimum in bitrate_config['encoder_minimums'].items():
            print(f"  {encoder}: {minimum}")
        
        print(f"\nMinimum Resolution: {bitrate_config['min_resolution']['width']}x{bitrate_config['min_resolution']['height']}")
        print(f"Minimum FPS: {bitrate_config['min_fps']}")
        
        print("\nFallback Resolutions:")
        for i, res in enumerate(bitrate_config['fallback_resolutions'], 1):
            print(f"  {i}. {res[0]}x{res[1]}")
        
        print(f"\nFPS Reduction Steps: {bitrate_config['fps_reduction_steps']}")
        print("=" * 50)
    
    def _validate_config(self):
        """Validate configuration"""
        if self.config.validate_config():
            logger.info("OK Configuration is valid")
        else:
            logger.error("✗ Configuration validation failed")
            sys.exit(1)
    
    def _display_compression_results(self, results: Dict[str, Any]):
        """Display video compression results"""
        print("\n" + "="*60)
        print("VIDEO COMPRESSION RESULTS")
        print("="*60)
        print(f"Input File:       {results.get('input_file', 'N/A')}")
        print(f"Output File:      {results.get('output_file', 'N/A')}")
        print(f"Method:           {results.get('method', 'N/A')}")
        print(f"Original Size:    {results.get('original_size_mb', 0):.2f} MB")
        print(f"Compressed Size:  {results.get('compressed_size_mb', 0):.2f} MB")
        print(f"Compression:      {results.get('compression_ratio', 0):.1f}% reduction")
        print(f"Space Saved:      {results.get('space_saved_mb', 0):.2f} MB")
        
        video_info = results.get('video_info', {})
        print(f"Resolution:       {video_info.get('width', 0)}x{video_info.get('height', 0)}")
        print(f"Duration:         {video_info.get('duration', 0):.1f} seconds")
        print(f"Frame Rate:       {video_info.get('fps', 0):.1f} fps")
        print("="*60)
    
    def _display_segmentation_results(self, results: Dict[str, Any]):
        """Display video segmentation results"""
        print("\n" + "="*60)
        print("🎬 VIDEO SEGMENTATION RESULTS")
        print("="*60)
        print(f"Method:            Video Segmentation (High Quality)")
        print(f"Segments Folder:   {results.get('segments_folder', 'N/A')}")
        print(f"Segments Created:  {results.get('segments_created', 0)}")
        print(f"Segments Failed:   {results.get('segments_failed', 0)}")
        print(f"Total Size:        {results.get('total_size_mb', 0):.2f} MB")
        print(f"Total Frames:      {results.get('frame_count', 0)}")
        print()
        
        # Display individual segment details
        segments = results.get('segments', [])
        if segments:
            print("Individual Segments:")
            print("-" * 60)
            for i, segment in enumerate(segments, 1):
                duration = segment.get('duration', 0)
                start_time = segment.get('start_time', 0)
                end_time = start_time + duration
                
                print(f"{i:2d}. {segment.get('name', 'N/A')}")
                print(f"    Time:   {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)")
                print(f"    Size:   {segment.get('size_mb', 0):.2f} MB")
                print(f"    Frames: {segment.get('frame_count', 0)}")
                print()
        
        print("📁 All segments are saved in the segments folder above.")
        print("📤 Each segment can be uploaded individually to Discord, Twitter, etc.")
        print("🎯 High quality maintained while respecting platform size limits!")
        print("="*60)

    def _display_single_segment_conversion_results(self, results: Dict[str, Any]):
        """Display single segment conversion results"""
        print("\n" + "="*60)
        print("🎬 SINGLE SEGMENT CONVERSION RESULTS")
        print("="*60)
        print(f"Method:            Single Segment Conversion")
        print(f"Output File:       {results.get('output_file', 'N/A')}")
        print(f"File Size:         {results.get('file_size_mb', 0):.2f} MB")
        print(f"Frame Count:       {results.get('frame_count', 0)}")
        print(f"Resolution:        {results.get('width', 0)}x{results.get('height', 0)}")
        print(f"Frame Rate:        {results.get('fps', 0)} fps")
        print(f"Optimization Type: {results.get('optimization_type', 'N/A')}")
        print()
        print("✅ Video was processed using segmentation but only one segment was needed.")
        print("✅ Converted to regular GIF format for simplicity.")
        print("✅ High quality maintained while staying under size limits!")
        print("="*60)

    def _display_gif_results(self, results: Dict[str, Any]):
        """Display GIF creation results"""
        print("\n" + "="*60)
        print("GIF CREATION RESULTS")
        print("="*60)
        print(f"Input Video:      {results.get('input_video', 'N/A')}")
        print(f"Output GIF:       {results.get('output_gif', 'N/A')}")
        print(f"File Size:        {results.get('file_size_mb', 0):.2f} MB")
        print(f"Duration:         {results.get('duration_seconds', 0):.1f} seconds")
        print(f"Frame Count:      {results.get('frame_count', 0)}")
        print(f"Frame Rate:       {results.get('fps', 0)} fps")
        print(f"Resolution:       {results.get('width', 0)}x{results.get('height', 0)}")
        print(f"Colors:           {results.get('colors', 0)}")
        print(f"Compression:      {results.get('compression_ratio', 0):.1f}% from original video")
        print("="*60)
    
    def _display_gif_optimization_results(self, results: Dict[str, Any]):
        """Display GIF optimization results"""
        print("\n" + "="*60)
        print("GIF OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Input GIF:        {results.get('input_gif', 'N/A')}")
        print(f"Output GIF:       {results.get('output_gif', 'N/A')}")
        print(f"Method:           {results.get('method', 'N/A')}")
        print(f"Original Size:    {results.get('original_size_mb', 0):.2f} MB")
        print(f"Optimized Size:   {results.get('optimized_size_mb', 0):.2f} MB")
        print(f"Compression:      {results.get('compression_ratio', 0):.1f}% reduction")
        print(f"Space Saved:      {results.get('space_saved_mb', 0):.2f} MB")
        print("="*60)
    
    def _display_quality_gif_results(self, results: Dict[str, Any]):
        """Display quality-optimized GIF results"""
        print("\n" + "="*60)
        print("QUALITY-OPTIMIZED GIF RESULTS")
        print("="*60)
        print(f"Output File:       {results.get('output_file', 'N/A')}")
        # Handle target size formatting
        target_size = results.get('target_size_mb', 'N/A')
        target_size_str = f"{target_size:.2f} MB" if isinstance(target_size, (int, float)) else f"{target_size} MB"
        print(f"Target Size:       {target_size_str}")
        
        # Handle actual size formatting
        actual_size = results.get('size_mb', 'N/A')
        actual_size_str = f"{actual_size:.2f} MB" if isinstance(actual_size, (int, float)) else f"{actual_size} MB"
        print(f"Actual Size:       {actual_size_str}")
        
        # Handle size efficiency formatting
        efficiency = results.get('size_efficiency', 'N/A')
        efficiency_str = f"{efficiency:.1%} of target" if isinstance(efficiency, (int, float)) else f"{efficiency} of target"
        print(f"Size Efficiency:   {efficiency_str}")
        
        # Handle quality score formatting
        quality = results.get('quality_score', 'N/A')
        quality_str = f"{quality:.2f}/10" if isinstance(quality, (int, float)) else f"{quality}/10"
        print(f"Quality Score:     {quality_str}")
        print(f"Optimization Method: {results.get('optimization_method', 'N/A')}")
        
        if 'params' in results:
            params = results['params']
            print(f"Resolution:        {params.get('width', 'N/A')}x{params.get('height', 'N/A')}")
            print(f"Frame Rate:        {params.get('fps', 'N/A')} fps")
            print(f"Colors:            {params.get('colors', 'N/A')}")
            print(f"Dithering:         {params.get('dither', 'N/A')}")
            print(f"Lossy Compression: {params.get('lossy', 'N/A')}")
        
        print(f"Frame Count:       {results.get('frame_count', 'N/A')}")
        print("="*60)
    
    def _validate_segment_folder_gifs_cli(self, segments_folder: str, max_size_mb: float) -> tuple:
        """
        Validates all GIFs in a segment folder for CLI usage.
        Returns a tuple of (valid_gifs, invalid_gifs).
        """
        valid_gifs = []
        invalid_gifs = []
        
        if not os.path.exists(segments_folder) or not os.path.isdir(segments_folder):
            logger.warning(f"Segments folder not found or not a directory: {segments_folder}")
            return [], []
        
        # Collect GIF files first
        gif_files = [os.path.join(segments_folder, f) for f in os.listdir(segments_folder) if f.lower().endswith('.gif')]
        if not gif_files:
            return [], []
        
        # Determine reasonable parallelism using config if available
        try:
            # Access the same calculation as AutomatedWorkflow when possible
            max_workers = getattr(self.automated_workflow, '_calculate_optimal_segmentation_workers')()
            if not isinstance(max_workers, int) or max_workers < 1:
                raise ValueError
        except Exception:
            max_workers = max(1, min(4, (os.cpu_count() or 2)))
        
        def _validate(gif_path: str):
            try:
                is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                    gif_path, original_path=None, max_size_mb=max_size_mb
                )
                return gif_path, bool(is_valid)
            except Exception:
                return gif_path, False
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_validate, p) for p in gif_files]
            for fut in concurrent.futures.as_completed(futures):
                path, ok = fut.result()
                if ok:
                    valid_gifs.append(path)
                else:
                    invalid_gifs.append(path)
        
        return valid_gifs, invalid_gifs
    
    def _has_existing_output_cli(self, input_file: str, output_dir: str) -> bool:
        """
        Comprehensive check for existing output files in CLI batch processing.
        
        Checks for:
        1. Optimized MP4 in output directory
        2. Single GIF in output directory  
        3. Segment folder in output directory
        4. Any of the above in subdirectories (recursive search)
        
        Args:
            input_file: Path to the input video file (string)
            output_dir: Output directory path (string)
            
        Returns:
            True if any valid output exists, False otherwise
        """
        if not os.path.exists(output_dir):
            return False
        
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Define all possible output patterns
        output_patterns = [
            f"{base_name}.mp4",           # Optimized MP4
            f"{base_name}.gif",           # Single GIF
            f"{base_name}_segments"       # Segment folder
        ]
        
        # Check root output directory first (most common case)
        for pattern in output_patterns:
            output_path = os.path.join(output_dir, pattern)
            if self._is_valid_existing_output_cli(output_path, pattern, input_file):
                logger.info(f"Found existing output for {os.path.basename(input_file)}: {output_path}")
                return True
        
        # Recursive search in all subdirectories
        try:
            for root, dirs, files in os.walk(output_dir):
                # Check files in this directory
                for pattern in output_patterns:
                    if pattern.endswith('_segments'):
                        # Check for segment folders
                        if pattern in dirs:
                            output_path = os.path.join(root, pattern)
                            if self._is_valid_existing_output_cli(output_path, pattern, input_file):
                                logger.info(f"Found existing output for {os.path.basename(input_file)}: {output_path}")
                                return True
                    else:
                        # Check for files
                        if pattern in files:
                            output_path = os.path.join(root, pattern)
                            if self._is_valid_existing_output_cli(output_path, pattern, input_file):
                                logger.info(f"Found existing output for {os.path.basename(input_file)}: {output_path}")
                                return True
        except Exception as e:
            logger.warning(f"Error during recursive output search for {os.path.basename(input_file)}: {e}")
        
        return False
    
    def _is_valid_existing_output_cli(self, output_path: str, pattern_type: str, input_file: str = None) -> bool:
        """
        Validate that an existing output is actually valid and was created from the specific input (CLI version).
        
        Args:
            output_path: Path to the potential output (string)
            pattern_type: Type of output ('*.mp4', '*.gif', or '*_segments')
            input_file: Path to the original input file for verification
            
        Returns:
            True if the output exists, is valid, and was created from the input
        """
        try:
            if not os.path.exists(output_path):
                return False
            
            # Basic validation first
            basic_valid = False
            if pattern_type.endswith('.mp4'):
                # Validate MP4 file
                is_valid, _ = self.file_validator.is_valid_video(output_path)
                basic_valid = is_valid
                
            elif pattern_type.endswith('.gif'):
                # Fast validation for existing single GIF
                is_valid, _ = self.file_validator.is_valid_gif_fast(output_path, max_size_mb=10.0)
                basic_valid = is_valid
                
            elif pattern_type.endswith('_segments'):
                # Validate segment folder - check if it contains valid GIFs
                if os.path.isdir(output_path):
                    valid_gifs, invalid_gifs = self._validate_segment_folder_gifs_cli(output_path, 10.0)
                    basic_valid = len(valid_gifs) > 0  # At least one valid GIF makes it usable
            
            if not basic_valid:
                return False
            
            # Enhanced validation: verify the output was created from this specific input
            if input_file and os.path.exists(input_file):
                return self._verify_output_source_relationship_cli(input_file, output_path, pattern_type)
            
            # If no input file provided, just return basic validation
            return basic_valid
                    
        except Exception as e:
            logger.debug(f"Error validating existing output {output_path}: {e}")
            return False
        
        return False
    
    def _verify_output_source_relationship_cli(self, input_file: str, output_path: str, pattern_type: str) -> bool:
        """
        CLI version of source relationship verification.
        """
        try:
            # Get input file modification time
            input_mtime = os.path.getmtime(input_file)
            output_mtime = os.path.getmtime(output_path)
            
            # Output should be created after input (with tolerance)
            if output_mtime < input_mtime - 60:  # Allow 1 minute tolerance
                logger.debug(f"Output {os.path.basename(output_path)} is older than input {os.path.basename(input_file)}")
                return False
            
            # Get input duration for comparison
            try:
                from .ffmpeg_utils import FFmpegUtils
                input_duration = FFmpegUtils.get_video_duration(input_file)
            except Exception:
                input_duration = None
            
            # Type-specific validation (simplified for CLI)
            if pattern_type.endswith('.mp4') and input_duration:
                try:
                    output_duration = FFmpegUtils.get_video_duration(output_path)
                    duration_diff = abs(input_duration - output_duration)
                    if duration_diff > 2.0:  # Allow 2 second difference
                        logger.debug(f"MP4 duration mismatch: {duration_diff:.1f}s difference")
                        return False
                except Exception:
                    pass  # If we can't verify, assume valid
            
            elif pattern_type.endswith('_segments') and input_duration:
                # Check segment count makes sense for input duration
                try:
                    segment_files = [f for f in os.listdir(output_path) if f.lower().endswith('.gif')]
                    expected_segments = max(1, int(input_duration / 20))
                    actual_segments = len(segment_files)
                    
                    if actual_segments < 1 or actual_segments > expected_segments * 3:
                        logger.debug(f"Segment count mismatch: {actual_segments} segments for {input_duration:.1f}s video")
                        return False
                except Exception:
                    pass  # If we can't verify, assume valid
            
            return True
            
        except Exception as e:
            logger.debug(f"Error verifying CLI output source relationship: {e}")
            return True  # Default to valid on error
    
    def _run_automated_workflow(self, args: argparse.Namespace):
        """Handle automated workflow command"""
        logger.info("Starting automated workflow...")
        
        if getattr(args, 'debug', False):
            print("\n" + "="*60)
            print("AUTOMATED VIDEO PROCESSING WORKFLOW")
            print("="*60)
            in_dir_display = os.path.abspath(args.input_dir) if getattr(args, 'input_dir', None) else os.path.abspath('input')
            print(f"Input directory:    {in_dir_display}")
            # Respect global --output-dir if provided
            out_dir_display = os.path.abspath(args.output_dir) if getattr(args, 'output_dir', None) else os.path.abspath('output')
            print(f"Output directory:   {out_dir_display}")
            try:
                temp_display = os.path.abspath(self.config.get_temp_dir()) if hasattr(self, 'config') and self.config else os.path.abspath('temp')
            except Exception:
                temp_display = os.path.abspath('temp')
            print(f"Temp directory:     {temp_display}")
            print(f"Check interval:     {args.check_interval} seconds")
            print(f"Max file size:      {args.max_size} MB")
            print("="*60)
            print("\nPlace video files in the 'input' directory to process them automatically.")
            print("Press Ctrl+C to stop the workflow gracefully.\n")
        else:
            from .logger_setup import get_app_base_dir
            base_dir = get_app_base_dir()
            out_dir_display = os.path.abspath(args.output_dir) if getattr(args, 'output_dir', None) else os.path.join(base_dir, 'output')
            in_dir_display = os.path.abspath(args.input_dir) if getattr(args, 'input_dir', None) else os.path.join(base_dir, 'input')
            print(f"Watching '{in_dir_display}' → '{out_dir_display}' every {args.check_interval}s (max {args.max_size}MB). Ctrl+C to stop.")
        
        try:
            self.automated_workflow.run_automated_workflow(
                check_interval=args.check_interval,
                max_size_mb=args.max_size,
                verbose=getattr(args, 'debug', False),
                max_files=getattr(args, 'max_files', None),
                input_dir=getattr(args, 'input_dir', None),
                output_dir=getattr(args, 'output_dir', None),
                no_cache=getattr(args, 'no_cache', False),
                max_input_size_bytes=self._parse_max_input_size_to_bytes(getattr(args, 'max_input_size', None))
            )
        except KeyboardInterrupt:
            logger.info("Automated workflow stopped by user")
        except Exception as e:
            logger.error(f"Automated workflow error: {e}")
            raise

    def _handle_cache_command(self, args: argparse.Namespace):
        """Handle cache management commands"""
        if args.cache_action == 'clear':
            self._clear_cache()
        elif args.cache_action == 'stats':
            self._show_cache_stats()
        elif args.cache_action == 'validate':
            self._validate_cache()
        else:
            logger.error("Cache command requires an action (clear|stats|validate)")

    def _clear_cache(self):
        """Clear all cache entries."""
        try:
            self.automated_workflow.clear_cache()
            print("✅ Cache cleared successfully")
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            logger.error(f"Failed to clear cache: {e}")

    def _show_cache_stats(self):
        """Show cache statistics."""
        try:
            stats = self.automated_workflow.get_cache_stats()
            print("\n📊 Cache Statistics:")
            print("=" * 50)
            print(f"Total Entries:        {stats['total_entries']}")
            print(f"Recent Entries:       {stats['current_session_entries']} (<1 hour)")
            print(f"Older Entries:        {stats['old_entries']} (≥1 hour)")
            print(f"Cache Age:            {stats['cache_age_info']}")
            print(f"Session Start:        {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['session_start_time']))}")
            print("=" * 50)
        except Exception as e:
            print(f"❌ Failed to get cache stats: {e}")
            logger.error(f"Failed to get cache stats: {e}")

    def _validate_cache(self):
        """Validate cache entries and remove invalid ones."""
        try:
            result = self.automated_workflow.validate_and_clean_cache()
            print(f"\n🔍 Cache Validation Results:")
            print("=" * 50)
            print(f"Total Entries:     {result['total']}")
            print(f"Valid Entries:     {result['valid']}")
            print(f"Invalid Entries:   {result['invalid']}")
            print(f"Cleaned Entries:   {result['cleaned']}")
            print("=" * 50)
            
            if result['cleaned'] > 0:
                print(f"✅ Successfully cleaned {result['cleaned']} invalid cache entries")
            else:
                print("✅ All cache entries are valid")
                
        except Exception as e:
            print(f"❌ Failed to validate cache: {e}")
            logger.error(f"Failed to validate cache: {e}")

    def _handle_performance_command(self, args: argparse.Namespace):
        """Handle performance monitoring commands"""
        from .performance_controls import PerformanceController
        
        # Initialize performance controller if not already done
        if not hasattr(self, 'performance_controller'):
            self.performance_controller = PerformanceController(self.config)
        
        if args.perf_action == 'summary':
            self._show_performance_summary()
        elif args.perf_action == 'bottlenecks':
            self._show_performance_bottlenecks()
        elif args.perf_action == 'trends':
            self._show_performance_trends()
        elif args.perf_action == 'report':
            self._generate_performance_report()
        elif args.perf_action == 'profile':
            self._handle_performance_profile_command(args)
        elif args.perf_action == 'metrics':
            self._handle_performance_metrics_command(args)
        else:
            logger.error("Performance command requires an action (summary|bottlenecks|trends|report|profile|metrics)")

    def _show_performance_summary(self):
        """Show performance summary."""
        try:
            summary = self.performance_controller.get_performance_summary()
            
            print("\n📊 Performance Summary:")
            print("=" * 60)
            print(f"Current Profile:      {summary['current_profile']}")
            print()
            
            recent = summary['recent_performance']
            print("Recent Performance:")
            print(f"  Total Sessions:     {recent['total_sessions']}")
            print(f"  Success Rate:       {recent['success_rate']}")
            print(f"  Avg Processing:     {recent['average_processing_time']}")
            print()
            
            cache = summary['cache_performance']
            print("Cache Performance:")
            print(f"  Hit Rate:           {cache['hit_rate']}")
            print(f"  Total Lookups:      {cache['total_lookups']}")
            print(f"  Efficiency:         {cache['efficiency_status']}")
            print()
            
            system = summary['system_status']
            print("System Status:")
            print(f"  Memory Usage:       {system['memory_usage']}")
            print(f"  Memory Status:      {system['memory_status']}")
            print(f"  Available Memory:   {system['available_memory']}")
            print()
            
            print("Recommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Failed to get performance summary: {e}")
            logger.error(f"Failed to get performance summary: {e}")

    def _show_performance_bottlenecks(self):
        """Show performance bottleneck analysis."""
        try:
            report = self.performance_controller.get_bottleneck_report()
            
            print("\n🔍 Performance Bottleneck Analysis:")
            print("=" * 60)
            print(f"Status: {report['status']}")
            
            if report['total_issues'] > 0:
                print(f"Total Issues: {report['total_issues']}")
                if report['critical_issues'] > 0:
                    print(f"Critical Issues: {report['critical_issues']}")
                if report['high_priority_issues'] > 0:
                    print(f"High Priority: {report['high_priority_issues']}")
                print()
                
                print("Top Recommendations:")
                for i, rec in enumerate(report['top_recommendations'], 1):
                    print(f"  {i}. {rec}")
                print()
                
                print("Detailed Issues:")
                for issue in report['detailed_bottlenecks']:
                    print(f"  • {issue['type']} ({issue['severity']})")
                    print(f"    {issue['description']}")
                    print(f"    Impact: {issue['impact']}")
                    print()
            else:
                print("✅ No performance bottlenecks detected")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Failed to analyze bottlenecks: {e}")
            logger.error(f"Failed to analyze bottlenecks: {e}")

    def _show_performance_trends(self):
        """Show performance trends analysis."""
        try:
            trends = self.performance_controller.analyzer.analyze_performance_trends(days=7)
            
            print("\n📈 Performance Trends (Last 7 Days):")
            print("=" * 60)
            
            summary = trends.get('session_summary', {})
            print(f"Total Sessions:       {summary.get('total_sessions', 0)}")
            print(f"Successful Sessions:  {summary.get('successful_sessions', 0)}")
            print(f"Success Rate:         {summary.get('success_rate', 0):.1f}%")
            print(f"Average Duration:     {summary.get('avg_duration', 0):.1f}s")
            print()
            
            perf_trends = trends.get('performance_trends', {})
            duration_trend = perf_trends.get('duration_trend', {})
            print("Performance Trends:")
            print(f"  Duration Trend:     {duration_trend.get('direction', 'unknown')}")
            print(f"  Trend Significance: {duration_trend.get('significance', 'unknown')}")
            
            utilization_trend = perf_trends.get('utilization_trend', {})
            print(f"  Size Utilization:   {utilization_trend.get('avg_utilization', 0):.1f}%")
            
            quality_trend = perf_trends.get('quality_trend', {})
            if quality_trend.get('avg_quality'):
                print(f"  Average Quality:    {quality_trend.get('avg_quality', 0):.1f}")
            print()
            
            # Platform analysis
            platform_analysis = trends.get('platform_analysis', {})
            if platform_analysis:
                print("Platform Performance:")
                for platform, stats in platform_analysis.items():
                    print(f"  {platform}:")
                    print(f"    Sessions: {stats['session_count']}")
                    print(f"    Avg Duration: {stats['avg_duration']:.1f}s")
                    print(f"    Success Rate: {stats['success_rate']:.1f}%")
                print()
            
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Failed to analyze trends: {e}")
            logger.error(f"Failed to analyze trends: {e}")

    def _generate_performance_report(self):
        """Generate comprehensive performance report."""
        try:
            report = self.performance_controller.export_performance_report()
            
            print("\n📋 Comprehensive Performance Report:")
            print("=" * 60)
            print(f"Report Generated: {report['report_timestamp']}")
            print()
            
            # Executive summary
            user_summary = report['user_summary']
            print(f"Current Profile: {user_summary['current_profile']}")
            print()
            
            recent = user_summary['recent_performance']
            print("Recent Performance:")
            print(f"  Sessions: {recent['total_sessions']}")
            print(f"  Success Rate: {recent['success_rate']}")
            print(f"  Avg Time: {recent['average_processing_time']}")
            print()
            
            # Optimization potential
            optimization = report['optimization_potential']
            print(f"Optimization Potential: {optimization}")
            print()
            
            # Key recommendations
            print("Key Recommendations:")
            for i, rec in enumerate(report['key_recommendations'], 1):
                print(f"  {i}. {rec}")
            print()
            
            # Save detailed report to file
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"performance_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📄 Detailed report saved to: {report_file}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Failed to generate performance report: {e}")
            logger.error(f"Failed to generate performance report: {e}")

    def _handle_performance_profile_command(self, args: argparse.Namespace):
        """Handle performance profile commands."""
        try:
            if args.profile_action == 'list':
                profiles = self.performance_controller.get_available_profiles()
                
                print("\n⚙️  Available Performance Profiles:")
                print("=" * 60)
                
                for profile in profiles:
                    print(f"Profile: {profile['name']}")
                    print(f"  Description: {profile['description']}")
                    print(f"  Recommended for:")
                    for rec in profile['recommended_for']:
                        print(f"    • {rec}")
                    print()
                
                print("=" * 60)
                
            elif args.profile_action == 'show':
                current_profile = self.performance_controller.monitor.get_performance_profile()
                
                if current_profile:
                    print(f"\n⚙️  Current Performance Profile: {current_profile.name}")
                    print("=" * 50)
                    print(f"Max Quality Eval Time: {current_profile.max_quality_evaluation_time}s")
                    print(f"Max Iterations:        {current_profile.max_compression_iterations}")
                    print(f"Fast Estimation:       {'Enabled' if current_profile.enable_fast_estimation else 'Disabled'}")
                    print(f"Cache Enabled:         {'Yes' if current_profile.cache_enabled else 'No'}")
                    print(f"Parallel Processing:   {'Yes' if current_profile.parallel_processing else 'No'}")
                    print(f"Memory Optimization:   {'Yes' if current_profile.memory_optimization else 'No'}")
                    print("=" * 50)
                else:
                    print("❌ No performance profile is currently set")
                    
            elif args.profile_action == 'set':
                success = self.performance_controller.set_performance_profile(args.profile_name)
                
                if success:
                    print(f"✅ Performance profile set to: {args.profile_name}")
                else:
                    print(f"❌ Failed to set performance profile: {args.profile_name}")
                    print("Available profiles: fast, balanced, quality")
            
        except Exception as e:
            print(f"❌ Failed to handle profile command: {e}")
            logger.error(f"Failed to handle profile command: {e}")

    def _handle_performance_metrics_command(self, args: argparse.Namespace):
        """Handle performance metrics commands."""
        try:
            if args.metrics_action == 'export':
                from datetime import datetime, timedelta
                
                # Export last 30 days by default
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                metrics = self.performance_controller.monitor.export_metrics(start_date, end_date)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                metrics_file = f"performance_metrics_{timestamp}.json"
                
                import json
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
                
                print(f"📊 Performance metrics exported to: {metrics_file}")
                print(f"Sessions exported: {len(metrics.get('sessions', []))}")
                
            elif args.metrics_action == 'clear':
                days = args.days
                removed_count = self.performance_controller.monitor.clear_metrics(days)
                
                print(f"🗑️  Cleared {removed_count} metrics files older than {days} days")
            
        except Exception as e:
            print(f"❌ Failed to handle metrics command: {e}")
            logger.error(f"Failed to handle metrics command: {e}")

    def _handle_open_command(self, args: argparse.Namespace):
        """Handle open folder command"""
        import subprocess
        import platform
        
        # Check if any folder flag is set
        if not any([args.input, args.output, args.logs, args.config]):
            print("❌ No folder specified. Use -i, -o, -l, or -c to open folders.")
            print("   Examples:")
            print("     disdrop open -i        # Open input folder")
            print("     disdrop open -o        # Open output folder")
            print("     disdrop open -l        # Open logs folder")
            print("     disdrop open -c        # Open config folder")
            print("     disdrop open -i -o -l  # Open multiple folders")
            sys.exit(1)
        
        # Determine folder paths
        folders_to_open = []
        
        if args.input:
            from .logger_setup import get_app_base_dir
            input_dir = os.path.join(get_app_base_dir(), 'input')
            folders_to_open.append(('input', input_dir))
        
        if args.output:
            from .logger_setup import get_app_base_dir
            output_dir = os.path.join(get_app_base_dir(), 'output')
            folders_to_open.append(('output', output_dir))
        
        if args.logs:
            from .logger_setup import get_app_base_dir
            logs_dir = os.path.join(get_app_base_dir(), 'logs')
            folders_to_open.append(('logs', logs_dir))
        
        if args.config:
            from .logger_setup import get_app_base_dir
            config_dir = os.path.join(get_app_base_dir(), 'config')
            folders_to_open.append(('config', config_dir))
        
        # Create folders if they don't exist
        for folder_name, folder_path in folders_to_open:
            if not os.path.exists(folder_path):
                try:
                    os.makedirs(folder_path, exist_ok=True)
                    print(f"✅ Created {folder_name} folder: {folder_path}")
                except Exception as e:
                    print(f"❌ Failed to create {folder_name} folder: {e}")
                    continue
        
        # Detect platform and open folders
        system = platform.system()
        
        for folder_name, folder_path in folders_to_open:
            try:
                if system == 'Windows':
                    # Windows: use explorer
                    subprocess.Popen(['explorer', folder_path])
                elif system == 'Darwin':
                    # macOS: use open
                    subprocess.Popen(['open', folder_path])
                else:
                    # Linux: use xdg-open
                    subprocess.Popen(['xdg-open', folder_path])
                
                print(f"📁 Opened {folder_name} folder: {folder_path}")
                
            except Exception as e:
                print(f"❌ Failed to open {folder_name} folder: {e}")
                logger.error(f"Failed to open {folder_name} folder: {e}")
        
        # Exit immediately after opening folders
        sys.exit(0)

def main():
    """Entry point for the CLI application"""
    cli = VideoCompressorCLI()
    cli.main()

if __name__ == '__main__':
    main() 