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
from typing import Dict, Any, Optional
import logging
import time

from .logger_setup import setup_logging, get_logger, _cleanup_old_logs
from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .video_compressor import DynamicVideoCompressor
from .gif_generator import GifGenerator
from .gif_optimizer_advanced import AdvancedGifOptimizer
from .automated_workflow import AutomatedWorkflow
from .file_validator import FileValidator

logger = None  # Will be initialized after logging setup

class VideoCompressorCLI:
    def __init__(self):
        self.config = None
        self.hardware = None
        self.video_compressor = None
        self.gif_generator = None
        self.automated_workflow = None
        
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
        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            if logger:
                logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
            else:
                print(f"\nReceived {signal_name} signal, cleaning up...")
            
            # Request shutdown for all components
            if hasattr(self, 'gif_generator') and self.gif_generator:
                self.gif_generator.request_shutdown()
                # Also shutdown the GIF optimizer
                if hasattr(self.gif_generator, 'optimizer') and hasattr(self.gif_generator.optimizer, 'request_shutdown'):
                    self.gif_generator.optimizer.request_shutdown()
            if hasattr(self, 'video_compressor') and self.video_compressor:
                self.video_compressor.request_shutdown()
            if hasattr(self, 'automated_workflow') and self.automated_workflow:
                # Automated workflow uses shutdown_requested flag instead of method
                self.automated_workflow.shutdown_requested = True
            
            # Clean up any remaining temp files
            self._cleanup_temp_files_on_exit()
            
            if logger:
                logger.info("Graceful shutdown completed")
            sys.exit(0)
        
        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)  # Termination request
    
    def _cleanup_temp_files_on_exit(self):
        """Clean up temporary files on program exit"""
        try:
            if hasattr(self, 'config') and self.config:
                temp_dir = self.config.get_temp_dir()
                if temp_dir and os.path.exists(temp_dir):
                    # Look for segment temp folders and other temp files
                    for item in os.listdir(temp_dir):
                        item_path = os.path.join(temp_dir, item)
                        try:
                            if os.path.isdir(item_path) and '_segments_temp' in item:
                                # Clean up segment temp folders
                                shutil.rmtree(item_path)
                                if logger:
                                    logger.info(f"Cleaned up temp segment folder: {item}")
                            elif os.path.isfile(item_path) and ('temp_' in item or item.startswith('candidate_')):
                                # Clean up other temp files
                                os.remove(item_path)
                                if logger:
                                    logger.debug(f"Cleaned up temp file: {item}")
                        except Exception as e:
                            if logger:
                                logger.warning(f"Could not clean up {item}: {e}")
        except Exception as e:
            if logger:
                logger.warning(f"Error during temp file cleanup: {e}")
    
    def _parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Video Compressor - Compress videos and create GIFs for social media platforms",
            epilog="Examples:\n"
                   "  %(prog)s compress input.mp4 output.mp4 --platform instagram\n"
                   "  %(prog)s gif input.mp4 output.gif --platform twitter --duration 10\n"
                   "  %(prog)s batch-compress *.mp4 --platform tiktok --output-dir compressed/\n"
                   "  %(prog)s hardware-info\n",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        # Global options
        parser.add_argument('--config-dir', default='config',
                          help='Configuration directory (default: config)')
        # Default to quieter console; use --debug to enable verbose output
        parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default=None, help='Override logging level (default: WARNING to console, DEBUG to file)')
        parser.add_argument('--debug', action='store_true',
                          help='Enable verbose debug output in console and logs')
        parser.add_argument('--temp-dir', help='Temporary directory for processing')
        parser.add_argument('--max-size', type=float, metavar='MB',
                          help='Maximum output file size in MB (overrides platform defaults)')
        parser.add_argument('--max-files', type=int, metavar='N',
                          help='Maximum number of files to process before exiting')
        parser.add_argument('--output-dir', help='Output directory for generated files (default varies by mode, typically ./output)')
        parser.add_argument('--force-software', action='store_true',
                          help='Force software encoding (bypass hardware acceleration)')
        parser.add_argument('--no-cache', action='store_true',
                          help='Do not use success cache; verify and process all files even if previously successful')
        # Segmentation preference: prefer a fixed number of segments (1-10)
        parser.add_argument('--prefer-segments', type=int, choices=list(range(1, 11)), metavar='N',
                          help='Prefer N segments (1-10). If impossible, fall back to normal operations')
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Compress video command
        compress_parser = subparsers.add_parser('compress', help='Compress a video file')
        compress_parser.add_argument('input', help='Input video file')
        compress_parser.add_argument('output', help='Output video file')
        compress_parser.add_argument('--platform', choices=['instagram', 'twitter', 'tiktok', 'youtube_shorts', 'facebook'],
                                   help='Target social media platform')
        compress_parser.add_argument('--encoder', help='Force specific encoder (overrides hardware detection)')
        compress_parser.add_argument('--quality', type=int, metavar='CRF', help='Quality setting (CRF value, lower = better)')
        compress_parser.add_argument('--bitrate', help='Target bitrate (e.g., 1000k)')
        compress_parser.add_argument('--resolution', help='Target resolution (e.g., 1080x1080)')
        compress_parser.add_argument('--fps', type=int, help='Target frame rate')
        
        # Create GIF command
        gif_parser = subparsers.add_parser('gif', help='Create GIF from video')
        gif_parser.add_argument('input', help='Input video file')
        gif_parser.add_argument('output', help='Output GIF file')
        gif_parser.add_argument('--platform', choices=['twitter', 'discord', 'slack'],
                               help='Target platform for GIF optimization')
        gif_parser.add_argument('--start', type=float, default=0, metavar='SECONDS',
                               help='Start time in seconds (default: 0)')
        gif_parser.add_argument('--duration', type=float, metavar='SECONDS',
                               help='Duration in seconds (default: platform/config limit)')
        gif_parser.add_argument('--max-size', type=float, metavar='MB',
                               help='Maximum file size in MB (enables quality optimization)')
        gif_parser.add_argument('--fps', type=int, help='Frame rate for GIF')
        gif_parser.add_argument('--width', type=int, help='GIF width in pixels')
        gif_parser.add_argument('--height', type=int, help='GIF height in pixels')
        gif_parser.add_argument('--colors', type=int, help='Number of colors in GIF palette')
        
        # Batch compress command
        batch_parser = subparsers.add_parser('batch-compress', help='Compress multiple video files')
        batch_parser.add_argument('input_pattern', help='Input files pattern (e.g., *.mp4)')
        batch_parser.add_argument('--output-dir', help='Output directory (overrides global)')
        batch_parser.add_argument('--platform', choices=['instagram', 'twitter', 'tiktok', 'youtube_shorts', 'facebook'],
                                 help='Target social media platform')
        batch_parser.add_argument('--suffix', default='_compressed', help='Suffix for output files')
        batch_parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel processes')
        
        # Batch GIF command
        batch_gif_parser = subparsers.add_parser('batch-gif', help='Create GIFs from multiple videos')
        batch_gif_parser.add_argument('input_pattern', help='Input files pattern (e.g., *.mp4)')
        batch_gif_parser.add_argument('--output-dir', help='Output directory (overrides global)')
        batch_gif_parser.add_argument('--platform', choices=['twitter', 'discord', 'slack'],
                                     help='Target platform for GIF optimization')
        batch_gif_parser.add_argument('--duration', type=float, metavar='SECONDS',
                                     help='Duration for each GIF')
        batch_gif_parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel processes')
        
        # Optimize existing GIF
        optimize_parser = subparsers.add_parser('optimize-gif', help='Optimize an existing GIF file')
        optimize_parser.add_argument('input', help='Input GIF file')
        optimize_parser.add_argument('output', help='Output GIF file')
        
        # Quality-optimized GIF command
        quality_gif_parser = subparsers.add_parser('quality-gif', help='Create GIF with iterative quality optimization')
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
        
        # Hardware info command
        subparsers.add_parser('hardware-info', help='Display hardware acceleration information')
        
        # Config command
        config_parser = subparsers.add_parser('config', help='Configuration management')
        config_subparsers = config_parser.add_subparsers(dest='config_action')
        config_subparsers.add_parser('show', help='Show current configuration')
        config_subparsers.add_parser('validate', help='Validate configuration files')
        
        # Automated workflow command
        auto_parser = subparsers.add_parser('auto', help='Run automated workflow (process videos from input folder)')
        auto_parser.add_argument('--check-interval', type=int, default=5, metavar='SECONDS',
                                help='How often to check for new files (default: 5 seconds)')
        auto_parser.add_argument('--max-size', type=float, default=10.0, metavar='MB',
                                help='Maximum output file size in MB (default: 10.0)')
        auto_parser.add_argument('--no-cache', action='store_true',
                                help='Do not use success cache in automated workflow')
        
        # Cache management command
        cache_parser = subparsers.add_parser('cache', help='Cache management operations')
        cache_subparsers = cache_parser.add_subparsers(dest='cache_action', help='Cache operations')
        cache_subparsers.add_parser('clear', help='Clear all cache entries')
        cache_subparsers.add_parser('stats', help='Show cache statistics')
        
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
            self.advanced_optimizer = AdvancedGifOptimizer(self.config)
            self.file_validator = FileValidator()
            self.automated_workflow = AutomatedWorkflow(self.config, self.hardware)
            # Propagate preferred segments (legacy single-segment retained via value 1)
            try:
                preferred = getattr(args, 'prefer_segments', None)
                if preferred is not None:
                    self.automated_workflow.preferred_segments = int(preferred)
                else:
                    self.automated_workflow.preferred_segments = None
            except Exception:
                self.automated_workflow.preferred_segments = None
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _extract_config_overrides(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract configuration overrides from CLI arguments"""
        overrides = {}
        
        # Temporary directory
        if hasattr(args, 'temp_dir') and args.temp_dir:
            overrides['video_compression.temp_dir'] = args.temp_dir
        
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
        
        return overrides
    
    def _execute_command(self, args: argparse.Namespace):
        """Execute the requested command"""
        
        if args.command == 'compress':
            self._compress_video(args)
        
        elif args.command == 'gif':
            self._create_gif(args)
        
        elif args.command == 'batch-compress':
            self._batch_compress(args)
        
        elif args.command == 'batch-gif':
            self._batch_gif(args)
        
        elif args.command == 'optimize-gif':
            self._optimize_gif(args)
        
        elif args.command == 'quality-gif':
            self._create_quality_gif(args)
        
        elif args.command == 'hardware-info':
            self._show_hardware_info()
        
        elif args.command == 'config':
            self._handle_config_command(args)
        
        elif args.command == 'auto':
            self._run_automated_workflow(args)
        
        elif args.command == 'cache':
            self._handle_cache_command(args)
        
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
    
    def _compress_video(self, args: argparse.Namespace):
        """Handle video compression command"""
        logger.info(f"Compressing video: {args.input} -> {args.output}")
        
        try:
            # Validate input file
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Input file not found: {args.input}")
            
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
        
        # Find input files
        input_files = glob.glob(args.input_pattern)
        if not input_files:
            logger.error(f"No files found matching pattern: {args.input_pattern}")
            return

        # Apply processing limit if specified
        if getattr(args, 'max_files', None):
            input_files = input_files[: int(args.max_files)]
        
        logger.info(f"Found {len(input_files)} files to compress")
        
        # Resolve output directory: prefer provided --output-dir, else default to ./output
        effective_output_dir = args.output_dir if getattr(args, 'output_dir', None) else os.path.join(os.getcwd(), 'output')
        os.makedirs(effective_output_dir, exist_ok=True)
        
        # Determine number of parallel processes
        max_workers = args.parallel or min(4, len(input_files))
        
        def compress_single(input_file):
            try:
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
                logger.error(f"Failed to compress {input_file}: {e}")
                return {'success': False, 'input': input_file, 'error': str(e)}
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(compress_single, input_files))
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info(f"Batch compression completed: {successful} successful, {failed} failed")
    
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
        
        def create_single_gif(input_file):
            try:
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
                logger.error(f"Failed to create GIF from {input_file}: {e}")
                return {'success': False, 'input': input_file, 'error': str(e)}
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(create_single_gif, input_files))
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        logger.info(f"Batch GIF creation completed: {successful} successful, {failed} failed")
    
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
            
            # Create quality-optimized GIF
            results = self.advanced_optimizer.optimize_gif_with_quality_target(
                input_video=args.input,
                output_path=args.output,
                max_size_mb=args.target_size,
                platform=args.platform,
                start_time=args.start,
                duration=args.duration,
                quality_preference=args.quality_preference
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
        
        for filename in os.listdir(segments_folder):
            if filename.lower().endswith('.gif'):
                gif_path = os.path.join(segments_folder, filename)
                try:
                    is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                        gif_path,
                        original_path=None,
                        max_size_mb=max_size_mb
                    )
                    if is_valid:
                        valid_gifs.append(gif_path)
                    else:
                        invalid_gifs.append(gif_path)
                        logger.debug(f"Invalid GIF {filename}: {error_msg}")
                except Exception as e:
                    logger.warning(f"Error validating GIF {filename}: {e}")
                    invalid_gifs.append(gif_path)
        
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
                # Validate single GIF file
                is_valid, _ = self.file_validator.is_valid_gif(output_path, max_size_mb=10.0)
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
            print(f"Input directory:    {os.path.abspath('input')}")
            # Respect global --output-dir if provided
            out_dir_display = os.path.abspath(args.output_dir) if getattr(args, 'output_dir', None) else os.path.abspath('output')
            print(f"Output directory:   {out_dir_display}")
            print(f"Temp directory:     {os.path.abspath('temp')}")
            print(f"Check interval:     {args.check_interval} seconds")
            print(f"Max file size:      {args.max_size} MB")
            print("="*60)
            print("\nPlace video files in the 'input' directory to process them automatically.")
            print("Press Ctrl+C to stop the workflow gracefully.\n")
        else:
            out_dir_display = os.path.abspath(args.output_dir) if getattr(args, 'output_dir', None) else os.path.abspath('output')
            print(f"Watching '{os.path.abspath('input')}' → '{out_dir_display}' every {args.check_interval}s (max {args.max_size}MB). Ctrl+C to stop.")
        
        try:
            self.automated_workflow.run_automated_workflow(
                check_interval=args.check_interval,
                max_size_mb=args.max_size,
                verbose=getattr(args, 'debug', False),
                max_files=getattr(args, 'max_files', None),
                output_dir=getattr(args, 'output_dir', None),
                no_cache=getattr(args, 'no_cache', False)
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
        else:
            logger.error("Cache command requires an action (clear|stats)")

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

def main():
    """Entry point for the CLI application"""
    cli = VideoCompressorCLI()
    cli.main()

if __name__ == '__main__':
    main() 