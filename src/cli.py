"""
Command Line Interface for Video Compressor
Main entry point with argument parsing and command execution
"""

import argparse
import os
import sys
import traceback
from typing import Dict, Any, Optional
import logging

from .logger_setup import setup_logging, get_logger
from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .video_compressor import DynamicVideoCompressor
from .gif_generator import GifGenerator
from .gif_optimizer_advanced import AdvancedGifOptimizer
from .automated_workflow import AutomatedWorkflow

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
            
            # Setup logging
            global logger
            logger = setup_logging(log_level=args.log_level)
            
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
        parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default='INFO', help='Logging level (default: INFO)')
        parser.add_argument('--temp-dir', help='Temporary directory for processing')
        parser.add_argument('--max-size', type=float, metavar='MB',
                          help='Maximum output file size in MB (overrides platform defaults)')
        
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
        batch_parser.add_argument('--output-dir', required=True, help='Output directory')
        batch_parser.add_argument('--platform', choices=['instagram', 'twitter', 'tiktok', 'youtube_shorts', 'facebook'],
                                 help='Target social media platform')
        batch_parser.add_argument('--suffix', default='_compressed', help='Suffix for output files')
        batch_parser.add_argument('--parallel', type=int, metavar='N', help='Number of parallel processes')
        
        # Batch GIF command
        batch_gif_parser = subparsers.add_parser('batch-gif', help='Create GIFs from multiple videos')
        batch_gif_parser.add_argument('input_pattern', help='Input files pattern (e.g., *.mp4)')
        batch_gif_parser.add_argument('--output-dir', required=True, help='Output directory')
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
        
        # Set default command if none provided
        args = parser.parse_args()
        if not args.command:
            # Default to automated workflow if no command specified
            args.command = 'auto'
            args.check_interval = 5
            args.max_size = 10.0
            
        return args
    
    def _initialize_components(self, args: argparse.Namespace):
        """Initialize all components"""
        logger.info("Initializing Video Compressor...")
        
        # Initialize configuration manager
        config_dir = args.config_dir
        self.config = ConfigManager(config_dir)
        
        # Update config with CLI arguments
        cli_overrides = self._extract_config_overrides(args)
        if cli_overrides:
            self.config.update_from_args(cli_overrides)
        
        # Validate configuration
        if not self.config.validate_config():
            raise RuntimeError("Configuration validation failed")
        
        # Initialize hardware detector
        logger.info("Detecting hardware acceleration capabilities...")
        self.hardware = HardwareDetector()
        
        # Initialize processors
        self.video_compressor = DynamicVideoCompressor(self.config, self.hardware)
        self.gif_generator = GifGenerator(self.config)
        self.advanced_optimizer = AdvancedGifOptimizer(self.config)
        self.automated_workflow = AutomatedWorkflow(self.config, self.hardware, 
                                                   self.video_compressor, self.gif_generator)
        
        logger.info("Initialization complete")
    
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
            
            # Display results
            self._display_gif_results(results)
            
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
        
        logger.info(f"Found {len(input_files)} files to compress")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Determine number of parallel processes
        max_workers = args.parallel or min(4, len(input_files))
        
        def compress_single(input_file):
            try:
                # Generate output filename
                basename = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(args.output_dir, f"{basename}{args.suffix}.mp4")
                
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
                
                # Create GIF
                result = self.gif_generator.create_gif(
                    input_video=input_file,
                    output_path=output_file,
                    platform=args.platform,
                    max_size_mb=args.max_size,
                    duration=args.duration
                )
                
                return {'success': True, 'input': input_file, 'output': output_file, 'result': result}
                
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
            logger.info("✓ Configuration is valid")
        else:
            logger.error("✗ Configuration validation failed")
            sys.exit(1)
    
    def _display_compression_results(self, results: Dict[str, Any]):
        """Display video compression results"""
        print("\n" + "="*60)
        print("VIDEO COMPRESSION RESULTS")
        print("="*60)
        print(f"Input File:       {results['input_file']}")
        print(f"Output File:      {results['output_file']}")
        print(f"Method:           {results['method']}")
        print(f"Original Size:    {results['original_size_mb']:.2f} MB")
        print(f"Compressed Size:  {results['compressed_size_mb']:.2f} MB")
        print(f"Compression:      {results['compression_ratio']:.1f}% reduction")
        print(f"Space Saved:      {results['space_saved_mb']:.2f} MB")
        
        video_info = results['video_info']
        print(f"Resolution:       {video_info['width']}x{video_info['height']}")
        print(f"Duration:         {video_info['duration']:.1f} seconds")
        print(f"Frame Rate:       {video_info['fps']:.1f} fps")
        print("="*60)
    
    def _display_gif_results(self, results: Dict[str, Any]):
        """Display GIF creation results"""
        print("\n" + "="*60)
        print("GIF CREATION RESULTS")
        print("="*60)
        print(f"Input Video:      {results['input_video']}")
        print(f"Output GIF:       {results['output_gif']}")
        print(f"File Size:        {results['file_size_mb']:.2f} MB")
        print(f"Duration:         {results['duration_seconds']:.1f} seconds")
        print(f"Frame Count:      {results['frame_count']}")
        print(f"Frame Rate:       {results['fps']} fps")
        print(f"Resolution:       {results['width']}x{results['height']}")
        print(f"Colors:           {results['colors']}")
        print(f"Compression:      {results['compression_ratio']:.1f}% from original video")
        print("="*60)
    
    def _display_gif_optimization_results(self, results: Dict[str, Any]):
        """Display GIF optimization results"""
        print("\n" + "="*60)
        print("GIF OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Input GIF:        {results['input_gif']}")
        print(f"Output GIF:       {results['output_gif']}")
        print(f"Method:           {results['method']}")
        print(f"Original Size:    {results['original_size_mb']:.2f} MB")
        print(f"Optimized Size:   {results['optimized_size_mb']:.2f} MB")
        print(f"Compression:      {results['compression_ratio']:.1f}% reduction")
        print(f"Space Saved:      {results['space_saved_mb']:.2f} MB")
        print("="*60)
    
    def _display_quality_gif_results(self, results: Dict[str, Any]):
        """Display quality-optimized GIF results"""
        print("\n" + "="*60)
        print("QUALITY-OPTIMIZED GIF RESULTS")
        print("="*60)
        print(f"Output File:       {results.get('output_file', 'N/A')}")
        print(f"Target Size:       {results.get('target_size_mb', 'N/A')} MB")
        print(f"Actual Size:       {results.get('size_mb', 'N/A'):.2f} MB")
        print(f"Size Efficiency:   {results.get('size_efficiency', 'N/A'):.1%} of target")
        print(f"Quality Score:     {results.get('quality_score', 'N/A'):.2f}/10")
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
    
    def _run_automated_workflow(self, args: argparse.Namespace):
        """Handle automated workflow command"""
        logger.info("Starting automated workflow...")
        
        print("\n" + "="*60)
        print("AUTOMATED VIDEO PROCESSING WORKFLOW")
        print("="*60)
        print(f"Input directory:    {os.path.abspath('input')}")
        print(f"Output directory:   {os.path.abspath('output')}")
        print(f"Temp directory:     {os.path.abspath('temp')}")
        print(f"Check interval:     {args.check_interval} seconds")
        print(f"Max file size:      {args.max_size} MB")
        print("="*60)
        print("\nPlace video files in the 'input' directory to process them automatically.")
        print("Press Ctrl+C to stop the workflow gracefully.\n")
        
        try:
            self.automated_workflow.run_automated_workflow(
                check_interval=args.check_interval,
                max_size_mb=args.max_size
            )
        except KeyboardInterrupt:
            logger.info("Automated workflow stopped by user")
        except Exception as e:
            logger.error(f"Automated workflow error: {e}")
            raise

def main():
    """Entry point for the CLI application"""
    cli = VideoCompressorCLI()
    cli.main()

if __name__ == '__main__':
    main() 