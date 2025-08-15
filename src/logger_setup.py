"""
Logging Setup for Video Compressor
Initializes logging configuration from YAML file
"""

import os
import sys
import logging
import logging.config
import yaml
from colorama import init, Fore, Style
from typing import Optional
import codecs
import atexit
from datetime import datetime
import glob

# Initialize colorama for Windows compatibility
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to console output"""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.MAGENTA + Style.BRIGHT,
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
        
        return super().format(record)

class UTF8StreamHandler(logging.StreamHandler):
    """Stream handler that forces UTF-8 encoding"""
    
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Force UTF-8 encoding
            if hasattr(stream, 'buffer'):
                stream.buffer.write(msg.encode('utf-8'))
                stream.buffer.write(b'\n')
                stream.buffer.flush()
            else:
                stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)

class TeeStream:
    """A stream that writes to the original stream and a log file."""

    def __init__(self, original_stream, tee_file, name_label: str):
        self._original_stream = original_stream
        self._tee_file = tee_file
        # Preserve behavior expected by formatter logic
        self.name = name_label  # e.g., '<stdout>' or '<stderr>'
        # Expose encoding attribute like standard streams
        self.encoding = getattr(original_stream, 'encoding', 'utf-8')

    def write(self, data):
        try:
            if isinstance(data, bytes):
                text = data.decode('utf-8', errors='replace')
            else:
                text = data
            # Write to console first
            self._original_stream.write(text)
            # Then to file
            self._tee_file.write(text)
        except Exception:
            # Avoid crashing stdout on errors
            try:
                self._original_stream.write(text)
            except Exception:
                pass

    def flush(self):
        try:
            self._original_stream.flush()
        except Exception:
            pass
        try:
            self._tee_file.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self._original_stream.isatty()
        except Exception:
            return False

    def fileno(self):
        try:
            return self._original_stream.fileno()
        except Exception:
            raise OSError("fileno not supported for TeeStream")


# Module-level state to avoid double-wrapping
_TEE_ENABLED = False
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None
_TEE_FILE_HANDLE = None

def _enable_terminal_tee(logs_dir: str = "logs") -> str:
    """Enable teeing of stdout/stderr to a timestamped terminal log file.

    Returns the path to the terminal log file.
    """
    global _TEE_ENABLED, _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _TEE_FILE_HANDLE

    if _TEE_ENABLED:
        # Already enabled; try to expose the current file path if possible
        try:
            return getattr(_TEE_FILE_HANDLE, 'name', os.path.join(logs_dir, 'terminal.log'))
        except Exception:
            return os.path.join(logs_dir, 'terminal.log')

    os.makedirs(logs_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    terminal_log_path = os.path.join(logs_dir, f'terminal_{timestamp}.log')
    _TEE_FILE_HANDLE = open(terminal_log_path, 'a', encoding='utf-8', newline='')

    # Remember originals and wrap
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _TEE_FILE_HANDLE, name_label='<stdout>')
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _TEE_FILE_HANDLE, name_label='<stderr>')

    def _restore_streams():
        global _TEE_ENABLED, _TEE_FILE_HANDLE, _ORIGINAL_STDOUT, _ORIGINAL_STDERR
        try:
            if _TEE_FILE_HANDLE:
                _TEE_FILE_HANDLE.flush()
        except Exception:
            pass
        try:
            if _TEE_FILE_HANDLE:
                _TEE_FILE_HANDLE.close()
        except Exception:
            pass
        # Restore original streams
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
        _TEE_ENABLED = False
        _TEE_FILE_HANDLE = None

    atexit.register(_restore_streams)
    _TEE_ENABLED = True
    return terminal_log_path

def _cleanup_old_logs(logs_dir: str = "logs", keep_count: int = 5):
    """
    Clean up old log files, keeping only the last N executions
    
    Args:
        logs_dir: Directory containing log files
        keep_count: Number of most recent log files to keep
    """
    try:
        # Find all terminal log files (timestamped)
        terminal_logs = glob.glob(os.path.join(logs_dir, "terminal_*.log"))
        
        print(f"Log cleanup: Found {len(terminal_logs)} terminal log files")
        
        if len(terminal_logs) <= keep_count:
            print(f"Log cleanup: No cleanup needed, only {len(terminal_logs)} files found")
            return  # No cleanup needed
        
        # Sort by modification time (newest first)
        terminal_logs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Show what we're keeping and what we're removing
        files_to_keep = terminal_logs[:keep_count]
        files_to_remove = terminal_logs[keep_count:]
        
        print(f"Log cleanup: Keeping {len(files_to_keep)} most recent log files:")
        for f in files_to_keep:
            print(f"  - {os.path.basename(f)}")
        
        print(f"Log cleanup: Removing {len(files_to_remove)} old log files:")
        for f in files_to_remove:
            print(f"  - {os.path.basename(f)}")
        
        # Remove old files
        removed_count = 0
        for old_log in files_to_remove:
            try:
                os.remove(old_log)
                print(f"Log cleanup: Successfully removed {os.path.basename(old_log)}")
                removed_count += 1
            except Exception as e:
                print(f"Log cleanup warning: Could not remove {os.path.basename(old_log)}: {e}")
        
        print(f"Log cleanup: Completed. Removed {removed_count} old log files.")
                
    except Exception as e:
        print(f"Log cleanup error: {e}")
        import traceback
        traceback.print_exc()

def setup_logging(config_path: str = "config/logging.yaml", log_level: Optional[str] = None):
    """
    Setup logging configuration from YAML file
    
    Args:
        config_path: Path to logging configuration file
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    
    # Default logging configuration if file not found
    default_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'console': {
                'format': '%(asctime)s | %(levelname)-8s | %(message)s',
                'datefmt': '%H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'WARNING',
                'formatter': 'console',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': 'logs/video_compressor.log',
                'mode': 'a'
            },
            'error_file': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'detailed',
                'filename': 'logs/errors.log',
                'mode': 'a'
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file']
        }
    }
    
    try:
        # Load configuration from file
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                logging_config = config_data.get('logging', default_config)
        else:
            print(f"Warning: Logging config file not found at {config_path}, using default configuration")
            logging_config = default_config

        # Clean logs at program start
        try:
            video_log = os.path.join(logs_dir, 'video_compressor.log')
            error_log = os.path.join(logs_dir, 'errors.log')
            # Truncate files
            open(video_log, 'w', encoding='utf-8').close()
            open(error_log, 'w', encoding='utf-8').close()
        except Exception:
            pass

        # Enable terminal tee BEFORE applying logging config so console handler uses the tee stream
        terminal_log_path = _enable_terminal_tee(logs_dir)
        
        # Clean up old logs AFTER terminal tee is set up (keep only last 5 executions)
        _cleanup_old_logs(logs_dir, keep_count=5)

        # Default: quieter console (WARNING) and DEBUG to file unless overridden
        try:
            # Ensure console handler exists
            console_handler = logging_config['handlers'].get('console') if 'handlers' in logging_config else None
            if console_handler:
                console_handler['level'] = console_handler.get('level', 'WARNING')
                # Force console to WARNING when no explicit log_level passed
                if not log_level:
                    console_handler['level'] = 'WARNING'
            # Ensure file handlers are DEBUG by default
            for hname in ('file', 'error_file'):
                handler = logging_config.get('handlers', {}).get(hname)
                if handler:
                    if hname == 'file':
                        handler['level'] = 'DEBUG'
                    elif hname == 'error_file':
                        handler['level'] = 'ERROR'
            # Root level should be DEBUG to capture all into file handler
            if 'root' in logging_config:
                logging_config['root']['level'] = 'DEBUG'
        except Exception:
            pass

        # Override levels if an explicit log_level was provided (e.g., --debug)
        if log_level:
            log_level = log_level.upper()
            if 'root' in logging_config:
                logging_config['root']['level'] = log_level
            if 'handlers' in logging_config and 'console' in logging_config['handlers']:
                logging_config['handlers']['console']['level'] = log_level

        # Apply logging configuration
        try:
            logging.config.dictConfig(logging_config)
        except Exception as config_error:
            # If dictConfig fails, fall back to basic configuration
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            logger = logging.getLogger('video_compressor')
            logger.error(f"Failed to apply logging configuration: {config_error}")
            logger.info("Using basic logging configuration as fallback")
            return logger
        
        # Get the main logger and add colored formatter to console handler
        logger = logging.getLogger('video_compressor')
        
        # Find console handler and apply colored formatter
        try:
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and hasattr(handler.stream, 'name') and handler.stream.name == '<stdout>':
                    colored_formatter = ColoredFormatter(
                        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
                        datefmt='%H:%M:%S'
                    )
                    handler.setFormatter(colored_formatter)
        except Exception as format_error:
            logger.debug(f"Could not apply colored formatter: {format_error}")
        
        logger.info("Logging initialized")
        logger.debug(f"Terminal output tee: {terminal_log_path}")
        
        return logger
        
    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger('video_compressor')
        logger.error(f"Failed to load logging configuration: {e}")
        logger.info("Using basic logging configuration")
        return logger

def get_logger(name: str = None) -> logging.Logger:
    """Get a logger instance"""
    if name:
        return logging.getLogger(f'video_compressor.{name}')
    return logging.getLogger('video_compressor') 