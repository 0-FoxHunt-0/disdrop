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
import shutil
import importlib.resources as pkg_resources
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

class PILDebugFilter(logging.Filter):
    """Filter to exclude PIL/Pillow DEBUG messages from file logs"""
    
    def filter(self, record):
        # Filter out PIL/Pillow DEBUG messages
        if record.name.startswith('PIL') or record.name.startswith('Pillow'):
            if record.levelno == logging.DEBUG:
                return False
        return True

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
            # Write to console first with encoding-safe fallback
            try:
                self._original_stream.write(text)
            except Exception:
                # Windows cp1252 console may fail on some Unicode (e.g., U+29F8). Fallback to replacement.
                safe_text = text.encode(self.encoding or 'utf-8', errors='replace').decode(self.encoding or 'utf-8', errors='replace')
                try:
                    self._original_stream.write(safe_text)
                except Exception:
                    # As last resort, strip non-ASCII
                    ascii_text = ''.join(ch if ord(ch) < 128 else '?' for ch in text)
                    self._original_stream.write(ascii_text)
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

def get_package_base_dir() -> str:
    """Return the installed package directory (where packaged data like config lives).

    - In development (src-layout), this returns the `src` directory
      so that `src/config` is discoverable.
    - In installed environments, this returns the site-packages `disdrop` dir.
    """
    try:
        pkg_dir = os.path.abspath(os.path.dirname(__file__))
        return pkg_dir
    except Exception:
        return os.getcwd()


def get_app_base_dir() -> str:
    """Return the runtime base directory for user-writable data (logs/input/output/temp).

    Resolution order:
    1) DISDROP_BASE_DIR env var (created if missing)
    2) Project root when developing (parent of 'src')
    3) If installed: package dir if writable, else platform-specific user data dir
    """
    # 1) Explicit override via environment
    try:
        env = os.getenv("DISDROP_BASE_DIR")
        if env:
            os.makedirs(env, exist_ok=True)
            return env
    except Exception:
        pass

    try:
        pkg_dir = get_package_base_dir()
        # 2) In a src-layout checkout, prefer project root for runtime dirs
        if os.path.basename(pkg_dir).lower() == 'src':
            return os.path.abspath(os.path.join(pkg_dir, os.pardir))

        # 3) Installed: prefer per-user data directory (avoid writing inside site-packages)
        try:
            # Lazy import to avoid mandatory dependency at import-time if unused
            from platformdirs import user_data_dir
            base = user_data_dir(appname="Disdrop", appauthor="Disdrop")
            os.makedirs(base, exist_ok=True)
            return base
        except Exception:
            # Fallback to a directory under the user's home; as last resort, use CWD
            try:
                home_base = os.path.join(os.path.expanduser("~"), "Disdrop")
                os.makedirs(home_base, exist_ok=True)
                return home_base
            except Exception:
                return os.getcwd()
    except Exception:
        return os.getcwd()

def get_default_logs_dir() -> str:
    """Return the default logs directory under the package base dir."""
    return os.path.join(get_app_base_dir(), 'logs')

def _enable_terminal_tee(logs_dir: str = None) -> str:
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

    logs_dir = logs_dir or get_default_logs_dir()
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

def _cleanup_old_logs(logs_dir: str = None, keep_count: int = 5):
    """
    Clean up old log files, keeping only the last N executions
    
    Args:
        logs_dir: Directory containing log files
        keep_count: Number of most recent log files to keep
    """
    try:
        logs_dir = logs_dir or get_default_logs_dir()
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
    
    # Resolve logs directory to installed package location
    logs_dir = get_default_logs_dir()
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
        'loggers': {
            'video_compressor': {
                'level': 'DEBUG',
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'PIL': {
                'level': 'WARNING',
                'handlers': [],
                'propagate': False
            },
            'Pillow': {
                'level': 'WARNING',
                'handlers': [],
                'propagate': False
            }
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file']
        }
    }
    
    try:
        # Resolve logging config path: prefer explicit path; else use project config folder
        resolved_config_path = None
        try:
            # 1) As provided (absolute or relative to CWD)
            if config_path and os.path.exists(config_path):
                resolved_config_path = config_path
            else:
                # 2) Packaged default under the installed package base
                candidate_pkg_config = os.path.join(get_package_base_dir(), 'config', 'logging.yaml')
                if os.path.exists(candidate_pkg_config):
                    resolved_config_path = candidate_pkg_config
                else:
                    # 3) Relative to runtime base (developer repo root)
                    candidate_runtime = os.path.join(get_app_base_dir(), 'config', 'logging.yaml')
                    if os.path.exists(candidate_runtime):
                        resolved_config_path = candidate_runtime
        except Exception:
            resolved_config_path = None

        # Load configuration from resolved file, else use defaults
        if resolved_config_path and os.path.exists(resolved_config_path):
            with open(resolved_config_path, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)
                logging_config = config_data.get('logging', default_config)
        else:
            if config_path:
                print(f"Warning: Logging config file not found at {config_path}, using defaults (package dir)")
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
                # Force UTF-8 console handler to avoid cp1252 Unicode errors
                console_handler['class'] = f"{__name__}.UTF8StreamHandler"
                console_handler.pop('stream', None)
            # Ensure file handlers are DEBUG by default and use UTF-8 encoding
            for hname in ('file', 'error_file'):
                handler = logging_config.get('handlers', {}).get(hname)
                if handler:
                    if hname == 'file':
                        handler['level'] = 'DEBUG'
                    elif hname == 'error_file':
                        handler['level'] = 'ERROR'
                    handler['encoding'] = 'utf-8'
            # Root level should be DEBUG to capture all into file handler
            if 'root' in logging_config:
                logging_config['root']['level'] = 'DEBUG'
        except Exception:
            pass

        # Ensure handler filenames use absolute logs_dir paths
        try:
            if 'handlers' in logging_config:
                if 'file' in logging_config['handlers']:
                    logging_config['handlers']['file']['filename'] = os.path.join(logs_dir, 'video_compressor.log')
                if 'error_file' in logging_config['handlers']:
                    logging_config['handlers']['error_file']['filename'] = os.path.join(logs_dir, 'errors.log')
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
        
        # Add PIL debug filter to file handler to prevent verbose PIL logs
        try:
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if isinstance(handler, logging.FileHandler) and 'video_compressor.log' in handler.baseFilename:
                    pil_filter = PILDebugFilter()
                    handler.addFilter(pil_filter)
        except Exception as filter_error:
            logger.debug(f"Could not add PIL filter to file handler: {filter_error}")
        
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