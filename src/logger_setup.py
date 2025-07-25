"""
Logging Setup for Video Compressor
Initializes logging configuration from YAML file
"""

import os
import logging
import logging.config
import yaml
from colorama import init, Fore, Style
from typing import Optional

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
                'level': 'INFO',
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
        
        # Override log level if specified
        if log_level:
            log_level = log_level.upper()
            if 'root' in logging_config:
                logging_config['root']['level'] = log_level
            if 'handlers' in logging_config and 'console' in logging_config['handlers']:
                logging_config['handlers']['console']['level'] = log_level
        
        # Apply logging configuration
        logging.config.dictConfig(logging_config)
        
        # Get the main logger and add colored formatter to console handler
        logger = logging.getLogger('video_compressor')
        
        # Find console handler and apply colored formatter
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream.name == '<stdout>':
                colored_formatter = ColoredFormatter(
                    fmt='%(asctime)s | %(levelname)-8s | %(message)s',
                    datefmt='%H:%M:%S'
                )
                handler.setFormatter(colored_formatter)
        
        logger.info("Logging system initialized successfully")
        
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