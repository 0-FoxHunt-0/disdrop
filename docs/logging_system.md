# LoggingSystem

The `LoggingSystem` class provides a centralized, enhanced logging solution for the application. It's designed to offer consistent, configurable, and feature-rich logging across all modules.

## Features

-   **Singleton Pattern**: Ensures a single logging configuration instance is shared across the application
-   **Custom Log Levels**: Adds a "SUCCESS" level for highlighting successful operations
-   **Colored Console Output**: Makes logs easier to read with color-coded log levels
-   **Multiple Output Destinations**: Supports both console and file logging with different configurations
-   **DirectX Diagnostics**: Automatically captures dxdiag information on Windows
-   **Log File Management**: Clears processing log file at startup and provides file rotation capabilities
-   **Thread Safety**: All operations are protected with locks for thread-safe use
-   **System Information Logging**: Can log detailed system information for troubleshooting
-   **Sectioned Logs**: Supports creating visual sections in logs for better organization

## Usage

### Basic Usage

```python
from src.logging_system import LoggingSystem

# Create logging system with config
config = {
    'logging': {
        'level': 'INFO',
        'console_level': 'INFO',
        'file_level': 'DEBUG',
        'file': './logs/processing.log',
        'clear_logs': True  # Clear log file at startup
    }
}

# Initialize logging system (singleton pattern ensures this is done only once)
logging_system = LoggingSystem(config)

# Get a logger for a specific module
logger = logging_system.get_logger('my_module')

# Use standard logging methods
logger.debug("Debug message")
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical error message")

# Use custom SUCCESS level
logger.success("Operation completed successfully!")

# Create a new section in the log
logging_system.start_new_log_section("New Processing Phase")

# Log system information
logging_system.log_system_info()
```

### Integration with Other Modules

```python
class MyProcessor:
    def __init__(self, config):
        self.config = config

        # Get the logging system using the singleton pattern
        self.logging_system = LoggingSystem(self.config)
        self.logger = self.logging_system.get_logger('my_processor')

        self.logger.info("Processor initialized")

    def process(self):
        self.logger.info("Starting processing")

        # Log success when operation completes
        self.logger.success("Processing completed successfully")
```

## Configuration Options

The `LoggingSystem` accepts a configuration dictionary with the following options:

| Option                   | Description                            | Default                                              |
| ------------------------ | -------------------------------------- | ---------------------------------------------------- |
| `logging.level`          | Main log level for all loggers         | INFO                                                 |
| `logging.directory`      | Directory to store log files           | ./logs                                               |
| `logging.file`           | Main log file path                     | ./logs/processing.log                                |
| `logging.console_level`  | Log level for console output           | INFO                                                 |
| `logging.file_level`     | Log level for file output              | DEBUG                                                |
| `logging.console_format` | Format string for console log messages | %(asctime)s - %(levelname)s - %(name)s - %(message)s |
| `logging.file_format`    | Format string for file log messages    | %(asctime)s - %(levelname)s - %(name)s - %(message)s |
| `logging.clear_logs`     | Whether to clear log files at startup  | True                                                 |

## API Reference

### Constructor

```python
LoggingSystem(config: Dict = None)
```

Initializes the logging system with the given configuration.

### Methods

#### `get_logger(name: str) -> logging.Logger`

Gets a properly configured logger for the specified name.

#### `capture_dxdiag() -> bool`

Captures DirectX diagnostic information on Windows systems and saves it to the logs directory.

#### `clear_log_file(log_file: Union[str, Path]) -> None`

Clears a log file by truncating it.

#### `create_rotating_file_handler(name: str, log_file: Union[str, Path], level: Union[str, int] = 'INFO', max_bytes: int = 10485760, backup_count: int = 5) -> logging.Handler`

Creates a rotating file handler for logs that exceed a certain size.

#### `add_custom_handler(logger_name: str, handler: logging.Handler) -> None`

Adds a custom handler to a specified logger.

#### `start_new_log_section(section_name: str) -> None`

Adds a section divider to the log to help with readability.

#### `log_system_info() -> None`

Logs basic system information to help with troubleshooting.

## Best Practices

1. **Early Initialization**: Initialize the LoggingSystem early in the application lifecycle
2. **Consistent Naming**: Use consistent module names when getting loggers
3. **Appropriate Log Levels**: Use the right log level for each message
    - DEBUG: Detailed information for diagnosing problems
    - INFO: Confirmation that things are working as expected
    - SUCCESS: Successful completion of important operations
    - WARNING: An indication that something unexpected happened
    - ERROR: Due to a more serious problem, the software couldn't perform an operation
    - CRITICAL: A serious error, indicating that the program itself may be unable to continue running
4. **Log Sections**: Use `start_new_log_section()` to organize logs into logical groups
5. **Context Information**: Include relevant context in log messages to make them more useful

## Integration with the Application

In the main application, the LoggingSystem should be initialized after loading the configuration:

```python
def main():
    # Load configuration
    config = load_config()

    # Initialize logging system
    logging_system = LoggingSystem(config)
    logger = logging_system.get_logger('main')

    # Log application startup
    logger.info("Application starting")

    # Log system information
    logging_system.log_system_info()

    # Continue with application initialization...
```

## DirectX Diagnostics

On Windows systems, the LoggingSystem automatically captures DirectX diagnostic information during initialization, which is saved to `logs/dxdiag.txt`. This information is valuable for troubleshooting GPU-related issues and understanding the system's graphics capabilities.

## Custom Log Levels

The LoggingSystem adds a custom "SUCCESS" log level (value 25), which sits between INFO (20) and WARNING (30). This level is perfect for highlighting successful operations without raising them to warning level importance.
