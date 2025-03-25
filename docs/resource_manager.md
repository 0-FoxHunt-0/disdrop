# ResourceManager

The `ResourceManager` class provides a robust solution for handling interruption signals (like CTRL+C) and managing temporary resources in Python applications. It ensures that resources are properly cleaned up when a program exits, either normally or due to an interruption.

## Features

-   **Signal Handling**: Gracefully captures interruption signals (CTRL+C, SIGTERM, SIGBREAK on Windows)
-   **Resource Tracking**: Maintains a registry of temporary files and directories to clean up
-   **Custom Cleanup Hooks**: Allows registering functions to execute during cleanup
-   **Thread Safety**: All operations are thread-safe
-   **Graceful Shutdown**: Provides a flag to check if shutdown is in progress

## Usage

### Basic Usage

```python
from src.resource_manager import ResourceManager

# Initialize the ResourceManager
resource_manager = ResourceManager()

# Register temporary resources
resource_manager.register_temp_dir(Path('/path/to/temp/dir'))
resource_manager.register_temp_file(Path('/path/to/temp/file.txt'))

# Register a custom cleanup function
def my_cleanup_function():
    print("Cleaning up custom resources...")
    # Close connections, release locks, etc.

resource_manager.register_cleanup_hook(my_cleanup_function)
```

### Integration with Existing Code

```python
# Initialize your processor with a shutdown flag check
video_processor = VideoProcessor(config)
video_processor.shutdown_requested = False

# Register a hook to update the shutdown flag
def check_shutdown():
    video_processor.shutdown_requested = resource_manager.get_shutdown_flag()

resource_manager.register_cleanup_hook(check_shutdown)

# In your processing loop
for file in files:
    if video_processor.shutdown_requested:
        print("Shutdown requested, stopping processing")
        break

    # Process the file...
```

### Configuration Options

```python
# Initialize with custom configuration
config = {
    'raise_after_cleanup': False  # Don't re-raise the signal after cleanup
}
resource_manager = ResourceManager(config)
```

## API Reference

### Constructor

```python
ResourceManager(config: Dict = None)
```

-   **config**: Optional dictionary with configuration options
    -   `raise_after_cleanup`: If True (default), re-raise the signal after cleanup

### Methods

#### `register_temp_dir(dir_path: Path) -> None`

Register a temporary directory to be monitored and cleaned when the program exits.

#### `register_temp_file(file_path: Path) -> None`

Register a specific temporary file to be deleted when the program exits.

#### `register_cleanup_hook(hook: Callable) -> None`

Register a custom function to be called during cleanup.

#### `cleanup_resources() -> None`

Manually trigger the cleanup process. This is automatically called when the program exits or when an interruption signal is received.

#### `get_shutdown_flag() -> bool`

Check if a shutdown is in progress. Returns True if shutdown has been initiated.

## Example

See the complete example in `examples/resource_manager_example.py`.

## Best Practices

1. **Initialize Early**: Create the ResourceManager early in your program's lifecycle
2. **Register All Resources**: Register any temporary files and directories as soon as they're created
3. **Check the Flag**: In long-running operations, periodically check the shutdown flag
4. **Custom Hooks**: Use cleanup hooks to handle application-specific cleanup tasks
5. **Thread Safety**: The ResourceManager is thread-safe; you can use it from multiple threads

## Integration with Main Application

In the main application, the ResourceManager should be initialized after the configuration is loaded but before any processing begins. All processors that create temporary files or have long-running operations should be connected to the ResourceManager to ensure they can be gracefully interrupted.

```python
# In main.py
resource_manager = ResourceManager(config)

# Register temp directories
temp_dir = Path(config.get('directories', {}).get('temp', './temp'))
resource_manager.register_temp_dir(temp_dir)

# Connect processors
video_processor = VideoProcessor(config)
video_processor.shutdown_requested = False
def check_shutdown():
    video_processor.shutdown_requested = resource_manager.get_shutdown_flag()
resource_manager.register_cleanup_hook(check_shutdown)
```
