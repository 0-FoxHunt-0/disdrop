#!/usr/bin/env python3
"""
Example script demonstrating the use of ResourceManager for handling
interruption signals (CTRL+C) and cleaning up temporary resources.
"""

import os
import time
import logging
import tempfile
from pathlib import Path

# Import our ResourceManager
from src.resource_manager import ResourceManager


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('resource_manager_example')


def create_temp_files(temp_dir, count=5):
    """Create temporary files for demonstration."""
    file_paths = []
    for i in range(count):
        file_path = temp_dir / f"temp_file_{i}.txt"
        with open(file_path, 'w') as f:
            f.write(f"This is temporary file {i} content.\n")
        file_paths.append(file_path)
    return file_paths


def long_running_task(resource_manager, temp_dir):
    """Simulate a long-running task that can be interrupted."""
    logger = logging.getLogger('long_running_task')
    logger.info("Starting long-running task...")

    # Create some temporary files
    temp_files = create_temp_files(temp_dir)

    # Register temp files with resource manager
    for file_path in temp_files:
        resource_manager.register_temp_file(file_path)

    # Simulate long-running process
    try:
        for i in range(30):
            # Check if shutdown was requested
            if resource_manager.get_shutdown_flag():
                logger.info("Shutdown requested, stopping task")
                break

            logger.info(f"Processing step {i+1}/30...")
            time.sleep(1)  # Simulate work being done

        logger.info("Task completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in long-running task: {e}")
        return False


def cleanup_hook():
    """Example of a custom cleanup hook."""
    logger = logging.getLogger('cleanup_hook')
    logger.info("Custom cleanup hook executed")
    # In a real application, you might close database connections,
    # release file locks, or perform other cleanup operations here


def main():
    """Main function demonstrating ResourceManager usage."""
    logger = setup_logging()
    logger.info("Starting ResourceManager example")

    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    logger.info(f"Created temporary directory: {temp_dir}")

    # Initialize ResourceManager
    config = {
        'raise_after_cleanup': False  # Don't re-raise signal after cleanup
    }
    resource_manager = ResourceManager(config)

    # Register the temporary directory
    resource_manager.register_temp_dir(temp_dir)

    # Register a custom cleanup hook
    resource_manager.register_cleanup_hook(cleanup_hook)

    logger.info("ResourceManager initialized and configured")
    logger.info("Press CTRL+C at any time to test interruption handling")

    # Run the long-running task
    result = long_running_task(resource_manager, temp_dir)

    if resource_manager.get_shutdown_flag():
        logger.info("Application was interrupted with CTRL+C")
    else:
        logger.info(f"Task finished with result: {result}")

    # In normal operation, the atexit handler will clean up resources
    # But we can also manually trigger cleanup
    if not resource_manager.get_shutdown_flag():
        logger.info("Manually triggering resource cleanup")
        resource_manager.cleanup_resources()

    logger.info("Example completed")


if __name__ == "__main__":
    main()
