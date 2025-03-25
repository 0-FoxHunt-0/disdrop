import os
import sys
import yaml
import glob
import logging
import time
from pathlib import Path
from tqdm import tqdm
from blessed import Terminal
import atexit


class TerminalGUI:
    """
    Terminal-based GUI for the video and GIF compression application.
    Provides functionality to select config files and display processing progress.
    Uses full-screen terminal and keyboard navigation.
    """

    # ASCII art banner - fixed formatting for proper alignment
    BANNER = """
 ::::::::: ::::::::::: ::::::::  :::::::::  :::::::::   ::::::::  ::::::::: 
:+:    :+:    :+:    :+:    :+: :+:    :+: :+:    :+: :+:    :+: :+:    :+: 
+:+    +:+    +:+    +:+        +:+    +:+ +:+    +:+ +:+    +:+ +:+    +:+  
+#+    +:+    +#+    +#++:++#++ +#+    +:+ +#++:++#:  +#+    +:+ +#++:++#+    
+#+    +#+    +#+           +#+ +#+    +#+ +#+    +#+ +#+    +#+ +#+           
#+#    #+#    #+#    #+#    #+# #+#    #+# #+#    #+# #+#    #+# #+#            
#########  ###########  ########  #########   ###    ###   ########   ###             
"""

    def __init__(self):
        self.config_dir = Path("config")
        self.selected_config = None
        self.config_data = None
        self.logger = logging.getLogger(__name__)
        self.progress_bars = {}
        self.term = Terminal()
        self.default_config = "default.yaml"

        # Register cleanup function to ensure terminal is reset on exit
        atexit.register(self.cleanup_terminal)

    def _is_default_config_available(self):
        """Check if the default config file exists."""
        default_config_path = self.config_dir / self.default_config
        return default_config_path.exists()

    def _load_config_data(self):
        """Load the data from the selected config file."""
        if self.selected_config:
            config_path = self.config_dir / self.selected_config
            try:
                with open(config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f)
                self.logger.info(f"Loaded config file: {self.selected_config}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}")
        return False

    def cleanup_terminal(self):
        """Clean up terminal state on exit"""
        print(self.term.normal_cursor, end='')
        print(self.term.exit_fullscreen, end='')
        print(self.term.clear, end='')
        sys.stdout.flush()

    def clear_screen(self):
        """Clear the terminal screen."""
        print(self.term.clear, end='')
        sys.stdout.flush()

    def print_centered(self, y, text):
        """Print text centered horizontally at position y."""
        x = max(0, (self.term.width // 2) - (len(text) // 2))
        print(self.term.move(y, x) + text, end='')
        sys.stdout.flush()

    def print_at(self, y, x, text):
        """Print text at specific coordinates."""
        print(self.term.move(y, x) + text, end='')
        sys.stdout.flush()

    def display_banner(self):
        """Display the ASCII art banner at the top of the screen."""
        # Split the banner into lines and remove empty lines
        banner_lines = [
            line for line in self.BANNER.split('\n') if line.strip()]

        # Calculate the starting y position
        start_y = 1

        # Print each line of the banner centered horizontally
        for i, line in enumerate(banner_lines):
            self.print_centered(start_y + i, line)

    def display_footer(self, message):
        """Display controls and information at the bottom of the screen."""
        y = self.term.height - 2
        self.print_at(y, 2, message)

    def list_config_files(self):
        """List all YAML config files in the config directory."""
        if not self.config_dir.exists():
            self.logger.error(
                f"Config directory {self.config_dir} does not exist.")
            return []

        configs = sorted([f.name for f in self.config_dir.glob(
            "*.yaml") or self.config_dir.glob("*.yml")])

        return configs

    def main_menu(self):
        """Display the main menu and handle user input."""
        try:
            # Use the alternate screen buffer to prevent scrolling
            with self.term.fullscreen(), self.term.hidden_cursor(), self.term.cbreak():
                selected = 0
                options = ["Choose Configuration", "Start Processing", "Exit"]

                while True:
                    self.clear_screen()
                    self.display_banner()

                    # Calculate starting position below the banner
                    start_y = 10  # Adjusted to accommodate the ASCII banner

                    # Display menu options
                    for i, option in enumerate(options):
                        y = start_y + i
                        if i == selected:
                            self.print_at(y, 5, f"> {option} <")
                        else:
                            self.print_at(y, 5, f"  {option}  ")

                    # Display current config if available
                    if self.selected_config:
                        self.print_at(
                            start_y - 2, 5, f"Current config: {self.selected_config}")
                    else:
                        self.print_at(
                            start_y - 2, 5, "No configuration selected. Please choose one.")

                    # Display footer with controls
                    self.display_footer(
                        "Use ↑/↓ to navigate, ENTER to select, Q to quit")

                    # Get user input
                    key = self.term.inkey()

                    if key.code == self.term.KEY_UP:
                        selected = max(0, selected - 1)
                    elif key.code == self.term.KEY_DOWN:
                        selected = min(len(options) - 1, selected + 1)
                    elif key.code == self.term.KEY_ENTER:
                        if selected == 0:  # Choose Configuration
                            self.config_menu()
                        elif selected == 1:  # Start Processing
                            if self.selected_config:
                                # Load the config data if not already loaded
                                if not self.config_data:
                                    self._load_config_data()
                                return self.config_data
                            else:
                                # If no config selected, go to config selection menu
                                self.display_warning(
                                    "No configuration selected. Please choose one first.")
                                time.sleep(1.5)
                                self.config_menu()
                        elif selected == 2:  # Exit
                            return None
                    elif key.lower() == 'q':
                        return None
                    elif key.isdigit() and 1 <= int(key) <= len(options):
                        selected = int(key) - 1
        finally:
            # Ensure terminal is reset if an exception occurs
            self.cleanup_terminal()

    def config_menu(self):
        """Display the config selection menu and handle user input."""
        config_files = self.list_config_files()

        if not config_files:
            with self.term.fullscreen():
                self.clear_screen()
                self.display_banner()
                self.print_at(
                    10, 2, "No config files found in the config directory.")
                self.display_footer("Press any key to return to main menu")
                self.term.inkey()
                return None

        # Initialize selected index - don't automatically select
        selected = 0
        if self.selected_config in config_files:
            selected = config_files.index(self.selected_config)

        try:
            with self.term.fullscreen(), self.term.hidden_cursor(), self.term.cbreak():
                while True:
                    self.clear_screen()
                    self.display_banner()

                    # Adjust y positions to accommodate ASCII banner
                    start_y = 10

                    # Display current config
                    self.print_at(
                        start_y, 2, f"Current config: {self.selected_config or 'None'}")

                    # Display config files
                    self.print_at(start_y + 2, 2, "Available configurations:")
                    for i, config in enumerate(config_files):
                        y = start_y + 4 + i
                        if i == selected:
                            self.print_at(y, 4, f"→ {i+1}. {config}")
                        else:
                            self.print_at(y, 4, f"  {i+1}. {config}")

                    # Display footer with controls
                    self.display_footer(
                        "Use ↑/↓ to navigate, ENTER to select, Q to return to main menu")

                    # Get user input
                    key = self.term.inkey()

                    if key.code == self.term.KEY_UP:
                        selected = max(0, selected - 1)
                    elif key.code == self.term.KEY_DOWN:
                        selected = min(len(config_files) - 1, selected + 1)
                    elif key.code == self.term.KEY_ENTER:
                        self.selected_config = config_files[selected]
                        # Load the selected config data
                        self._load_config_data()
                        self.display_info(
                            f"Selected configuration: {self.selected_config}")
                        time.sleep(1)
                        return
                    elif key.lower() == 'q':
                        return
                    elif key.isdigit() and 1 <= int(key) <= len(config_files):
                        selected = int(key) - 1
                        self.selected_config = config_files[selected]
                        # Load the selected config data
                        self._load_config_data()
                        self.display_info(
                            f"Selected configuration: {self.selected_config}")
                        time.sleep(1)
                        return
        except Exception as e:
            self.logger.error(f"Error in config menu: {e}", exc_info=True)

    def select_config_file(self):
        """
        Public method to select a config file.
        This is the main entry point for configuration selection.

        Returns:
            Dict: The loaded configuration data
        """
        # Force the user to make a choice in the main menu
        return self.main_menu()

    def initialize_progress_display(self, video_files=None, gif_files=None):
        """Initialize progress bars for batch processing."""
        self.progress_bars = {}

        if video_files:
            self.progress_bars['videos'] = tqdm(
                total=len(video_files),
                desc="Processing Videos",
                unit="file"
            )

        if gif_files:
            self.progress_bars['gifs'] = tqdm(
                total=len(gif_files),
                desc="Processing GIFs",
                unit="file"
            )

    def update_progress(self, file_type, increment=1, message=None):
        """Update progress bar for a specific file type."""
        if file_type in self.progress_bars:
            self.progress_bars[file_type].update(increment)
            if message:
                self.progress_bars[file_type].set_description(message)

    def display_summary(self):
        """Display a summary of the processing results."""
        print("\n=== Processing Summary ===")

        for file_type, progress_bar in self.progress_bars.items():
            print(
                f"{file_type.capitalize()}: Processed {progress_bar.n}/{progress_bar.total} files")
            progress_bar.close()

        print("==========================\n")

    def display_warning(self, message):
        """Display a warning message in the terminal."""
        if hasattr(self, 'term') and self.term:
            try:
                current_pos = self.term.get_location()
                warning_y = self.term.height - 4

                # Create a bordered warning box
                box_width = len(message) + 4

                # Draw the box
                self.print_at(warning_y - 1, 2, "╔" + "═" * box_width + "╗")
                self.print_at(warning_y, 2, "║ " +
                              self.term.bold_yellow(message) + " ║")
                self.print_at(warning_y + 1, 2, "╚" + "═" * box_width + "╝")

                # Reset cursor position
                if current_pos:
                    self.print_at(current_pos[0], current_pos[1], "")
            except Exception as e:
                # Fallback to simple print
                print(f"\nWARNING: {message}\n")
        else:
            # Fallback to simple print
            print(f"\nWARNING: {message}\n")

    def display_error(self, message):
        """Display an error message to the user."""
        if hasattr(self, 'term') and self.term:
            try:
                print(self.term.bold_red(f"\nERROR: {message}\n"))
            except:
                # Fallback to simple print
                print(f"\nERROR: {message}\n")
        else:
            # Fallback to simple print
            print(f"\nERROR: {message}\n")

    def display_info(self, message):
        """Display an informational message to the user."""
        if hasattr(self, 'term') and self.term:
            try:
                print(self.term.bold_cyan(f"\nINFO: {message}\n"))
            except:
                # Fallback to simple print
                print(f"\nINFO: {message}\n")
        else:
            # Fallback to simple print
            print(f"\nINFO: {message}\n")


# Simple test if run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gui = TerminalGUI()
    config = gui.select_config_file()

    if config:
        print("\nSelected config:", config)

        # Simulate processing
        gui.initialize_progress_display(video_files=["video1.mp4", "video2.avi"],
                                        gif_files=["image1.gif", "image2.gif"])

        import time
        for _ in range(2):
            time.sleep(1)
            gui.update_progress('videos', message="Processing videos...")

        for _ in range(2):
            time.sleep(1)
            gui.update_progress('gifs', message="Optimizing GIFs...")

        gui.display_summary()
    else:
        print("No config selected. Exiting.")
