import os
import sys
import yaml
import glob
import logging
import time
from pathlib import Path
from tqdm import tqdm
from blessed import Terminal


class TerminalGUI:
    """
    Terminal-based GUI for the video and GIF compression application.
    Provides functionality to select config files and display processing progress.
    Uses full-screen terminal and keyboard navigation.
    """

    def __init__(self):
        self.config_dir = Path("config")
        self.selected_config = None
        self.config_data = None
        self.logger = logging.getLogger(__name__)
        self.progress_bars = {}
        self.term = Terminal()
        self.default_config = "default.yaml"

    def clear_screen(self):
        """Clear the terminal screen."""
        print(self.term.clear)

    def print_centered(self, y, text):
        """Print text centered horizontally at position y."""
        x = max(0, (self.term.width // 2) - (len(text) // 2))
        print(self.term.move(y, x) + text)

    def print_at(self, y, x, text):
        """Print text at specific coordinates."""
        print(self.term.move(y, x) + text)

    def display_banner(self):
        """Display the application banner at the top of the screen."""
        banner_text = "VIDEO AND GIF COMPRESSION TOOL"
        banner_decoration = "=" * len(banner_text)

        self.print_centered(1, banner_decoration)
        self.print_centered(2, banner_text)
        self.print_centered(3, banner_decoration)

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

        # If configs exist and no config is selected yet, select the default
        if configs and not self.selected_config:
            if self.default_config in configs:
                self.selected_config = self.default_config
            else:
                self.selected_config = configs[0]

        return configs

    def main_menu(self):
        """Display the main menu and handle user input."""
        with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
            selected = 0
            options = ["Choose Configuration", "Start Processing", "Exit"]

            while True:
                self.clear_screen()
                self.display_banner()

                # Display menu options
                for i, option in enumerate(options):
                    y = 6 + i
                    if i == selected:
                        self.print_at(y, 5, f"> {option} <")
                    else:
                        self.print_at(y, 5, f"  {option}  ")

                # Display current config if available
                if self.selected_config:
                    self.print_at(self.term.height - 4, 2,
                                  f"Current config: {self.selected_config}")

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
                            config_path = self.config_dir / self.selected_config
                            try:
                                with open(config_path, 'r') as f:
                                    self.config_data = yaml.safe_load(f)
                                self.logger.info(
                                    f"Selected config file: {self.selected_config}")
                                return self.config_data
                            except Exception as e:
                                self.logger.error(
                                    f"Failed to load config file: {e}")
                        else:
                            self.config_menu()
                    elif selected == 2:  # Exit
                        sys.exit(0)
                elif key.lower() == 'q':
                    sys.exit(0)
                elif key.isdigit() and 1 <= int(key) <= len(options):
                    selected = int(key) - 1

    def config_menu(self):
        """Display the config selection menu and handle user input."""
        config_files = self.list_config_files()

        if not config_files:
            with self.term.fullscreen():
                self.clear_screen()
                self.display_banner()
                self.print_at(
                    6, 2, "No config files found in the config directory.")
                self.display_footer("Press any key to return to main menu")
                self.term.inkey()
                return None

        # Find index of the selected config
        selected = 0
        if self.selected_config in config_files:
            selected = config_files.index(self.selected_config)

        with self.term.fullscreen(), self.term.cbreak(), self.term.hidden_cursor():
            while True:
                self.clear_screen()
                self.display_banner()

                # Display current config
                self.print_at(
                    5, 2, f"Current config: {self.selected_config or 'None'}")

                # Display config files
                self.print_at(7, 2, "Available configurations:")
                for i, config in enumerate(config_files):
                    y = 9 + i
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
                    return
                elif key.lower() == 'q':
                    return
                elif key.isdigit() and 1 <= int(key) <= len(config_files):
                    selected = int(key) - 1
                    self.selected_config = config_files[selected]
                    return

    def select_config_file(self):
        """Display menus to select a config file."""
        return self.main_menu()

    def initialize_progress_display(self, video_files=None, gif_files=None):
        """Initialize progress display for processing files."""
        self.progress_bars = {}

        if video_files:
            self.progress_bars['videos'] = {
                'total': len(video_files),
                'completed': 0,
                'bar': tqdm(total=len(video_files), desc="Videos", unit="file")
            }

        if gif_files:
            self.progress_bars['gifs'] = {
                'total': len(gif_files),
                'completed': 0,
                'bar': tqdm(total=len(gif_files), desc="GIFs", unit="file")
            }

    def update_progress(self, file_type, increment=1, message=None):
        """Update the progress bar for a specific file type."""
        if file_type in self.progress_bars:
            self.progress_bars[file_type]['completed'] += increment
            self.progress_bars[file_type]['bar'].update(increment)

            if message:
                self.progress_bars[file_type]['bar'].set_description(message)

    def display_summary(self):
        """Display a summary of processed files."""
        with self.term.fullscreen():
            self.clear_screen()
            self.display_banner()

            y = 6
            self.print_at(y, 2, "PROCESSING SUMMARY")
            self.print_at(y+1, 2, "="*20)

            y += 3
            for file_type, data in self.progress_bars.items():
                self.print_at(
                    y, 2, f"{file_type.capitalize()}: {data['completed']}/{data['total']} files processed")
                data['bar'].close()
                y += 1

            self.display_footer("Press any key to continue...")
            self.term.inkey()

    def display_error(self, message):
        """Display an error message."""
        with self.term.fullscreen():
            self.clear_screen()
            self.display_banner()

            self.print_at(6, 2, f"ERROR: {message}")
            self.display_footer("Press any key to continue...")
            self.term.inkey()

    def display_info(self, message):
        """Display an informational message."""
        with self.term.fullscreen():
            self.clear_screen()
            self.display_banner()

            self.print_at(6, 2, f"INFO: {message}")
            self.display_footer("Press any key to continue...")
            self.term.inkey()


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
