import sys
from pathlib import Path

from src.default_config import INPUT_DIR, LOG_DIR, OUTPUT_DIR, TEMP_FILE_DIR
from src.main import main


# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Create directories first
for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

try:
    main()
except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()
