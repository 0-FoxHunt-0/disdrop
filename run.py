from src.main import main
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


if __name__ == "__main__":
    main()
