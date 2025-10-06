import sys
from pathlib import Path

# Ensure the project source directory is importable when tests run after an editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
