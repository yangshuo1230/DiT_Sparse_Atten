import sys
from pathlib import Path

# The repository is intentionally lightweight and is not installed as a wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
