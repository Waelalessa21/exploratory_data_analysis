import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from exploratory_data_analysis import config

paths = config.make_paths(Path("."))
ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    print(paths)
