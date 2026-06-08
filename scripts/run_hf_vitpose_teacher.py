import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.teacher.hf_vitpose_teacher import main


if __name__ == "__main__":
    main()
