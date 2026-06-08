import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Student pose model.")
    parser.add_argument("--config", type=Path, default=Path("configs/training/student_baseline.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/student_baseline"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.training.train import train_student

    config_path = args.config
    if args.epochs is not None or args.batch_size is not None or args.num_workers is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        if args.batch_size is not None:
            config["training"]["batch_size"] = args.batch_size
        if args.num_workers is not None:
            config["training"]["num_workers"] = args.num_workers
        temp_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(config, temp_file)
        temp_file.close()
        config_path = Path(temp_file.name)

    train_student(config_path=config_path, data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
