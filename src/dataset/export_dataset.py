"""Create a portable archive for Kaggle datasets."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def export_zip(input_dir: Path, output_path: Path, compression: int) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in input_dir.rglob("*") if path.is_file())
    with zipfile.ZipFile(output_path, mode="w", compression=compression) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(input_dir.parent))

    manifest = {
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "num_files": len(files),
        "size_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dataset directory to a Kaggle-friendly zip archive.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--store",
        action="store_true",
        help="Use no compression. Faster and often better for already-compressed JPG crops.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = zipfile.ZIP_STORED if args.store else zipfile.ZIP_DEFLATED
    export_zip(input_dir=args.input_dir, output_path=args.output_path, compression=compression)


if __name__ == "__main__":
    main()
