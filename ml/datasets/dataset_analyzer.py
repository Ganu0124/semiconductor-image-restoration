"""
dataset_analyzer.py

Inspects the ACTUAL dataset on disk under `paths.dataset_root` (configs/config.yaml)
and reports real, measured statistics: image counts per split, dimensions, channels,
format, and whether degraded/clean pairs are present. Nothing here is invented —
if a split or pairing is missing, the report says so explicitly rather than assuming.

Usage:
    python ml/datasets/dataset_analyzer.py
    python ml/datasets/dataset_analyzer.py --root data/synthetic --out data/dataset_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.utils.config import load_config, resolve_path  # noqa: E402


def _list_images(folder: Path, extensions: list[str]) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in [e.lower() for e in extensions]
    )


def analyze_split(split_dir: Path, extensions: list[str]) -> dict:
    """Analyze one split. Supports two layouts:
    1) <split>/clean + <split>/degraded  (paired, preferred)
    2) <split>/*.<ext>                    (flat, unpaired)
    """
    result = {
        "exists": split_dir.exists(),
        "layout": None,
        "num_images": 0,
        "num_clean": 0,
        "num_degraded": 0,
        "paired": False,
        "num_pairs": 0,
        "dimensions": [],
        "channels": Counter(),
        "formats": Counter(),
        "min_width": None, "max_width": None,
        "min_height": None, "max_height": None,
        "corrupt_files": [],
    }
    if not split_dir.exists():
        return result

    clean_dir = split_dir / "clean"
    degraded_dir = split_dir / "degraded"

    if clean_dir.exists() and degraded_dir.exists():
        result["layout"] = "paired(clean/degraded subfolders)"
        clean_files = _list_images(clean_dir, extensions)
        degraded_files = _list_images(degraded_dir, extensions)
        result["num_clean"] = len(clean_files)
        result["num_degraded"] = len(degraded_files)
        clean_names = {p.name for p in clean_files}
        degraded_names = {p.name for p in degraded_files}
        common = clean_names & degraded_names
        result["paired"] = len(common) > 0
        result["num_pairs"] = len(common)
        result["num_images"] = len(clean_files) + len(degraded_files)
        all_files = clean_files + degraded_files
    else:
        result["layout"] = "flat"
        all_files = _list_images(split_dir, extensions)
        result["num_images"] = len(all_files)

    widths, heights = [], []
    for f in all_files:
        try:
            with Image.open(f) as im:
                w, h = im.size
                mode = im.mode
                widths.append(w)
                heights.append(h)
                result["channels"][mode] += 1
                result["formats"][im.format] += 1
        except Exception as e:  # noqa: BLE001
            result["corrupt_files"].append({"file": str(f), "error": str(e)})

    if widths:
        result["min_width"], result["max_width"] = min(widths), max(widths)
        result["min_height"], result["max_height"] = min(heights), max(heights)
        result["dimensions"] = sorted(set(zip(widths, heights)))[:20]  # sample, capped

    result["channels"] = dict(result["channels"])
    result["formats"] = dict(result["formats"])
    return result


def analyze_dataset(dataset_root: Path, splits: list[str], extensions: list[str]) -> dict:
    report = {
        "dataset_root": str(dataset_root),
        "root_exists": dataset_root.exists(),
        "splits": {},
    }
    for split in splits:
        report["splits"][split] = analyze_split(dataset_root / split, extensions)

    total_images = sum(s["num_images"] for s in report["splits"].values())
    report["total_images"] = total_images
    report["is_paired_dataset"] = any(s["paired"] for s in report["splits"].values())
    return report


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=cfg["paths"]["dataset_root"])
    parser.add_argument("--out", default="data/dataset_report.json")
    args = parser.parse_args()

    dataset_root = resolve_path(args.root)
    extensions = cfg["dataset"]["image_extensions"]
    splits = cfg["dataset"]["splits"]

    report = analyze_dataset(dataset_root, splits, extensions)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\n[dataset_analyzer] Report written to {out_path}")


if __name__ == "__main__":
    main()
