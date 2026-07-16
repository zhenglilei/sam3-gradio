#!/usr/bin/env python3
"""Run periodic template matching from a JSON request file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from periodic_template_matching import smart_annotation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a seed polygon to repeated template matches in an image."
    )
    parser.add_argument("--request", required=True, type=Path, help="Input JSON request")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON list")
    args = parser.parse_args()

    with args.request.open("r", encoding="utf-8") as file:
        request = json.load(file)

    matches = smart_annotation(request)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(matches, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"matched {len(matches)} repeated instances -> {args.output}")


if __name__ == "__main__":
    main()
