#!/usr/bin/env python3
"""Prepare the exact anti-invention T4 smoke data contract.

This script is intentionally a preflight. It validates rows, writes an
inspectable manifest, and prints the pass/fail rule for the next GPU run. It
does not download models or start training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root / "src"))

    from opendistillation.experiment_manifest import (  # noqa: PLC0415
        build_anti_invention_smoke_manifest,
        format_anti_invention_smoke_report,
    )

    manifest = build_anti_invention_smoke_manifest(
        args.notes,
        repo_root=repo_root,
        training_output_dir=args.training_output_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for line in format_anti_invention_smoke_report(manifest):
        print(line)
    print(f"Manifest written: {args.output}")
    marker = {
        "ready": bool(manifest["validation"]["ready"]),
        "manifest_path": str(args.output),
        "facts": manifest["data"]["fact_count"],
        "train_rows": manifest["data"]["train_row_count"],
        "eval_rows": manifest["data"]["eval_row_count"],
        "known_values_only_rows": manifest["readiness"]["known_values_only_train_row_count"],
        "required_trained_exact_hits": manifest["quality_rule"]["required_trained_exact_hits"],
        "maximum_invented_value_misses": manifest["quality_rule"]["maximum_invented_value_misses"],
    }
    print("OD_ANTI_INVENTION_SMOKE_MANIFEST " + json.dumps(marker, sort_keys=True))

    if args.print_json:
        print(json.dumps(manifest, indent=2, sort_keys=True))

    return 0 if marker["ready"] else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and write the preflight manifest for the next "
            "OpenDistillation anti-invention T4 smoke."
        )
    )
    default_repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_repo_root,
        help="OpenDistillation repository root. Defaults to the script's parent repo.",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=default_repo_root / "examples" / "sample-notes.md",
        help="TXT/MD notes file for the exact smoke contract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/opendistillation_anti_invention_smoke_manifest.json"),
        help="Where to write the manifest JSON. Defaults to /tmp, outside the repo.",
    )
    parser.add_argument(
        "--training-output-dir",
        type=Path,
        default=Path("outputs") / "notes-lora",
        help="Adapter output directory that the later Colab training run should use.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the full manifest JSON to stdout.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
