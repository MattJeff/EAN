"""Remove intermediate output files to keep the repository lightweight.

Deletes CSV logs, GIF/MP4 animations, and cached weight snapshots under the
*weights/* and *bench_results/* directories.  Use with caution: only files
matching known extensions are removed, directory structure is preserved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

EXTS = {".csv", ".gif", ".mp4", ".json"}


def _clean_dir(root: Path, dry_run: bool = True) -> None:  # noqa: D401
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in EXTS:
            print(f"{'[DRY] ' if dry_run else ''}Removing {path.relative_to(root)}")
            if not dry_run:
                path.unlink(missing_ok=True)


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Clean generated output files")
    parser.add_argument("--dry", action="store_true", help="Perform a dry-run without deleting files")
    args = parser.parse_args()

    roots = [Path("weights"), Path("bench_results"), Path("analysis/reports")]
    for r in roots:
        if r.exists():
            _clean_dir(r, dry_run=args.dry)

    print("Cleanup complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
