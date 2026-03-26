from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "reference_data"
DEST_DIR = ROOT / "web" / "public" / "reference_data"

# These files are not used by the web app and are much larger than the rest.
EXCLUDED_FILES = {
    ".DS_Store",
    "betfair.csv",
    "closing_odds.csv",
}


def should_skip(relative_path: Path) -> bool:
    return any(part in EXCLUDED_FILES for part in relative_path.parts)


def iter_source_files() -> list[Path]:
    return sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.is_file() and not should_skip(path.relative_to(SOURCE_DIR))
    )


def iter_dest_files() -> list[Path]:
    if not DEST_DIR.exists():
        return []
    return sorted(
        path
        for path in DEST_DIR.rglob("*")
        if path.is_file() and not should_skip(path.relative_to(DEST_DIR))
    )


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    source_files = iter_source_files()
    source_relpaths = {path.relative_to(SOURCE_DIR) for path in source_files}

    copied = 0
    for source_path in source_files:
        relative_path = source_path.relative_to(SOURCE_DIR)
        dest_path = DEST_DIR / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)
        copied += 1

    removed = 0
    for dest_path in iter_dest_files():
        relative_path = dest_path.relative_to(DEST_DIR)
        if relative_path not in source_relpaths:
            dest_path.unlink()
            removed += 1

    # Clean up empty directories left behind after file removal.
    for directory in sorted(
        (path for path in DEST_DIR.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass

    print(
        f"Synced {copied} files from {SOURCE_DIR.relative_to(ROOT)} "
        f"to {DEST_DIR.relative_to(ROOT)}; removed {removed} stale files."
    )


if __name__ == "__main__":
    main()
