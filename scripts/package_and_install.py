#!/usr/bin/env python3
"""Package the Anki add-on into a .ankiaddon bundle and optionally symlink into Anki's addons21 folder."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_DIR = REPO_ROOT / "addon"
DIST_DIR = REPO_ROOT / "dist"
MACOS_ADDONS_DIR = Path.home() / "Library" / "Application Support" / "Anki2" / "addons21"
ADDON_FOLDER_NAME = "anki_ai_vocab_importer"


def build_ankiaddon_package() -> Path:
    """Create dist/anki_ai_vocab_importer.ankiaddon (zip archive without top-level folder)."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIST_DIR / f"{ADDON_FOLDER_NAME}.ankiaddon"

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(ADDON_DIR.rglob("*")):
            if file_path.is_dir():
                continue
            if "__pycache__" in file_path.parts or file_path.name.startswith("."):
                continue
            arcname = file_path.relative_to(ADDON_DIR)
            zf.write(file_path, arcname)

    return out_path


def install_dev_symlink() -> Path | None:
    """Create or update a symlink in ~/Library/Application Support/Anki2/addons21/."""
    if not MACOS_ADDONS_DIR.exists():
        return None
    target = MACOS_ADDONS_DIR / ADDON_FOLDER_NAME
    if target.is_symlink():
        target.unlink()
    elif target.exists():
        return target
    target.symlink_to(ADDON_DIR)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and install AI Vocabulary Importer add-on.")
    parser.add_argument("--symlink", action="store_true", help="Symlink addon/ into macOS Anki addons21 directory.")
    args = parser.parse_args()

    pkg = build_ankiaddon_package()
    print(f"Built package: {pkg}")

    if args.symlink:
        link_path = install_dev_symlink()
        if link_path:
            print(f"Installed symlink in Anki addons21: {link_path} -> {ADDON_DIR}")
        else:
            print(f"Anki addons21 directory not found at {MACOS_ADDONS_DIR}")


if __name__ == "__main__":
    main()

