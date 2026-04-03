#!/usr/bin/env python3
"""Setup script for EngineAI PMv2 on Colab.

Run from Colab after mounting Google Drive:
  !python /content/drive/MyDrive/humanoid_colab/setup_colab.py

This script:
1. Extracts PMv2 asset, tracking config, velocity config from the archive
2. Copies them to the correct locations in the mjlab repo
3. Updates robots __init__.py
4. Verifies registration
"""

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

MJLAB_ROOT = Path("/content/mjlab")
DRIVE_DIR = Path("/content/drive/MyDrive/humanoid_colab")
ARCHIVE = DRIVE_DIR / "humanoid_colab.tar.gz"

ASSET_SRC = "/tmp/colab_extract/engineai_pmv2_asset"
TRACK_SRC = "/tmp/colab_extract/engineai_pmv2_tracking"
VELOC_SRC = "/tmp/colab_extract/engineai_pmv2_velocity"
ROBOTS_INIT_SRC = "/tmp/colab_extract/robots_init.py"

ASSET_DST = MJLAB_ROOT / "src/mjlab/asset_zoo/robots/engineai_pmv2"
TRACK_DST = MJLAB_ROOT / "src/mjlab/tasks/tracking/config/engineai_pmv2"
VELOC_DST = MJLAB_ROOT / "src/mjlab/tasks/velocity/config/engineai_pmv2"
ROBOTS_INIT_DST = MJLAB_ROOT / "src/mjlab/asset_zoo/robots/__init__.py"


def main():
    print("=" * 60)
    print("EngineAI PMv2 Setup for Google Colab")
    print("=" * 60)

    # 1. Extract archive
    print("\n[1/5] Extracting archive...")
    extract_dir = Path("/tmp/colab_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with tarfile.open(ARCHIVE, "r:gz") as tar:
        tar.extractall(extract_dir)
    print(f"  Extracted to {extract_dir}")
    for item in sorted(extract_dir.iterdir()):
        print(f"  - {item.name}")

    # 2. Copy asset
    print("\n[2/5] Copying PMv2 asset...")
    if ASSET_DST.exists():
        shutil.rmtree(ASSET_DST)
    shutil.copytree(ASSET_SRC, ASSET_DST)
    print(f"  → {ASSET_DST}")

    # 3. Copy tracking config
    print("\n[3/5] Copying tracking config...")
    if TRACK_DST.exists():
        shutil.rmtree(TRACK_DST)
    shutil.copytree(TRACK_SRC, TRACK_DST)
    print(f"  → {TRACK_DST}")

    # 4. Copy velocity config
    print("\n[4/5] Copying velocity config...")
    if VELOC_DST.exists():
        shutil.rmtree(VELOC_DST)
    shutil.copytree(VELOC_SRC, VELOC_DST)
    print(f"  → {VELOC_DST}")

    # 5. Update robots __init__.py
    print("\n[5/5] Updating robots __init__.py...")
    shutil.copy2(ROBOTS_INIT_SRC, ROBOTS_INIT_DST)
    print(f"  → {ROBOTS_INIT_DST}")

    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    for path, label in [
        (ASSET_DST / "pmv2_constants.py", "PMv2 constants"),
        (ASSET_DST / "xmls/pmv2.xml", "PMv2 MJCF"),
        (TRACK_DST / "__init__.py", "Tracking config"),
        (VELOC_DST / "__init__.py", "Velocity config"),
    ]:
        exists = "✅" if path.exists() else "❌"
        print(f"  {exists} {label}: {path}")

    print("\n✨ Setup complete! Run 'uv sync' then test training.")


if __name__ == "__main__":
    main()
