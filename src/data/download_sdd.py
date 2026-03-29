"""
Download Stanford Drone Dataset annotations only (no video).

Uses the annotations-only GitHub mirror (~few MB) instead of the
full 71 GB dataset from Stanford.

SDD annotation format (space-delimited, no header):
    track_id  xmin  ymin  xmax  ymax  frame  lost  occluded  generated  label

Labels: "Pedestrian", "Biker", "Skateboarder", "Cart", "Car", "Bus"
Scenes: bookstore, coupa, deathCircle, gates, hyang, little, nexus, quad
"""

import shutil
import subprocess
import sys
from pathlib import Path

ANNOTATIONS_REPO = "https://github.com/flclain/StanfordDroneDataset.git"

EXPECTED_SCENES = [
    "bookstore", "coupa", "deathCircle", "gates",
    "hyang", "little", "nexus", "quad",
]


def download_annotations(data_dir: Path) -> Path:
    """Clone SDD annotations into data_dir/raw/SDD/annotations/."""
    raw_dir = data_dir / "raw" / "SDD"

    if raw_dir.exists() and any(raw_dir.rglob("annotations.txt")):
        print(f"SDD annotations already present at {raw_dir}")
        return raw_dir

    print("Cloning SDD annotations (no video)...")
    clone_target = data_dir / "raw" / "_sdd_clone"

    subprocess.run(
        ["git", "clone", "--depth", "1", ANNOTATIONS_REPO, str(clone_target)],
        check=True,
    )

    # The repo structure has annotations/ at the top level with scene dirs inside
    cloned_annotations = clone_target / "annotations"
    if not cloned_annotations.exists():
        # Some mirrors put them directly at root level with scene-name dirs
        cloned_annotations = clone_target

    raw_dir.mkdir(parents=True, exist_ok=True)
    annotations_dest = raw_dir / "annotations"

    if annotations_dest.exists():
        shutil.rmtree(annotations_dest)

    shutil.copytree(cloned_annotations, annotations_dest)

    # Clean up the clone
    shutil.rmtree(clone_target, ignore_errors=True)

    _verify(annotations_dest)
    print(f"SDD annotations saved to {annotations_dest}")
    return raw_dir


def _verify(annotations_dir: Path) -> None:
    """Check that expected scenes and annotation files are present."""
    found_scenes = [
        d.name for d in annotations_dir.iterdir()
        if d.is_dir() and d.name in EXPECTED_SCENES
    ]
    missing = set(EXPECTED_SCENES) - set(found_scenes)
    if missing:
        print(f"WARNING: Missing scenes: {missing}", file=sys.stderr)

    annotation_files = list(annotations_dir.rglob("annotations.txt"))
    print(f"Found {len(annotation_files)} annotation files across {len(found_scenes)} scenes")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data"
    download_annotations(data_dir)
