"""
Load and parse Stanford Drone Dataset annotations into DataFrames.

SDD annotation format (space-delimited, no header, 10 columns):
    track_id  xmin  ymin  xmax  ymax  frame  lost  occluded  generated  label

This module adds center-point columns (cx, cy) for trajectory extraction
and tags each row with its scene and video name.
"""

import pandas as pd
from pathlib import Path

COLUMN_NAMES = [
    "track_id", "xmin", "ymin", "xmax", "ymax",
    "frame", "lost", "occluded", "generated", "label",
]

COLUMN_DTYPES = {
    "track_id": int,
    "xmin": int, "ymin": int, "xmax": int, "ymax": int,
    "frame": int,
    "lost": int, "occluded": int, "generated": int,
    "label": str,
}


def load_scene_video(annotation_path: Path) -> pd.DataFrame:
    """Load a single annotations.txt file into a DataFrame.

    Adds columns: cx, cy (bounding box center-point).
    """
    df = pd.read_csv(
        annotation_path,
        sep=" ",
        header=None,
        names=COLUMN_NAMES,
        dtype=COLUMN_DTYPES,
        quotechar='"',
    )

    # Strip quotes from labels if present
    df["label"] = df["label"].str.strip('"')

    # Compute center-point of bounding box
    df["cx"] = (df["xmin"] + df["xmax"]) / 2.0
    df["cy"] = (df["ymin"] + df["ymax"]) / 2.0

    return df


def load_all_annotations(annotations_dir: Path) -> pd.DataFrame:
    """Load all SDD annotations into a single DataFrame.

    Adds columns: scene, video, cx, cy.
    Filters out rows where lost=1 (agent is out of view).
    """
    frames = []

    for annotation_file in sorted(annotations_dir.rglob("annotations.txt")):
        # Path structure: annotations_dir/scene/videoN/annotations.txt
        video_name = annotation_file.parent.name
        scene_name = annotation_file.parent.parent.name

        df = load_scene_video(annotation_file)
        df["scene"] = scene_name
        df["video"] = video_name
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No annotations.txt files found under {annotations_dir}"
        )

    combined = pd.concat(frames, ignore_index=True)

    # Filter out lost annotations (agent outside viewport)
    combined = combined[combined["lost"] == 0].reset_index(drop=True)

    return combined


def get_pedestrians_and_vehicles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split annotations into pedestrian and vehicle DataFrames."""
    pedestrians = df[df["label"] == "Pedestrian"].copy()
    vehicles = df[df["label"].isin(["Car", "Bus"])].copy()
    return pedestrians, vehicles


def summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the loaded dataset."""
    print(f"Total rows: {len(df):,}")
    print(f"Scenes: {df['scene'].nunique()}")
    print(f"Videos: {df.groupby(['scene', 'video']).ngroups}")
    print(f"Unique tracks: {df.groupby(['scene', 'video', 'track_id']).ngroups}")
    print(f"\nLabel distribution:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    annotations_dir = project_root / "data" / "raw" / "SDD" / "annotations"

    print(f"Loading annotations from {annotations_dir}...")
    df = load_all_annotations(annotations_dir)
    summary(df)

    peds, vehs = get_pedestrians_and_vehicles(df)
    print(f"\nPedestrians: {len(peds):,} rows, {peds.groupby(['scene', 'video', 'track_id']).ngroups} tracks")
    print(f"Vehicles:    {len(vehs):,} rows, {vehs.groupby(['scene', 'video', 'track_id']).ngroups} tracks")
