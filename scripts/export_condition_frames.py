"""
Export episode frames for all 6 bad-quality conditions for presentation slides.

For each condition, extracts one full episode as individual PNG frames where
top camera and wrist camera are stacked vertically (portrait orientation).
Frames are saved as frame_0000.png, frame_0001.png, ... for use with the
LaTeX beamer animate package.

Usage:
    python scripts/export_condition_frames.py \\
        --root /path/to/local/datasets \\
        --output_dir slides/condition_frames/

    # Test on one condition first:
    python scripts/export_condition_frames.py \\
        --root /path/to/local/datasets \\
        --output_dir slides/condition_frames/ \\
        --dry_run

    # Pick a specific episode:
    python scripts/export_condition_frames.py \\
        --root /path/to/local/datasets \\
        --output_dir slides/condition_frames/ \\
        --episode_idx 2
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------

CONDITIONS = {
    "bad_lighting":     "fabiangrob/pick_place_bad_lighting_realsense_downscaled",
    "wrong_cube":       "fabiangrob/pick_place_wrong_cube_realsense_downscaled",
    "extra_objects":    "fabiangrob/pick_place_extra_objects_realsense_downscaled",
    "shakiness":        "fabiangrob/pick_place_shakiness_realsense_downscaled",
    "task_fail":        "fabiangrob/pick_place_task_fail_realsense_downscaled",
    "occluded_top_cam": "fabiangrob/pick_place_occluded_top_cam_realsense_downscaled",
}

TOP_CAMERA_KEY   = "observation.images.top"
WRIST_CAMERA_KEY = "observation.images.wrist"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_info(dataset_root: str) -> dict:
    info_path = os.path.join(dataset_root, "meta", "info.json")
    with open(info_path) as f:
        return json.load(f)


def detect_version(dataset_root: str) -> str:
    info = load_info(dataset_root)
    version = info.get("codebase_version", "v2.1")
    if not version.startswith("v"):
        version = f"v{version}"
    return version


def find_video_path_v21(dataset_root: str, episode_idx: int, camera_key: str) -> str:
    """Locate video file for v2.1 datasets (one episode per file)."""
    info = load_info(dataset_root)
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = episode_idx // chunks_size
    vid_path = os.path.join(
        dataset_root, "videos",
        f"chunk-{chunk_idx:03d}",
        camera_key,
        f"episode_{episode_idx:06d}.mp4",
    )
    return vid_path, None  # no timestamps needed for v2.1


def find_video_path_v30(dataset_root: str, episode_idx: int, camera_key: str):
    """Locate video file and timestamps for v3.0 datasets."""
    import glob as glob_mod
    import pyarrow.parquet as pq
    import pyarrow as pa

    episodes_dir = os.path.join(dataset_root, "meta", "episodes")
    parquet_files = sorted(glob_mod.glob(os.path.join(episodes_dir, "**/file-*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files in {episodes_dir}")

    tables = [pq.read_table(f) for f in parquet_files]
    df = pa.concat_tables(tables).to_pandas()
    row = df[df["episode_index"] == episode_idx].iloc[0]

    full_camera_key = camera_key if camera_key.startswith("observation.images.") else f"observation.images.{camera_key}"
    chunk_col   = f"videos/{full_camera_key}/chunk_index"
    file_col    = f"videos/{full_camera_key}/file_index"
    from_ts_col = f"videos/{full_camera_key}/from_timestamp"
    to_ts_col   = f"videos/{full_camera_key}/to_timestamp"

    chunk_idx   = int(row[chunk_col])
    file_idx    = int(row[file_col])
    from_ts     = float(row[from_ts_col])
    to_ts       = float(row[to_ts_col])

    vid_path = os.path.join(
        dataset_root, "videos",
        full_camera_key,
        f"chunk-{chunk_idx:03d}",
        f"file-{file_idx:03d}.mp4",
    )
    return vid_path, {"from_timestamp": from_ts, "to_timestamp": to_ts}


def resolve_dataset_root(root_arg: str, repo_id: str) -> str:
    """
    Find the actual dataset directory on disk.

    The local cache may store datasets as:
      <root>/<repo_id>  (e.g. .cache/huggingface/lerobot/fabiangrob/pick_place_...)
    or directly as <root> if the user passed the full path.
    """
    # Try <root>/<repo_id> first (HF cache layout)
    candidate = os.path.join(root_arg, repo_id)
    if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "meta", "info.json")):
        return candidate

    # Try just <root>
    if os.path.exists(os.path.join(root_arg, "meta", "info.json")):
        return root_arg

    # Try <root>/datasets/<repo_id>
    candidate2 = os.path.join(root_arg, "datasets", repo_id)
    if os.path.isdir(candidate2) and os.path.exists(os.path.join(candidate2, "meta", "info.json")):
        return candidate2

    raise FileNotFoundError(
        f"Cannot find dataset '{repo_id}' under root '{root_arg}'.\n"
        f"Tried:\n  {os.path.join(root_arg, repo_id)}\n  {root_arg}\n  {candidate2}"
    )


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def extract_segment_to_temp(video_path: str, from_ts: float, to_ts: float) -> str:
    """Extract [from_ts, to_ts] from video_path to a temp mp4 file. Returns temp path."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(tmp_fd)
    duration = to_ts - from_ts
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(from_ts),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",   # re-encode to ensure seekable frames
        "-an",               # no audio
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        os.unlink(tmp_path)
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")
    return tmp_path


def read_all_frames_cv2(video_path: str, output_fps: float | None = None) -> list[np.ndarray]:
    """
    Read frames from a video file as RGB numpy arrays (H, W, 3).

    If output_fps is given, uniformly subsample to approximately that frame rate.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    keep_every = max(1, round(source_fps / output_fps)) if output_fps else 1

    frames = []
    frame_i = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_i % keep_every == 0:
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        frame_i += 1
    cap.release()
    return frames


def stack_frames_vertically(top_frame: np.ndarray, wrist_frame: np.ndarray) -> np.ndarray:
    """Stack top camera (top) and wrist camera (bottom) into one portrait image."""
    # Ensure same width by resizing wrist to match top if needed
    h_top, w_top = top_frame.shape[:2]
    h_wrist, w_wrist = wrist_frame.shape[:2]
    if w_top != w_wrist:
        new_h = int(h_wrist * w_top / w_wrist)
        wrist_frame = cv2.resize(
            cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR),
            (w_top, new_h),
            interpolation=cv2.INTER_AREA,
        )
        wrist_frame = cv2.cvtColor(wrist_frame, cv2.COLOR_BGR2RGB)
    return np.vstack([top_frame, wrist_frame])


# ---------------------------------------------------------------------------
# Per-condition export
# ---------------------------------------------------------------------------

def export_condition(
    condition_name: str,
    repo_id: str,
    root: str,
    episode_idx: int,
    output_dir: str,
    output_fps: float = 10.0,
) -> int:
    """Export frames for one condition. Returns number of frames saved."""
    print(f"\n[{condition_name}] Resolving dataset root...")
    dataset_root = resolve_dataset_root(root, repo_id)
    version = detect_version(dataset_root)
    print(f"  Dataset root: {dataset_root}")
    print(f"  Version: {version}")

    # Find video paths and timestamps
    if version == "v2.1":
        top_path, top_info   = find_video_path_v21(dataset_root, episode_idx, TOP_CAMERA_KEY)
        wrist_path, wrist_info = find_video_path_v21(dataset_root, episode_idx, WRIST_CAMERA_KEY)
    else:
        top_path, top_info     = find_video_path_v30(dataset_root, episode_idx, TOP_CAMERA_KEY)
        wrist_path, wrist_info = find_video_path_v30(dataset_root, episode_idx, WRIST_CAMERA_KEY)

    for p in (top_path, wrist_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Video not found: {p}")

    print(f"  Top video:   {top_path}")
    print(f"  Wrist video: {wrist_path}")

    # Extract episode segments to temp files if needed
    top_tmp = wrist_tmp = None
    try:
        if top_info and "from_timestamp" in top_info:
            print(f"  Extracting top segment [{top_info['from_timestamp']:.3f}s – {top_info['to_timestamp']:.3f}s]...")
            top_tmp = extract_segment_to_temp(top_path, top_info["from_timestamp"], top_info["to_timestamp"])
            top_read_path = top_tmp
        else:
            top_read_path = top_path

        if wrist_info and "from_timestamp" in wrist_info:
            print(f"  Extracting wrist segment [{wrist_info['from_timestamp']:.3f}s – {wrist_info['to_timestamp']:.3f}s]...")
            wrist_tmp = extract_segment_to_temp(wrist_path, wrist_info["from_timestamp"], wrist_info["to_timestamp"])
            wrist_read_path = wrist_tmp
        else:
            wrist_read_path = wrist_path

        print("  Reading frames...")
        top_frames   = read_all_frames_cv2(top_read_path,   output_fps=output_fps)
        wrist_frames = read_all_frames_cv2(wrist_read_path, output_fps=output_fps)

    finally:
        for tmp in (top_tmp, wrist_tmp):
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    print(f"  Top frames: {len(top_frames)}, Wrist frames: {len(wrist_frames)}")

    # Align frame counts (take the minimum)
    n_frames = min(len(top_frames), len(wrist_frames))
    if len(top_frames) != len(wrist_frames):
        print(f"  WARNING: frame count mismatch; using first {n_frames} frames from each.")

    # Create output directory
    cond_dir = os.path.join(output_dir, condition_name)
    os.makedirs(cond_dir, exist_ok=True)

    # Stack and save
    print(f"  Saving {n_frames} stacked frames to {cond_dir}/...")
    for i in range(n_frames):
        stacked = stack_frames_vertically(top_frames[i], wrist_frames[i])
        out_path = os.path.join(cond_dir, f"frame_{i:04d}.png")
        img = Image.fromarray(stacked)
        img.save(out_path)
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"    {i + 1}/{n_frames} frames saved", end="\r")
    print()

    print(f"  Done: {n_frames} frames in {cond_dir}/")
    print(f"  Frame size: {stacked.shape[1]}×{stacked.shape[0]}px  (width × height)")
    return n_frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Export stacked (top+wrist) episode frames per condition for presentation slides."
    )
    ap.add_argument(
        "--root", required=True,
        help="Local dataset cache root (directory containing dataset folders).",
    )
    ap.add_argument(
        "--output_dir", default="slides/condition_frames",
        help="Output directory for frame folders (default: slides/condition_frames).",
    )
    ap.add_argument(
        "--episode_idx", type=int, default=0,
        help="Episode index to export from each dataset (default: 0).",
    )
    ap.add_argument(
        "--output_fps", type=float, default=10.0,
        help="Target frame rate for exported frames (default: 10). "
             "Source is typically 30fps, so default keeps every 3rd frame.",
    )
    ap.add_argument(
        "--dry_run", action="store_true",
        help="Process only the first condition to verify output before full run.",
    )
    args = ap.parse_args()

    conditions = list(CONDITIONS.items())
    if args.dry_run:
        conditions = conditions[:1]
        print(f"DRY RUN: processing only '{conditions[0][0]}'")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Episode index:    {args.episode_idx}")
    print(f"Output FPS:       {args.output_fps}")
    print(f"Conditions:       {[c for c, _ in conditions]}")

    results = {}
    for condition_name, repo_id in conditions:
        try:
            n = export_condition(
                condition_name, repo_id, args.root, args.episode_idx, args.output_dir,
                output_fps=args.output_fps,
            )
            results[condition_name] = {"status": "ok", "n_frames": n}
        except Exception as e:
            print(f"\n  ERROR for {condition_name}: {e}")
            results[condition_name] = {"status": "error", "error": str(e)}

    print("\n" + "=" * 60)
    print("Summary:")
    for cond, info in results.items():
        if info["status"] == "ok":
            print(f"  {cond:<20} {info['n_frames']} frames")
        else:
            print(f"  {cond:<20} ERROR: {info['error']}")

    if not args.dry_run:
        print(f"\nAll frames saved to: {args.output_dir}/")
        print(f"\nLaTeX animate usage (play at {args.output_fps:.0f}fps, e.g. bad_lighting):")
        print(rf"  \animategraphics[width=\linewidth]{{{args.output_fps:.0f}}}{{bad_lighting/frame_}}{{0000}}{{NNNN}}")


if __name__ == "__main__":
    main()
