"""
Visual verification of the LeRobot → I-FailSense adapter.

Saves a labeled PNG grid for each episode so you can confirm:
  - Camera keys are correct
  - Top camera is on the top row, wrist on the bottom row
  - Timesteps progress left to right
  - Frame content looks reasonable (exposure, crop, etc.)

Usage:
    python scripts/verify_adapter.py \\
        --repo_id j-m-h/pick_place_clean_realsense_downscaled \\
        --n_episodes 3 \\
        --output_dir verify_output/

Output per episode:
    verify_output/ep{N}_grid.png        — the full 2×4 grid as I-FailSense sees it
    verify_output/ep{N}_frames/         — all 8 frames saved individually for close inspection
"""

import argparse
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from PIL import Image, ImageDraw, ImageFont


def save_grid(images, episode_idx, output_dir, frame_size):
    """
    Save the 2×4 grid exactly as I-FailSense receives it, with labels.
    Row 0: top camera t0..t3
    Row 1: wrist camera t0..t3
    """
    W, H = frame_size
    LABEL_H = 24
    PAD = 4

    grid_w = W * 4 + PAD * 3
    grid_h = H * 2 + PAD + LABEL_H * 3  # row labels + column labels
    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    row_labels = ["TOP camera", "WRIST camera"]
    col_labels = [f"t{i}" for i in range(4)]

    # Column headers
    for col, label in enumerate(col_labels):
        x = col * (W + PAD) + W // 2 - len(label) * 3
        draw.text((x, 4), label, fill=(220, 220, 220))

    for i, img in enumerate(images):
        row, col = divmod(i, 4)
        x = col * (W + PAD)
        y = LABEL_H + row * (H + PAD)
        grid.paste(img.resize((W, H)), (x, y))

    # Row labels on the right
    for row, label in enumerate(row_labels):
        y = LABEL_H + row * (H + PAD) + H // 2
        draw.text((grid_w - len(label) * 6 - 2, y), label, fill=(255, 200, 0))

    # Episode label at the bottom
    ep_label = f"Episode {episode_idx}  —  2×4 grid as sent to I-FailSense"
    draw.text((4, grid_h - LABEL_H + 4), ep_label, fill=(180, 180, 180))

    path = os.path.join(output_dir, f"ep{episode_idx:03d}_grid.png")
    grid.save(path)
    return path


def save_individual_frames(images, episode_idx, output_dir):
    """Save all 8 frames individually for detailed inspection."""
    frame_dir = os.path.join(output_dir, f"ep{episode_idx:03d}_frames")
    os.makedirs(frame_dir, exist_ok=True)

    names = [
        "top_t0", "top_t1", "top_t2", "top_t3",
        "wrist_t0", "wrist_t1", "wrist_t2", "wrist_t3",
    ]
    for img, name in zip(images, names):
        img.save(os.path.join(frame_dir, f"{name}.png"))
    return frame_dir


def main():
    ap = argparse.ArgumentParser(description="Visually verify the adapter output.")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", default=None, help="Local dataset root (omit to stream from HF).")
    ap.add_argument("--n_episodes", type=int, default=3,
                    help="Number of episodes to visualise (default 3).")
    ap.add_argument("--episode_indices", type=int, nargs="*", default=None,
                    help="Specific episode indices to visualise. Overrides --n_episodes.")
    ap.add_argument("--task_description",
                    default="pick up the orange cube and place it in the blue container")
    ap.add_argument("--top_camera_key", default="observation.images.top")
    ap.add_argument("--wrist_camera_key", default="observation.images.wrist")
    ap.add_argument("--frame_size", type=int, nargs=2, default=[224, 224],
                    metavar=("W", "H"),
                    help="Frame size in the grid (default 224 224). "
                         "Increase e.g. to 320 240 for more detail.")
    ap.add_argument("--output_dir", default="verify_output")
    ap.add_argument("--save_individual", action="store_true",
                    help="Also save each of the 8 frames as individual PNGs.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from score_lerobot_episodes.data import load_dataset_hf
    from score_lerobot_episodes.semantic_adapter import (
        episode_to_failsense_input,
        _sample_4_indices,
        _get_episode_frame_bounds,
    )

    print(f"Loading dataset: {args.repo_id}")
    dataset = load_dataset_hf(args.repo_id, root=args.root)

    # Print all feature keys so camera keys are immediately visible
    print("\nDataset features:")
    for key in sorted(dataset.features.keys()):
        print(f"  {key}")

    # Validate camera keys
    for key in (args.top_camera_key, args.wrist_camera_key):
        if key not in dataset.features:
            print(f"\nERROR: camera key '{key}' not in dataset features.")
            print("Pass the correct keys via --top_camera_key / --wrist_camera_key")
            sys.exit(1)
    print(f"\nCamera keys OK:")
    print(f"  top:   {args.top_camera_key}")
    print(f"  wrist: {args.wrist_camera_key}")

    # Determine which episodes to process
    if args.episode_indices:
        episodes = args.episode_indices
    else:
        total = dataset.meta.total_episodes
        step = max(1, total // args.n_episodes)
        episodes = [i * step for i in range(args.n_episodes)]
        episodes = [e for e in episodes if e < total]

    print(f"\nProcessing episodes: {episodes}")
    frame_size = tuple(args.frame_size)

    for ep_idx in episodes:
        print(f"\n  Episode {ep_idx}:")

        # Show which global frame indices will be sampled
        from_idx, to_idx = _get_episode_frame_bounds(dataset, ep_idx)
        sample_indices = _sample_4_indices(from_idx, to_idx)
        print(f"    Frame range: [{from_idx}, {to_idx})  ({to_idx - from_idx} frames total)")
        print(f"    Sampled global indices: {sample_indices}")

        # Show raw tensor stats from the first frame for debugging
        sample = dataset[sample_indices[0]]
        top_tensor = sample[args.top_camera_key]
        print(f"    Tensor dtype: {top_tensor.dtype}  "
              f"shape: {tuple(top_tensor.shape)}  "
              f"range: [{top_tensor.min():.3f}, {top_tensor.max():.3f}]")

        # Build the adapter output
        images, _ = episode_to_failsense_input(
            dataset, ep_idx, args.task_description,
            args.top_camera_key, args.wrist_camera_key,
            target_size=frame_size,
        )

        # Save grid
        grid_path = save_grid(images, ep_idx, args.output_dir, frame_size)
        print(f"    Grid saved:  {grid_path}")

        # Save individual frames
        if args.save_individual:
            frame_dir = save_individual_frames(images, ep_idx, args.output_dir)
            print(f"    Frames saved: {frame_dir}/")

    print(f"\nDone. Open {args.output_dir}/ to inspect.")
    print("Check: top row = top camera, bottom row = wrist camera, left→right = time.")


if __name__ == "__main__":
    main()
