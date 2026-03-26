"""
Recover the train/val split from a previous training run.

Reproduces the exact shuffle + split from train_fs_blocks.py using the same
seed and val_fraction, without loading any models or GPU. Outputs a
split_info.json that can be used with evaluate_semantic_baseline.py --split_file.

This is useful for retroactively evaluating only the val episodes from a run
that did not save split_info.json.

Example:
    python scripts/recover_split_info.py \
        --positive_repo_id j-m-h/pick_place_clean_realsense_downscaled \
        --negative_repo_ids \
            fabiangrob/pick_place_wrong_cube_realsense_downscaled \
            fabiangrob/pick_place_task_fail_realsense_downscaled \
            fabiangrob/pick_place_extra_objects_realsense_downscaled \
        --val_fraction 0.2 \
        --seed 42 \
        --output_path checkpoints/fs_blocks_custom/split_info.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))


def main():
    ap = argparse.ArgumentParser(
        description="Recover train/val split indices from a previous training run."
    )
    ap.add_argument("--positive_repo_id", required=True)
    ap.add_argument("--negative_repo_ids", nargs="+", required=True)
    ap.add_argument("--root", default=None,
                    help="Local dataset root (same as used during training).")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_path", required=True,
                    help="Where to write split_info.json.")
    ap.add_argument("--video_backend", default="pyav")
    args = ap.parse_args()

    from score_lerobot_episodes.data import load_dataset_hf

    all_items = []

    # Positives
    print(f"Loading positive dataset: {args.positive_repo_id}")
    pos_ds = load_dataset_hf(args.positive_repo_id, root=args.root,
                             video_backend=args.video_backend)
    n_pos = pos_ds.meta.total_episodes
    for ep_idx in range(n_pos):
        all_items.append({"source": args.positive_repo_id, "ep_idx": ep_idx,
                          "label": 1.0})
    print(f"  {n_pos} positive episodes")

    # Negatives
    for neg_repo in args.negative_repo_ids:
        print(f"Loading negative dataset: {neg_repo}")
        neg_ds = load_dataset_hf(neg_repo, root=args.root,
                                 video_backend=args.video_backend)
        n_neg = neg_ds.meta.total_episodes
        for ep_idx in range(n_neg):
            all_items.append({"source": neg_repo, "ep_idx": ep_idx,
                              "label": 0.0})
        print(f"  {n_neg} negative episodes from {neg_repo}")

    # Reproduce the exact same shuffle and split
    random.seed(args.seed)
    random.shuffle(all_items)

    n_val = max(1, int(len(all_items) * args.val_fraction))
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    split_info = {
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": 0.0,
        "train": train_items,
        "val": val_items,
        "test": [],
    }

    # Summary per source
    print(f"\nSplit: {len(train_items)} train, {len(val_items)} val")
    for split_name, items in [("train", train_items), ("val", val_items)]:
        by_source = {}
        for it in items:
            by_source.setdefault(it["source"], []).append(it["ep_idx"])
        for src, idxs in sorted(by_source.items()):
            pos = sum(1 for it in items if it["source"] == src and it["label"] == 1.0)
            neg = len(idxs) - pos
            print(f"  {split_name}: {src} -> {len(idxs)} episodes "
                  f"({pos} pos, {neg} neg)")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\nSaved to: {args.output_path}")


if __name__ == "__main__":
    main()
