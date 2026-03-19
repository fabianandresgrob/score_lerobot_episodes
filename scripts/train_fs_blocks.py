"""
FS block fine-tuning script.

Fine-tunes only the FS (FailSense) block classifiers of I-FailSense on our
LeRobot pick-and-place data. The VLM backbone is kept fully frozen throughout.

Training data:
  Positives (y=1, success): clean dataset
  Negatives (y=0, failure): wrong_cube + task_fail + extra_objects
  DO NOT use bad_lighting, shakiness, or occluded_top_cam as negatives —
  those are technical failures, not semantic ones.

Environment:
    Run from the score_lerobot_episodes/.venv environment.
    i-failsense must be installed (editable) in that venv:
        uv pip install -e ../I-FailSense --no-deps

Hardware:
    Designed for RTX 3080 (10 GB VRAM). VLM loads in float32 then casts to
    bfloat16. If OOM during load, ensure no other tensors are on the GPU.
    Default batch_size=1, gradient_accumulation_steps=4 (effective batch=4).

Example:
    python scripts/train_fs_blocks.py \\
        --positive_repo_id j-m-h/pick_place_clean_realsense_downscaled \\
        --negative_repo_ids \\
            fabiangrob/pick_place_wrong_cube_realsense_downscaled \\
            fabiangrob/pick_place_task_fail_realsense_downscaled \\
            fabiangrob/pick_place_extra_objects_realsense_downscaled \\
        --task_description "pick up the orange cube and place it in the blue container" \\
        --output_dir checkpoints/fs_blocks_custom \\
        --dry_run

DO NOT run full training without confirming with Fabian first.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class EpisodeDataset(torch.utils.data.Dataset):
    """
    Wraps a LeRobot dataset as a PyTorch Dataset of (images_list, text, label).

    Each item returns:
        images:  list of 8 PIL images (adapter output)
        text:    process_input(images, task_description)
        label:   float 0.0 or 1.0
    """

    def __init__(
        self,
        lerobot_dataset,
        task_description: str,
        label: float,
        process_input_fn,
        top_camera_key: str = "observation.images.top",
        wrist_camera_key: str = "observation.images.wrist",
    ):
        self.dataset = lerobot_dataset
        self.task_description = task_description
        self.label = label
        self.process_input = process_input_fn
        self.top_camera_key = top_camera_key
        self.wrist_camera_key = wrist_camera_key
        self.n_episodes = lerobot_dataset.meta.total_episodes

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        from score_lerobot_episodes.semantic_adapter import episode_to_failsense_input
        images, task_desc = episode_to_failsense_input(
            self.dataset, idx, self.task_description,
            self.top_camera_key, self.wrist_camera_key
        )
        text = self.process_input(images, task_desc)
        return {"images": images, "text": text, "label": self.label}


def collate_fn(batch):
    """Collate a list of episode dicts into batched lists."""
    return {
        "images": [item["images"] for item in batch],
        "texts":  [item["text"]   for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
    }


def load_combined_dataset(
    positive_repo_id: str,
    negative_repo_ids: List[str],
    task_description: str,
    process_input_fn,
    root: Optional[str],
    top_camera_key: str,
    wrist_camera_key: str,
    val_fraction: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
    video_backend: str = "pyav",
):
    """Load positive and negative LeRobot datasets and return train/val splits."""
    from score_lerobot_episodes.data import load_dataset_hf

    all_items = []

    # Positives
    print(f"Loading positive dataset: {positive_repo_id}")
    pos_ds = load_dataset_hf(positive_repo_id, root=root, video_backend=video_backend)
    n_pos = pos_ds.meta.total_episodes
    if dry_run:
        n_pos = min(3, n_pos)
    for ep_idx in range(n_pos):
        all_items.append({"source": positive_repo_id, "ep_idx": ep_idx,
                          "label": 1.0, "dataset": pos_ds})
    print(f"  {n_pos} positive episodes")

    # Negatives
    for neg_repo in negative_repo_ids:
        print(f"Loading negative dataset: {neg_repo}")
        neg_ds = load_dataset_hf(neg_repo, root=root, video_backend=video_backend)
        n_neg = neg_ds.meta.total_episodes
        if dry_run:
            n_neg = min(3, n_neg)
        for ep_idx in range(n_neg):
            all_items.append({"source": neg_repo, "ep_idx": ep_idx,
                               "label": 0.0, "dataset": neg_ds})
        print(f"  {n_neg} negative episodes from {neg_repo}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_items)
    n_val = max(1, int(len(all_items) * val_fraction))
    val_items = all_items[:n_val]
    train_items = all_items[n_val:]

    print(f"\nDataset split: {len(train_items)} train, {len(val_items)} val")
    return train_items, val_items, task_description, process_input_fn, top_camera_key, wrist_camera_key


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, items, task_description, process_input_fn,
                    top_camera_key, wrist_camera_key, optimizer,
                    criterion, accumulation_steps, epoch, log_path):
    """Train for one epoch over a list of episode items."""
    from score_lerobot_episodes.semantic_adapter import episode_to_failsense_input

    model.train()
    model.vlm_model.eval()  # VLM always in eval mode

    total_loss = 0.0
    correct = 0
    total = 0
    optimizer.zero_grad()

    random.shuffle(items)

    for step_idx, item in enumerate(items):
        ep_idx = item["ep_idx"]
        dataset = item["dataset"]
        label = item["label"]

        try:
            images, task_desc = episode_to_failsense_input(
                dataset, ep_idx, task_description, top_camera_key, wrist_camera_key
            )
            text = process_input_fn(images, task_desc)
        except Exception as e:
            print(f"  [skip] Episode {ep_idx} from {item['source']}: {e}")
            continue

        images_batch = [images]
        text_batch = [text]
        label_tensor = torch.tensor([label], dtype=torch.float32, device=model.device)

        # Forward through FS blocks (VLM frozen)
        logits = model(images_batch, text_batch)  # list of [1,1] tensors

        losses = [criterion(logit.squeeze(-1), label_tensor) for logit in logits]
        loss = sum(losses) / len(losses) / accumulation_steps

        loss.backward()

        if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(items):
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            probs = [torch.sigmoid(l.squeeze(-1)) for l in logits]
            avg_prob = torch.stack(probs).mean(dim=0)
            pred = (avg_prob > 0.5).float()
            correct += (pred == label_tensor).sum().item()
            total += 1

        actual_loss = loss.item() * accumulation_steps
        total_loss += actual_loss

        # Clear hook features periodically
        if step_idx % 50 == 0:
            model.layer_features.clear()
            if model.device.type == "cuda":
                torch.cuda.empty_cache()

        if (step_idx + 1) % 10 == 0:
            print(f"    step {step_idx+1}/{len(items)}  loss={actual_loss:.4f}  "
                  f"acc={correct/total:.4f}", flush=True)

    avg_loss = total_loss / len(items) if items else 0.0
    train_acc = correct / total if total > 0 else 0.0
    return avg_loss, train_acc


def validate(model, items, task_description, process_input_fn,
             top_camera_key, wrist_camera_key):
    """Compute validation accuracy."""
    from score_lerobot_episodes.semantic_adapter import episode_to_failsense_input

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for item in items:
            ep_idx = item["ep_idx"]
            dataset = item["dataset"]
            label = item["label"]

            try:
                images, task_desc = episode_to_failsense_input(
                    dataset, ep_idx, task_description, top_camera_key, wrist_camera_key
                )
                text = process_input_fn(images, task_desc)
            except Exception as e:
                print(f"  [skip val] Episode {ep_idx}: {e}")
                continue

            images_batch = [images]
            text_batch = [text]
            label_tensor = torch.tensor([label], dtype=torch.float32, device=model.device)

            _, avg_probs = model.predict(images_batch, text_batch, voting=False)
            pred = (avg_probs > 0.5).float()
            correct += (pred == label_tensor).sum().item()
            total += 1

    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Fine-tune I-FailSense FS blocks on LeRobot data.")
    ap.add_argument("--positive_repo_id", required=True,
                    help="HF repo ID for the clean (positive) dataset.")
    ap.add_argument("--negative_repo_ids", nargs="+", required=True,
                    help="HF repo IDs for semantic failure (negative) datasets. "
                         "Use: wrong_cube, task_fail, extra_objects. "
                         "Do NOT include bad_lighting, shakiness, occluded_top_cam.")
    ap.add_argument("--task_description", required=True,
                    help="Task description string passed to the VLM.")
    ap.add_argument("--root", default=None,
                    help="Local dataset root. Omit to stream from HF Hub.")
    ap.add_argument("--vlm_model_id", default="ACIDE/FailSense-Calvin-2p-3b",
                    help="HF model ID for the I-FailSense VLM adapter.")
    ap.add_argument("--output_dir", required=True,
                    help="Directory to save FS block checkpoints.")
    ap.add_argument("--num_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=1,
                    help="Per-step batch size. Default 1 for 10 GB VRAM.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="Effective batch = batch_size * gradient_accumulation_steps.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--top_camera_key", default="observation.images.top")
    ap.add_argument("--wrist_camera_key", default="observation.images.wrist")
    ap.add_argument("--dry_run", action="store_true",
                    help="Run on 3 episodes per dataset and 2 epochs to verify setup.")
    ap.add_argument("--video_backend", default="pyav",
                    help="Video decoding backend: 'pyav' (default) or 'torchcodec'.")
    args = ap.parse_args()

    from i_failsense.model import FailSense, process_input

    print(f"\n{'='*60}")
    print("FS Block Fine-Tuning")
    print(f"  model:        {args.vlm_model_id}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  epochs:       {args.num_epochs if not args.dry_run else 2} (dry_run={args.dry_run})")
    print(f"  batch:        {args.batch_size} × {args.gradient_accumulation_steps} accum")
    print(f"  device:       {args.device}")
    print(f"  negatives:    {args.negative_repo_ids}")
    print(f"{'='*60}\n")

    num_epochs = 2 if args.dry_run else args.num_epochs

    # Load data
    train_items, val_items, task_desc, process_input_fn, top_key, wrist_key = \
        load_combined_dataset(
            args.positive_repo_id, args.negative_repo_ids,
            args.task_description, process_input,
            args.root, args.top_camera_key, args.wrist_camera_key,
            val_fraction=args.val_fraction, seed=args.seed,
            dry_run=args.dry_run,
            video_backend=args.video_backend,
        )

    # Load model
    print(f"\nLoading model: {args.vlm_model_id}")
    print("Note: loads in float32 then casts to bfloat16 — ensure GPU has headroom.")
    model = FailSense(args.vlm_model_id, device=args.device)
    model.vlm_model = model.vlm_model.to(torch.bfloat16)
    print("Model ready.\n")

    # Only FS block params are trained
    trainable_params = []
    for i in range(model.num_classifiers):
        trainable_params.extend(model.classifiers[i].parameters())
        trainable_params.extend(model.att_poolings[i].parameters())

    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} total "
          f"({100*n_trainable/n_total:.2f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save training config
    config = vars(args)
    config["n_train"] = len(train_items)
    config["n_val"] = len(val_items)
    config["n_trainable_params"] = n_trainable
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(args.output_dir, "train_log.json")
    log = []
    best_val_acc = 0.0

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_items, task_desc, process_input_fn,
            top_key, wrist_key, optimizer, criterion,
            args.gradient_accumulation_steps, epoch, log_path
        )

        val_acc = validate(model, val_items, task_desc, process_input_fn,
                           top_key, wrist_key)

        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_acc={val_acc:.4f}  best_val={best_val_acc:.4f}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4),
        }
        log.append(epoch_record)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        # Save epoch checkpoint
        model.save_classifier(path=args.output_dir, epoch=epoch + 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_classifier(path=args.output_dir, epoch="best")
            print(f"  *** New best val accuracy: {best_val_acc:.4f} ***")

        model.vlm_model.eval()  # Keep VLM in eval after each epoch check

    model.cleanup()
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("Next step: run evaluate_semantic_baseline.py with "
          f"--fs_weights_path {args.output_dir}/components_epoch_best.pt")


if __name__ == "__main__":
    main()
