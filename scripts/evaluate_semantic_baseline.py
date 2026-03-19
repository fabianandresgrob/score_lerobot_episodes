"""
VLM-only baseline evaluation script for semantic scoring.

Runs the I-FailSense LoRA-adapted VLM backbone on a LeRobot dataset and reports
per-episode predictions. No FS blocks are used unless --fs_weights_path is given.

This script produces the "before fine-tuning" (VLM-only) column in the results
table. Run it again after FS block training with --fs_weights_path to get the
"after fine-tuning" (full model) column.

Environment:
    Run from the score_lerobot_episodes/.venv environment.
    i-failsense must be installed (editable) in that venv:
        uv pip install -e ../I-FailSense --no-deps
    peft and transformers are pulled in as dependencies automatically.

Example usage (VLM-only baseline):
    python scripts/evaluate_semantic_baseline.py \\
        --repo_id fabiangrob/pick_place_wrong_cube_realsense_downscaled \\
        --task_description "pick up the orange cube and place it in the blue container" \\
        --condition wrong_cube \\
        --ground_truth_label 0 \\
        --output_path results/baseline_wrong_cube.json \\
        --dry_run

Example usage (full model after FS block training):
    python scripts/evaluate_semantic_baseline.py \\
        --repo_id j-m-h/pick_place_clean_realsense_downscaled \\
        --task_description "pick up the orange cube and place it in the blue container" \\
        --condition clean \\
        --ground_truth_label 1 \\
        --fs_weights_path checkpoints/fs_blocks_custom/components_best.pt \\
        --output_path results/full_model_clean.json
"""

import argparse
import json
import os

import sys
from pathlib import Path

# Ensure score_lerobot_episodes src is importable when run as a script
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))


def compute_metrics(predictions: list, ground_truth_label: int) -> dict:
    """Compute accuracy, precision, recall, F1 for binary classification."""
    labels = [ground_truth_label] * len(predictions)
    tp = sum(p == 1 and l == 1 for p, l in zip(predictions, labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(predictions, labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(predictions, labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(predictions, labels))

    accuracy = (tp + tn) / len(labels) if labels else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Run semantic baseline evaluation on a LeRobot dataset."
    )
    ap.add_argument("--repo_id", required=True,
                    help="HuggingFace dataset repo ID, e.g. j-m-h/pick_place_clean_...")
    ap.add_argument("--root", default=None,
                    help="Local dataset root path (omit to stream from HF Hub).")
    ap.add_argument("--task_description", required=True,
                    help='Task description string, e.g. "pick up the orange cube..."')
    ap.add_argument("--vlm_model_id", default="ACIDE/FailSense-Calvin-2p-3b",
                    help="HuggingFace model ID for the I-FailSense VLM adapter.")
    ap.add_argument("--fs_weights_path", default=None,
                    help="Path to trained FS block .pt checkpoint. "
                         "Omit to run VLM-only baseline (no FS blocks).")
    ap.add_argument("--condition", required=True,
                    help='Label for this dataset condition, e.g. "wrong_cube".')
    ap.add_argument("--ground_truth_label", type=int, required=True, choices=[0, 1],
                    help="Ground truth label: 1=success (clean), 0=failure (contaminated).")
    ap.add_argument("--output_path", required=True,
                    help="Path to save per-episode results JSON.")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"],
                    help="Compute device.")
    ap.add_argument("--top_camera_key", default="observation.images.top",
                    help="Dataset feature key for the top camera.")
    ap.add_argument("--wrist_camera_key", default="observation.images.wrist",
                    help="Dataset feature key for the wrist camera.")
    ap.add_argument("--semantic_threshold", type=float, default=0.5,
                    help="Probability threshold for binary predicted_label.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Process only the first 5 episodes to verify correctness.")
    ap.add_argument("--video_backend", default="pyav",
                    help="Video decoding backend: 'pyav' (default) or 'torchcodec'.")
    args = ap.parse_args()

    mode = "vlm_only" if args.fs_weights_path is None else "full"
    print(f"\n{'='*60}")
    print(f"Semantic Baseline Evaluation")
    print(f"  repo_id:    {args.repo_id}")
    print(f"  condition:  {args.condition}")
    print(f"  mode:       {mode}")
    print(f"  dry_run:    {args.dry_run}")
    print(f"{'='*60}\n")

    # Load dataset
    from score_lerobot_episodes.data import load_dataset_hf
    print(f"Loading dataset: {args.repo_id}")
    dataset = load_dataset_hf(args.repo_id, root=args.root, video_backend=args.video_backend)
    n_episodes = dataset.meta.total_episodes
    print(f"  Total episodes: {n_episodes}")

    if args.dry_run:
        n_episodes = min(5, n_episodes)
        print(f"  DRY RUN: processing only {n_episodes} episodes\n")

    # Load SemanticScorer
    from score_lerobot_episodes.scores.semantic_score import SemanticScorer
    scorer = SemanticScorer(
        task_description=args.task_description,
        vlm_model_id=args.vlm_model_id,
        fs_weights_path=args.fs_weights_path,
        device=args.device,
        top_camera_key=args.top_camera_key,
        wrist_camera_key=args.wrist_camera_key,
    )

    # Prepare output
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    results = []

    # Load existing results if output file exists (resume after crash)
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            results = json.load(f)
        done_episodes = {r["episode_idx"] for r in results}
        print(f"Resuming: {len(done_episodes)} episodes already done.")
    else:
        done_episodes = set()

    # Score each episode
    for ep_idx in range(n_episodes):
        if ep_idx in done_episodes:
            continue

        print(f"  Episode {ep_idx}/{n_episodes - 1}...", end=" ", flush=True)
        try:
            semantic_score = scorer.score_episode(dataset, ep_idx)
        except Exception as e:
            print(f"ERROR: {e}")
            semantic_score = None

        predicted_label = (
            int(semantic_score >= args.semantic_threshold)
            if semantic_score is not None else None
        )

        entry = {
            "episode_idx": ep_idx,
            "semantic_score": semantic_score,
            "predicted_label": predicted_label,
            "ground_truth_label": args.ground_truth_label,
            "condition": args.condition,
            "mode": mode,
        }
        results.append(entry)

        status = "✓" if predicted_label == args.ground_truth_label else "✗"
        score_str = f"{semantic_score:.3f}" if semantic_score is not None else "err"
        print(f"score={score_str}  pred={predicted_label}  gt={args.ground_truth_label}  {status}")

        # Save incrementally after every episode
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)

    scorer.cleanup()

    # Compute metrics
    valid_results = [r for r in results if r["predicted_label"] is not None]
    predictions = [r["predicted_label"] for r in valid_results]
    metrics = compute_metrics(predictions, args.ground_truth_label)

    print(f"\n{'='*60}")
    print(f"Results for condition: {args.condition} (mode: {mode})")
    print(f"  Episodes scored: {len(valid_results)} / {len(results)}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"  Saved to: {args.output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
