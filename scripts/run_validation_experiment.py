"""
Validation experiment: compare three filtering strategies on a mixed dataset.

Reads pre-computed score JSON files (from score_dataset.py and
evaluate_semantic_baseline.py) and reports what each strategy would keep/remove.
The actual ACT policy training is run separately via LeRobot's training pipeline.

Three filtering strategies:
  1. HF tool only:  aggregate_score >= technical_threshold
  2. Semantic only: semantic_score >= semantic_threshold
  3. Combined:      both thresholds pass

Usage:
    python scripts/run_validation_experiment.py \\
        --scores_dir results/ \\
        --technical_threshold 0.5 \\
        --semantic_threshold 0.5 \\
        --output_path results/validation_summary.json

Input JSON format expected from score_dataset.py:
    [{"episode_id": 0, "aggregate_score": 0.82, "camera_type": "top", ...}, ...]

Input JSON format expected from evaluate_semantic_baseline.py:
    [{"episode_idx": 0, "semantic_score": 0.91, "condition": "clean", ...}, ...]

The script matches episodes by episode_id/episode_idx across files.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def load_technical_scores(path: str) -> Dict[int, float]:
    """
    Load aggregate scores from score_dataset.py output.
    Returns {episode_id: aggregate_score} using max aggregate across cameras.
    """
    with open(path) as f:
        data = json.load(f)

    ep_scores: Dict[int, float] = {}
    for entry in data:
        ep_id = entry["episode_id"]
        agg = entry["aggregate_score"]
        # Use max across cameras for robustness
        ep_scores[ep_id] = max(ep_scores.get(ep_id, 0.0), agg)
    return ep_scores


def load_semantic_scores(path: str) -> Dict[int, float]:
    """
    Load semantic scores from evaluate_semantic_baseline.py output.
    Returns {episode_idx: semantic_score}.
    """
    with open(path) as f:
        data = json.load(f)

    return {
        entry["episode_idx"]: entry["semantic_score"]
        for entry in data
        if entry.get("semantic_score") is not None
    }


def apply_filter(
    technical: Dict[int, float],
    semantic: Dict[int, float],
    tech_thresh: float,
    sem_thresh: float,
    strategy: str,
) -> Dict:
    """
    Apply a filtering strategy. Returns a summary dict.
    strategy: "technical_only" | "semantic_only" | "combined"
    """
    all_eps = sorted(set(technical.keys()) | set(semantic.keys()))

    kept = []
    removed_technical = []
    removed_semantic = []
    removed_both = []

    for ep in all_eps:
        tech_score = technical.get(ep)
        sem_score = semantic.get(ep)

        passes_tech = (tech_score is not None and tech_score >= tech_thresh)
        passes_sem = (sem_score is not None and sem_score >= sem_thresh)
        missing_tech = tech_score is None
        missing_sem = sem_score is None

        if strategy == "technical_only":
            if missing_tech:
                continue  # skip episodes with no technical score
            if passes_tech:
                kept.append(ep)
            else:
                removed_technical.append(ep)

        elif strategy == "semantic_only":
            if missing_sem:
                continue
            if passes_sem:
                kept.append(ep)
            else:
                removed_semantic.append(ep)

        elif strategy == "combined":
            has_both = not missing_tech and not missing_sem
            if not has_both:
                continue
            fail_tech = not passes_tech
            fail_sem = not passes_sem
            if fail_tech and fail_sem:
                removed_both.append(ep)
            elif fail_tech:
                removed_technical.append(ep)
            elif fail_sem:
                removed_semantic.append(ep)
            else:
                kept.append(ep)

    n_input = len(all_eps)
    n_kept = len(kept)
    n_removed = n_input - n_kept

    return {
        "strategy": strategy,
        "technical_threshold": tech_thresh,
        "semantic_threshold": sem_thresh,
        "input_episodes": n_input,
        "kept_episodes": n_kept,
        "removed_episodes": n_removed,
        "kept_fraction": round(n_kept / n_input, 4) if n_input > 0 else 0.0,
        "removed_by_technical_only": len(removed_technical),
        "removed_by_semantic_only": len(removed_semantic),
        "removed_by_both": len(removed_both),
        "kept_episode_ids": kept,
    }


def per_condition_recall(
    semantic_scores_by_condition: Dict[str, Dict[int, float]],
    sem_thresh: float,
) -> Dict[str, float]:
    """
    For each condition (ground truth label known), compute recall:
    what fraction of true failures (label=0) are correctly flagged?
    And what fraction of true successes (label=1) are correctly kept?
    """
    # Conditions that are failures (y=0): semantic score should be LOW → filtered out
    FAILURE_CONDITIONS = {"wrong_cube", "task_fail", "extra_objects",
                          "bad_lighting", "shakiness", "occluded_top_cam"}
    # Conditions that are successes (y=1): semantic score should be HIGH → kept
    SUCCESS_CONDITIONS = {"clean"}

    result = {}
    for condition, scores in semantic_scores_by_condition.items():
        if not scores:
            continue
        values = list(scores.values())
        if condition in FAILURE_CONDITIONS:
            # Recall = fraction correctly identified as failures (score < threshold)
            flagged = sum(1 for v in values if v < sem_thresh)
            result[condition] = round(flagged / len(values), 4)
        elif condition in SUCCESS_CONDITIONS:
            # Precision = fraction correctly kept (score >= threshold)
            kept = sum(1 for v in values if v >= sem_thresh)
            result[condition] = round(kept / len(values), 4)

    return result


def main():
    ap = argparse.ArgumentParser(description="Compare filtering strategies for dataset cleaning.")
    ap.add_argument("--technical_scores_path", default=None,
                    help="Path to score_dataset.py JSON output. "
                         "If omitted, semantic-only strategy only.")
    ap.add_argument("--semantic_scores_paths", nargs="*", default=None,
                    help="Paths to evaluate_semantic_baseline.py JSON outputs, "
                         "one per condition. If omitted, technical-only strategy only.")
    ap.add_argument("--technical_threshold", type=float, default=0.5,
                    help="Aggregate score threshold for the HF tool filter.")
    ap.add_argument("--semantic_threshold", type=float, default=0.5,
                    help="Semantic score threshold for the I-FailSense filter.")
    ap.add_argument("--output_path", required=True,
                    help="Path to save the validation summary JSON.")
    args = ap.parse_args()

    print(f"\nValidation Experiment")
    print(f"  technical_threshold: {args.technical_threshold}")
    print(f"  semantic_threshold:  {args.semantic_threshold}\n")

    # Load scores
    technical: Dict[int, float] = {}
    if args.technical_scores_path:
        technical = load_technical_scores(args.technical_scores_path)
        print(f"Loaded technical scores: {len(technical)} episodes "
              f"from {args.technical_scores_path}")

    semantic_combined: Dict[int, float] = {}
    semantic_by_condition: Dict[str, Dict[int, float]] = {}

    if args.semantic_scores_paths:
        for path in args.semantic_scores_paths:
            sem = load_semantic_scores(path)
            # Detect condition from content
            with open(path) as f:
                first = json.load(f)
            condition = first[0].get("condition", Path(path).stem) if first else Path(path).stem
            semantic_by_condition[condition] = sem
            semantic_combined.update(sem)
            print(f"Loaded semantic scores: {len(sem)} episodes from {path} ({condition})")

    print()

    # Run strategies
    strategies = []
    t = args.technical_threshold
    s = args.semantic_threshold

    if technical:
        result_tech = apply_filter(technical, semantic_combined, t, s, "technical_only")
        strategies.append(result_tech)
        print(f"[technical_only]  kept={result_tech['kept_episodes']}/"
              f"{result_tech['input_episodes']}  "
              f"({result_tech['kept_fraction']*100:.1f}%)")

    if semantic_combined:
        result_sem = apply_filter(technical, semantic_combined, t, s, "semantic_only")
        strategies.append(result_sem)
        print(f"[semantic_only]   kept={result_sem['kept_episodes']}/"
              f"{result_sem['input_episodes']}  "
              f"({result_sem['kept_fraction']*100:.1f}%)")

    if technical and semantic_combined:
        result_comb = apply_filter(technical, semantic_combined, t, s, "combined")
        strategies.append(result_comb)
        print(f"[combined]        kept={result_comb['kept_episodes']}/"
              f"{result_comb['input_episodes']}  "
              f"({result_comb['kept_fraction']*100:.1f}%)  "
              f"(removed: tech={result_comb['removed_by_technical_only']} "
              f"sem={result_comb['removed_by_semantic_only']} "
              f"both={result_comb['removed_by_both']})")

    # Per-condition recall
    per_cond = {}
    if semantic_by_condition:
        per_cond = per_condition_recall(semantic_by_condition, s)
        print("\nPer-condition semantic detection rate:")
        for cond, rate in sorted(per_cond.items()):
            label_type = "kept (success)" if cond == "clean" else "flagged (failure)"
            print(f"  {cond:30s}: {rate:.4f}  [{label_type}]")

    # Save output
    output = {
        "parameters": {
            "technical_threshold": t,
            "semantic_threshold": s,
        },
        "strategies": strategies,
        "per_condition_detection_rate": per_cond,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nValidation summary saved to: {args.output_path}")


if __name__ == "__main__":
    main()
