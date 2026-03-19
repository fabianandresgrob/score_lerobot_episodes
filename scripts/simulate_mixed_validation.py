"""
Simulate mixed-dataset filtering experiments using pre-computed score JSONs.

Instead of re-running inference on mixed datasets, this script samples episodes
from existing per-condition score files, merges them with ID offsets, and runs
the three filtering strategies (technical only, semantic only, combined).

This is equivalent to scoring the actual mixed datasets since the model scores
each episode independently.

Usage:
    python scripts/simulate_mixed_validation.py \
        --results_dir results/ \
        --clean_episodes 100 \
        --levels 10 20 50 \
        --semantic_threshold 0.5 \
        --technical_threshold 0.5 \
        --output_path results/mixed_validation_summary.json \
        --seed 123
"""

import argparse
import json
import random
from pathlib import Path


# Conditions with ground truth labels
SEMANTIC_FAILURES = {"wrong_cube", "task_fail", "extra_objects"}
TECHNICAL_FAILURES = {"bad_lighting", "shakiness", "occluded_top_cam"}
ALL_BAD = SEMANTIC_FAILURES | TECHNICAL_FAILURES


def load_scores(results_dir: Path, condition: str):
    """
    Load both technical and semantic (full model) scores for a condition.
    Returns (tech_dict, sem_dict) where each is {episode_idx: score}.
    """
    if condition == "clean":
        tech_path = results_dir / "j-m-h_pick_place_clean_realsense_downscaled_scores.json"
    else:
        tech_path = results_dir / f"fabiangrob_pick_place_{condition}_realsense_downscaled_scores.json"

    sem_path = results_dir / f"baseline_{condition}_full.json"

    # Technical scores: take max aggregate across cameras per episode
    tech = {}
    with open(tech_path) as f:
        for entry in json.load(f):
            ep = entry["episode_id"]
            tech[ep] = max(tech.get(ep, 0.0), entry["aggregate_score"])

    # Semantic scores
    sem = {}
    with open(sem_path) as f:
        for entry in json.load(f):
            ep = entry["episode_idx"]
            if entry.get("semantic_score") is not None:
                sem[ep] = entry["semantic_score"]

    return tech, sem


def sample_episodes(scores_dict: dict, n: int, rng: random.Random) -> dict:
    """Sample n episode IDs from a scores dict."""
    keys = list(scores_dict.keys())
    if n > len(keys):
        raise ValueError(f"Requested {n} but only {len(keys)} available")
    chosen = rng.sample(keys, n)
    return {k: scores_dict[k] for k in chosen}


def merge_with_offset(base: dict, extra: dict, offset: int) -> dict:
    """Merge two score dicts, applying an ID offset to extra to avoid collisions."""
    merged = dict(base)
    for k, v in extra.items():
        merged[k + offset] = v
    return merged


def apply_filter(tech: dict, sem: dict, tech_thresh: float, sem_thresh: float, strategy: str) -> dict:
    """Apply a filtering strategy, return summary dict."""
    all_eps = sorted(set(tech.keys()) | set(sem.keys()))

    kept, rem_tech, rem_sem, rem_both = [], [], [], []

    for ep in all_eps:
        t = tech.get(ep)
        s = sem.get(ep)
        p_tech = t is not None and t >= tech_thresh
        p_sem = s is not None and s >= sem_thresh

        if strategy == "technical_only":
            if t is None:
                continue
            (kept if p_tech else rem_tech).append(ep)

        elif strategy == "semantic_only":
            if s is None:
                continue
            (kept if p_sem else rem_sem).append(ep)

        elif strategy == "combined":
            if t is None or s is None:
                continue
            if p_tech and p_sem:
                kept.append(ep)
            elif not p_tech and not p_sem:
                rem_both.append(ep)
            elif not p_tech:
                rem_tech.append(ep)
            else:
                rem_sem.append(ep)

    n = len(all_eps)
    return {
        "strategy": strategy,
        "input_episodes": n,
        "kept": len(kept),
        "kept_fraction": round(len(kept) / n, 4) if n > 0 else 0.0,
        "removed_by_technical": len(rem_tech),
        "removed_by_semantic": len(rem_sem),
        "removed_by_both": len(rem_both),
        # IDs split: first n_clean are clean, rest are bad
        "bad_episodes_caught": {
            "technical_only": sum(1 for ep in rem_tech if ep >= 0),
            "semantic_only": len(rem_sem),
            "both": len(rem_both),
        },
    }


def run_condition(condition: str, results_dir: Path, n_clean: int, n_bad: int,
                  tech_thresh: float, sem_thresh: float, rng: random.Random) -> dict:
    """Run all three strategies for one condition at one contamination level."""
    clean_tech, clean_sem = load_scores(results_dir, "clean")
    bad_tech, bad_sem = load_scores(results_dir, condition)

    # Sample episode IDs once from the intersection of tech+sem keys,
    # then use the same IDs for both score types so they stay aligned.
    clean_common = sorted(set(clean_tech) & set(clean_sem))
    bad_common = sorted(set(bad_tech) & set(bad_sem))
    clean_ids = rng.sample(clean_common, n_clean)
    bad_ids_sampled = rng.sample(bad_common, n_bad)

    c_tech = {ep: clean_tech[ep] for ep in clean_ids}
    c_sem  = {ep: clean_sem[ep]  for ep in clean_ids}
    b_tech = {ep: bad_tech[ep]   for ep in bad_ids_sampled}
    b_sem  = {ep: bad_sem[ep]    for ep in bad_ids_sampled}

    # Merge with offset so IDs don't collide (clean IDs unchanged, bad IDs shifted)
    offset = max(c_tech.keys()) + 1
    tech_merged = merge_with_offset(c_tech, b_tech, offset)
    sem_merged = merge_with_offset(c_sem, b_sem, offset)

    results = {}
    for strategy in ("technical_only", "semantic_only", "combined"):
        results[strategy] = apply_filter(tech_merged, sem_merged, tech_thresh, sem_thresh, strategy)

    # How many bad episodes does each strategy catch?
    bad_ids_offset = set(ep + offset for ep in bad_ids_sampled)
    n_bad_total = len(bad_ids_offset)

    def bad_caught(strategy_result, strategy):
        # Episodes removed that are bad (have offset IDs)
        # We approximate: for technical_only, removed = input - kept; fraction in bad set
        # Actually we need to recompute with ID awareness
        return None  # computed below separately

    # Recompute with bad ID awareness
    def bad_recall(tech_merged, sem_merged, bad_ids_offset, strategy):
        all_eps = sorted(set(tech_merged) | set(sem_merged))
        flagged_bad = 0
        for ep in all_eps:
            t = tech_merged.get(ep)
            s = sem_merged.get(ep)
            p_tech = t is not None and t >= tech_thresh
            p_sem = s is not None and s >= sem_thresh
            is_bad = ep in bad_ids_offset
            if not is_bad:
                continue
            if strategy == "technical_only":
                if t is not None and not p_tech:
                    flagged_bad += 1
            elif strategy == "semantic_only":
                if s is not None and not p_sem:
                    flagged_bad += 1
            elif strategy == "combined":
                if t is not None and s is not None and not (p_tech and p_sem):
                    flagged_bad += 1
        return round(flagged_bad / n_bad_total, 4) if n_bad_total > 0 else 0.0

    for strategy in ("technical_only", "semantic_only", "combined"):
        results[strategy]["bad_recall"] = bad_recall(
            tech_merged, sem_merged, bad_ids_offset, strategy
        )

    return {
        "condition": condition,
        "failure_type": "semantic" if condition in SEMANTIC_FAILURES else "technical",
        "n_clean": n_clean,
        "n_bad": n_bad,
        "contamination_pct": round(n_bad / (n_clean + n_bad) * 100, 1),
        "strategies": results,
    }


def run_multi_condition(conditions: list, results_dir: Path, n_clean: int, n_bad_per_condition: int,
                         tech_thresh: float, sem_thresh: float, rng: random.Random) -> dict:
    """
    Run all three strategies for a mix of multiple bad conditions simultaneously.
    Samples n_bad_per_condition episodes from each bad condition and merges them all.
    """
    clean_tech, clean_sem = load_scores(results_dir, "clean")
    clean_common = sorted(set(clean_tech) & set(clean_sem))
    clean_ids = rng.sample(clean_common, n_clean)
    c_tech = {ep: clean_tech[ep] for ep in clean_ids}
    c_sem  = {ep: clean_sem[ep]  for ep in clean_ids}

    tech_merged = dict(c_tech)
    sem_merged  = dict(c_sem)
    all_bad_ids_offset = set()
    offset = max(c_tech.keys()) + 1

    for condition in conditions:
        bad_tech, bad_sem = load_scores(results_dir, condition)
        bad_common = sorted(set(bad_tech) & set(bad_sem))
        bad_ids_sampled = rng.sample(bad_common, n_bad_per_condition)
        for ep in bad_ids_sampled:
            tech_merged[ep + offset] = bad_tech[ep]
            sem_merged[ep + offset]  = bad_sem[ep]
            all_bad_ids_offset.add(ep + offset)
        offset += max(bad_common) + 1

    n_bad_total = len(all_bad_ids_offset)

    def bad_recall(strategy):
        flagged = 0
        for ep in all_bad_ids_offset:
            t = tech_merged.get(ep)
            s = sem_merged.get(ep)
            p_tech = t is not None and t >= tech_thresh
            p_sem  = s is not None and s >= sem_thresh
            if strategy == "technical_only":
                if t is not None and not p_tech:
                    flagged += 1
            elif strategy == "semantic_only":
                if s is not None and not p_sem:
                    flagged += 1
            elif strategy == "combined":
                if t is not None and s is not None and not (p_tech and p_sem):
                    flagged += 1
        return round(flagged / n_bad_total, 4) if n_bad_total > 0 else 0.0

    strategies = {}
    for strategy in ("technical_only", "semantic_only", "combined"):
        r = apply_filter(tech_merged, sem_merged, tech_thresh, sem_thresh, strategy)
        r["bad_recall"] = bad_recall(strategy)
        strategies[strategy] = r

    failure_types = set()
    for c in conditions:
        failure_types.add("semantic" if c in SEMANTIC_FAILURES else "technical")
    ftype = "mixed" if len(failure_types) > 1 else failure_types.pop()

    return {
        "condition": "+".join(conditions),
        "failure_type": ftype,
        "n_clean": n_clean,
        "n_bad": n_bad_total,
        "n_bad_per_condition": n_bad_per_condition,
        "contamination_pct": round(n_bad_total / (n_clean + n_bad_total) * 100, 1),
        "strategies": strategies,
    }


def print_summary_table(all_results):
    """Print a compact summary table."""
    header = f"{'Condition':<30} {'N_bad':<6} {'Type':<10} {'TechRecall':<12} {'SemRecall':<11} {'CombRecall':<11}"
    print("\n" + "=" * len(header))
    print("Bad Episode Recall by Filtering Strategy")
    print("(fraction of contaminating episodes correctly removed)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in all_results:
        s = r["strategies"]
        print(
            f"{r['condition']:<30} {r['n_bad']:<6} {r['failure_type']:<10} "
            f"{s['technical_only']['bad_recall']:<12.3f} "
            f"{s['semantic_only']['bad_recall']:<11.3f} "
            f"{s['combined']['bad_recall']:<11.3f}"
        )

    print("=" * len(header))


# Pre-defined multi-condition mixes
MULTI_CONDITION_MIXES = {
    "all_semantic":  list(SEMANTIC_FAILURES),                   # wrong_cube + task_fail + extra_objects
    "all_technical": list(TECHNICAL_FAILURES),                  # bad_lighting + shakiness + occluded_top_cam
    "all_mixed":     list(SEMANTIC_FAILURES | TECHNICAL_FAILURES),  # all 6 conditions
}


def main():
    ap = argparse.ArgumentParser(description="Simulate mixed-dataset filtering experiments.")
    ap.add_argument("--results_dir", default="results/",
                    help="Directory with pre-computed score JSON files.")
    ap.add_argument("--conditions", nargs="*", default=list(ALL_BAD),
                    help="Bad conditions to test individually. Default: all 6.")
    ap.add_argument("--clean_episodes", type=int, default=100,
                    help="Number of clean episodes in each simulated mix.")
    ap.add_argument("--levels", type=int, nargs="+", default=[10, 20, 50],
                    help="Contamination counts (per condition for multi-mixes, total for single).")
    ap.add_argument("--technical_threshold", type=float, default=0.5)
    ap.add_argument("--semantic_threshold", type=float, default=0.5)
    ap.add_argument("--output_path", default="results/mixed_validation_summary.json")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skip_single", action="store_true",
                    help="Skip single-condition experiments, only run multi-condition mixes.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    rng = random.Random(args.seed)
    all_results = []

    print(f"Simulating mixed dataset experiments")
    print(f"  clean_episodes={args.clean_episodes}, levels={args.levels}")
    print(f"  technical_threshold={args.technical_threshold}, semantic_threshold={args.semantic_threshold}\n")

    # --- Single-condition experiments ---
    if not args.skip_single:
        print("Single-condition experiments:")
        for condition in args.conditions:
            for n_bad in args.levels:
                print(f"  {condition} + {n_bad} bad...", end=" ", flush=True)
                try:
                    result = run_condition(
                        condition, results_dir,
                        args.clean_episodes, n_bad,
                        args.technical_threshold, args.semantic_threshold,
                        rng,
                    )
                    all_results.append(result)
                    s = result["strategies"]
                    print(
                        f"tech={s['technical_only']['bad_recall']:.2f}  "
                        f"sem={s['semantic_only']['bad_recall']:.2f}  "
                        f"comb={s['combined']['bad_recall']:.2f}"
                    )
                except Exception as e:
                    print(f"ERROR: {e}")

    # --- Multi-condition experiments ---
    print("\nMulti-condition experiments (N_bad = N per condition):")
    for mix_name, mix_conditions in MULTI_CONDITION_MIXES.items():
        for n_bad_per in args.levels:
            label = f"{mix_name} ({n_bad_per}/condition)"
            print(f"  {label}...", end=" ", flush=True)
            try:
                result = run_multi_condition(
                    mix_conditions, results_dir,
                    args.clean_episodes, n_bad_per,
                    args.technical_threshold, args.semantic_threshold,
                    rng,
                )
                result["condition"] = mix_name  # short name for table
                result["n_bad_per_condition"] = n_bad_per
                all_results.append(result)
                s = result["strategies"]
                print(
                    f"n_bad={result['n_bad']}  "
                    f"tech={s['technical_only']['bad_recall']:.2f}  "
                    f"sem={s['semantic_only']['bad_recall']:.2f}  "
                    f"comb={s['combined']['bad_recall']:.2f}"
                )
            except Exception as e:
                print(f"ERROR: {e}")

    print_summary_table(all_results)

    output = {
        "parameters": {
            "clean_episodes": args.clean_episodes,
            "levels": args.levels,
            "technical_threshold": args.technical_threshold,
            "semantic_threshold": args.semantic_threshold,
            "seed": args.seed,
        },
        "results": all_results,
    }

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
