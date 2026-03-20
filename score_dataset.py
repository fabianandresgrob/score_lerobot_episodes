import argparse, pathlib, re, sys, warnings, cv2, numpy as np, pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple

from score_lerobot_episodes.vlm import VLMInterface
from score_lerobot_episodes.data import organize_by_episode, load_dataset_hf, save_filtered_dataset, get_scorable_video_path, get_scorable_video_segment
from score_lerobot_episodes.scores import score_task_success, score_visual_clarity, score_smoothness, score_path_efficiency, score_collision, score_runtime, score_joint_stability, score_gripper_consistency, score_idle_velocity, score_actuator_saturation
from score_lerobot_episodes.scores import build_time_stats, DatasetScorer     # (your helper from the other file)
from train import start_training
from score_lerobot_episodes.evaluation import get_eval_episodes, run_eval
from score_lerobot_episodes.util import VideoSegment
from score_lerobot_episodes.data import evaluate_episodes
import hashlib
import pickle
import os
import uniplot

# Default path to I-FailSense/src relative to this file's location
_FAILSENSE_SRC_DEFAULT = str(Path(__file__).resolve().parent.parent / "I-FailSense" / "src")

# Handle different lerobot versions for imports
try:
    import lerobot
    from packaging import version
    lerobot_version = version.parse(lerobot.__version__)

    if lerobot_version <= version.parse("0.4.0"):
        # Old version: <= 0.4.0
        from lerobot.constants import HF_LEROBOT_HOME
    else:
        # New version: > 0.4.0
        from lerobot.utils.constants import HF_LEROBOT_HOME
except Exception:
    # Fallback: try new import first, then old
    try:
        from lerobot.utils.constants import HF_LEROBOT_HOME
    except (ImportError, AttributeError):
        from lerobot.constants import HF_LEROBOT_HOME

from lerobot.configs.train import TrainPipelineConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_id", required=True, type=str)
    ap.add_argument("--root", required=False, default=None, type=str)
    ap.add_argument("--output", required=False, type=str, default=None)
    ap.add_argument("--overwrite", required=False, type=bool, default=True)
    ap.add_argument("--overwrite_checkpoint", required=False, type=bool, default=False)
    ap.add_argument("--nominal", type=float)
    ap.add_argument("--vision_type", required=False, choices=["opencv", "vlm_gemini"], default="opencv")
    ap.add_argument("--policy_name", type = str, default = "act")
    ap.add_argument("--threshold", type = float, default = 0.5)
    ap.add_argument("--train-baseline", type=bool, default=False)
    ap.add_argument("--train-filtered", type=bool, default=False)
    ap.add_argument("--plot", required=False, type=bool, default=False)
    # Semantic scoring (I-FailSense)
    ap.add_argument("--semantic", action="store_true",
                    help="Enable semantic scoring with I-FailSense.")
    ap.add_argument("--task_description", type=str, default=None,
                    help="Task description string (required when --semantic is set).")
    ap.add_argument("--semantic_model_id", type=str,
                    default="ACIDE/FailSense-Calvin-2p-3b",
                    help="HuggingFace model ID for I-FailSense.")
    ap.add_argument("--semantic_fs_weights", type=str, default=None,
                    help="Path to fine-tuned FS block checkpoint (.pt). "
                         "Omit for VLM-only baseline mode.")
    ap.add_argument("--semantic_threshold", type=float, default=0.5,
                    help="Semantic score threshold for filtering (default 0.5).")
    ap.add_argument("--video_backend", type=str, default="pyav",
                    help="Video backend for LeRobot dataset loading (default: pyav).")
    args = ap.parse_args()

    if args.semantic and not args.task_description:
        raise ValueError("--task_description is required when --semantic is set.")


    # Load dataset.
    dataset = load_dataset_hf(args.repo_id, root=args.root, video_backend=args.video_backend)
    task = dataset.meta.tasks

    # This maps episode_id to video path (by camera key), states and actions.
    episode_map = organize_by_episode(dataset)


    # Compute runtimes stats of all episodes.
    states = [episode_map[i]['states'] for i in episode_map]
    time_stats = build_time_stats(states)         # ← q1, q3, mean, std, …

    if args.vision_type == 'opencv':
        vlm_interface = None
    else:
        vlm_interface = VLMInterface(args.vision_type)
    scorer = DatasetScorer(vlm_interface, time_stats=time_stats)

    # ------------------------------------------------------------------
    #  Evaluate every episode (technical scores)
    # ------------------------------------------------------------------
    rows, agg_mean, output_data = evaluate_episodes(episode_map, scorer, task, args.nominal)

    # ------------------------------------------------------------------
    #  Semantic scoring pass (per-episode, separate from technical scores)
    # ------------------------------------------------------------------
    semantic_scores = {}  # {episode_idx: float}
    if args.semantic:
        from score_lerobot_episodes.scores.semantic_score import SemanticScorer
        sem_scorer = SemanticScorer(
            task_description=args.task_description,
            vlm_model_id=args.semantic_model_id,
            fs_weights_path=args.semantic_fs_weights,
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        )
        print("\nRunning semantic scoring...")
        for ep_idx in sorted(episode_map.keys()):
            print(f"  Semantic scoring episode {ep_idx}...", end=" ", flush=True)
            try:
                score = sem_scorer.score_episode(dataset, ep_idx)
                semantic_scores[ep_idx] = score
                print(f"{score:.3f}")
            except Exception as e:
                import traceback
                print(f"ERROR: {e}")
                traceback.print_exc()
                semantic_scores[ep_idx] = None
        sem_scorer.cleanup()

        # Attach semantic_score to each output_data entry (per camera row)
        for entry in output_data:
            entry["semantic_score"] = semantic_scores.get(entry["episode_id"])

    # Create the results directory if it doesn't exist.
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Define the output file name based on the repo_id.
    repo_name = args.repo_id.replace("/", "_")
    output_file_path = os.path.join(results_dir, f"{repo_name}_scores.json")

    with open(output_file_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Successfully saved scores to: {output_file_path}")

    # ------------------------------------------------------------------
    #  Pretty-print results
    # ------------------------------------------------------------------
    crit_names = list(scorer.criteria.keys())

    EP_W, CAM_W, SC_W, AG_W, SEM_W, ST_W = 8, 15, 11, 10, 9, 16
    score_fmt = f'{{:>{SC_W}.3f}}'
    sem_fmt   = f'{{:>{SEM_W}}}'    # semantic score or '  —  '

    has_semantic = bool(semantic_scores)

    header_line = (
        f'{"Episode":<{EP_W}}{"Camera":<{CAM_W}}'
        + ''.join(f'{h:>{SC_W}s}' for h in crit_names)
        + f'  {"Aggregate":>{AG_W}s}'
        + (f'  {"Semantic":>{SEM_W}s}' if has_semantic else '')
        + f'  Status'
    )
    divider = '─' * len(header_line)

    print('\nEpisode scores (0–1 scale)')
    print(divider)
    print(header_line)

    distributions = {}
    good_episodes = {}          # {ep_idx: passes_technical}
    printed_semantic = set()    # avoid printing semantic column twice per episode

    for ep_idx, cam, _vid_path, total, subs in rows:
        for k in crit_names:
            distributions.setdefault(k, []).append(subs[k])

        cleaned_cam = cam.replace('observation.images.', '')
        passes_tech = total >= args.threshold

        if ep_idx not in good_episodes or good_episodes[ep_idx]:
            good_episodes[ep_idx] = passes_tech

        # Semantic status (show only on first camera row per episode)
        sem_score = semantic_scores.get(ep_idx)
        if has_semantic and ep_idx not in printed_semantic:
            sem_str = f'{sem_score:.3f}' if sem_score is not None else '  err'
            printed_semantic.add(ep_idx)
        elif has_semantic:
            sem_str = '     '   # blank for repeated episode rows
        else:
            sem_str = None

        passes_sem = (sem_score is not None and sem_score >= args.semantic_threshold) if has_semantic else None

        if has_semantic:
            if passes_tech and passes_sem:
                status = 'GOOD'
            elif not passes_tech and passes_sem:
                status = 'FAIL_TECHNICAL'
            elif passes_tech and not passes_sem:
                status = 'FAIL_SEMANTIC'
            else:
                status = 'FAIL_BOTH'
        else:
            status = 'GOOD' if passes_tech else 'BAD'

        row = (
            f'{ep_idx:<{EP_W}}{cleaned_cam:<{CAM_W}}'
            + ''.join(score_fmt.format(subs[k]) for k in crit_names)
            + f'  {total:>{AG_W}.3f}'
            + (f'  {sem_str:>{SEM_W}}' if has_semantic else '')
            + f'  {status}'
        )
        print(row)

    print(divider)
    print(f'Average aggregate over {len(rows)} videos: {agg_mean:.3f}')
    if has_semantic:
        valid_sem = [v for v in semantic_scores.values() if v is not None]
        if valid_sem:
            print(f'Average semantic score over {len(valid_sem)} episodes: '
                  f'{sum(valid_sem)/len(valid_sem):.3f}')
    print('')

    # Episodes that pass the technical threshold
    good_episodes_list = [k for k in good_episodes if good_episodes[k]]

    # Additionally filter on semantic threshold if enabled
    if has_semantic:
        good_episodes_list = [
            ep for ep in good_episodes_list
            if semantic_scores.get(ep) is not None
            and semantic_scores[ep] >= args.semantic_threshold
        ]

    if len(good_episodes_list) == 0:
        raise ValueError(f'All episodes filtered out, decrease threshold to fix this. Current threshold: {args.threshold}')
    total_episodes = len(episode_map)
    num_removed = total_episodes - len(good_episodes_list)

    print(f'Percentage of episodes removed: {float(num_removed)/total_episodes}, total: {num_removed}')
    print('')

    # Need to find actual dataset path on disk.
    dataset_path = args.root
    if not dataset_path:
        cache_dir = HF_LEROBOT_HOME
        dataset_path = os.path.join(cache_dir, args.repo_id)

    if args.output:
        save_filtered_dataset(dataset_path, args.output, good_episodes_list, overwrite=args.overwrite)

    # Training config required args.
    #  --dataset.repo_id=${HF_USER}/trossen_ai_stationary_test \
    #  --policy.type=act \
    #  --output_dir=outputs/train/act_trossen_ai_stationary_test \
    #  --job_name=act_trossen_ai_stationary_test \
    #  --device=cuda \
    #  --wandb.enable=true

    baseline_eval_episodes, filtered_eval_episodes = get_eval_episodes(good_episodes_list)

    if args.train_baseline:
        pretrained_model_path, wandb_id = start_training(args.repo_id, root=args.root, policy_name=args.policy_name, job_name='baseline', overwrite_checkpoint=args.overwrite_checkpoint)
        run_eval(pretrained_model_path, args.repo_id, wandb_id, baseline_eval_episodes, root=args.root)
    if args.train_filtered and num_removed == 0:
        print('WARNING: Not training because nothing was removed.')
    elif args.train_filtered:
        # We need to do this manually because the args.repo_id may not always match the supplied args.output
        filtered_job_name = f'filtered_{args.threshold}'
        filtered_repo_id = '/'.join(args.output.split('/')[-2:])
        pretrained_model_path, wandb_id = start_training(filtered_repo_id, root=args.output, policy_name=args.policy_name, job_name=filtered_job_name, overwrite_checkpoint=args.overwrite_checkpoint)
        run_eval(pretrained_model_path, filtered_repo_id, wandb_id, filtered_eval_episodes, root=args.output)

    if args.plot:
        for k in crit_names:
            uniplot.histogram(distributions[k],
                          bins=20,
                          bins_min=0,  # avoid breaking if all data lands in 1 bucket
                          title=f'distribution for {k}',
                          x_min=0,
                          x_max=1)

if __name__ == "__main__":
    main()
