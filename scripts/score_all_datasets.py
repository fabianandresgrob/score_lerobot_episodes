import argparse
import json
import os

from score_lerobot_episodes.data import load_dataset_hf, organize_by_episode, evaluate_episodes
from score_lerobot_episodes.scores import build_time_stats, DatasetScorer
from score_lerobot_episodes.vlm import VLMInterface

DATASETS = {
    "clean": "j-m-h/pick_place_clean_realsense_downscaled",
    "bad_lighting": "fabiangrob/pick_place_bad_lighting_realsense_downscaled",
    "wrong_cube": "fabiangrob/pick_place_wrong_cube_realsense_downscaled",
    "extra_objects": "fabiangrob/pick_place_extra_objects_realsense_downscaled",
    "shakiness": "fabiangrob/pick_place_shakiness_realsense_downscaled",
    "task_fail": "fabiangrob/pick_place_task_fail_realsense_downscaled",
    "occluded_top_cam": "fabiangrob/pick_place_occluded_top_cam_realsense_downscaled",
}

def score_repo(repo_id, root=None, vision_type="opencv", nominal=None):
    dataset = load_dataset_hf(repo_id, root=root)
    task = dataset.meta.tasks
    episode_map = organize_by_episode(dataset)

    states = [episode_map[i]["states"] for i in episode_map]
    time_stats = build_time_stats(states)

    if vision_type == "opencv":
        vlm_interface = None
    else:
        vlm_interface = VLMInterface(vision_type)

    scorer = DatasetScorer(vlm_interface, time_stats=time_stats)
    rows, agg_mean, output_data = evaluate_episodes(episode_map, scorer, task, nominal)
    return output_data, agg_mean, len(rows)


def main():
    ap = argparse.ArgumentParser(description="Score multiple LeRobot datasets and save JSON results")
    ap.add_argument("--root", required=False, default=None, type=str, help="Optional dataset root path")
    ap.add_argument("--vision_type", required=False, choices=["opencv", "vlm_gemini"], default="opencv")
    ap.add_argument("--nominal", type=float, default=None)
    ap.add_argument("--results_dir", required=False, default="results", type=str)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    for name, repo_id in DATASETS.items():
        print(f"\nScoring dataset: {name} ({repo_id})")
        output_data, agg_mean, num_rows = score_repo(
            repo_id,
            root=args.root,
            vision_type=args.vision_type,
            nominal=args.nominal,
        )

        repo_name = repo_id.replace("/", "_")
        output_file_path = os.path.join(args.results_dir, f"{repo_name}_scores.json")
        with open(output_file_path, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"Saved {num_rows} scored video segments to: {output_file_path}")
        print(f"Aggregate mean score: {agg_mean:.3f}")


if __name__ == "__main__":
    main()
