#!/usr/bin/env bash
# Run VLM-only baseline evaluation on all 7 datasets sequentially.
# Results are saved incrementally — safe to interrupt and re-run.
#
# Usage:
#   bash scripts/run_baseline_all.sh
#   bash scripts/run_baseline_all.sh --dry_run

set -e

DRY_RUN=""
FS_WEIGHTS=""
MODE_SUFFIX="vlmonly"

for arg in "$@"; do
    case $arg in
        --dry_run)   DRY_RUN="--dry_run" ;;
        --fs_weights_path=*) FS_WEIGHTS="${arg#*=}"; MODE_SUFFIX="full" ;;
    esac
done

if [[ -n "$DRY_RUN" ]]; then echo "DRY RUN MODE — processing 5 episodes per dataset"; fi
if [[ -n "$FS_WEIGHTS" ]]; then echo "FULL MODEL MODE — using FS weights: $FS_WEIGHTS"; fi

TASK="pick up the orange cube and place it in the blue container"
PYTHON="python"
SCRIPT="scripts/evaluate_semantic_baseline.py"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    local repo_id=$1
    local condition=$2
    local gt_label=$3
    echo ""
    echo "========================================"
    echo "  $condition  (gt=$gt_label)"
    echo "========================================"
    $PYTHON $SCRIPT \
        --repo_id "$repo_id" \
        --task_description "$TASK" \
        --condition "$condition" \
        --ground_truth_label "$gt_label" \
        --output_path "results/baseline_${condition}_${MODE_SUFFIX}.json" \
        ${FS_WEIGHTS:+--fs_weights_path "$FS_WEIGHTS"} \
        $DRY_RUN
}

# Semantic conditions (relevant for FS block training)
run "j-m-h/pick_place_clean_realsense_downscaled"              clean            1
run "fabiangrob/pick_place_wrong_cube_realsense_downscaled"    wrong_cube       0
run "fabiangrob/pick_place_task_fail_realsense_downscaled"     task_fail        0
run "fabiangrob/pick_place_extra_objects_realsense_downscaled" extra_objects    0

# Technical conditions (not used for FS training, but interesting to evaluate)
run "fabiangrob/pick_place_bad_lighting_realsense_downscaled"      bad_lighting     0
run "fabiangrob/pick_place_shakiness_realsense_downscaled"         shakiness        0
run "fabiangrob/pick_place_occluded_top_cam_realsense_downscaled"  occluded_top_cam 0

echo ""
echo "========================================"
echo "All done. Results in results/"
echo "========================================"
