"""
LeRobot episode → I-FailSense input adapter.

Converts a single episode from a LeRobot dataset into the flat list of 8 PIL
images that I-FailSense expects:
    [top_t0, top_t1, top_t2, top_t3, wrist_t0, wrist_t1, wrist_t2, wrist_t3]

Camera order matches the 2pov convention used in FailSense-Calvin-2p-3b:
images_1 (top) first, then images_2 (wrist), each with 4 evenly-spaced frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_episode_frame_bounds(dataset, episode_idx: int) -> Tuple[int, int]:
    """Return (dataset_from_index, dataset_to_index) for the episode."""
    ep = dataset.meta.episodes[episode_idx]
    return ep["dataset_from_index"], ep["dataset_to_index"]


def _sample_4_indices(from_idx: int, to_idx: int) -> List[int]:
    """Return 4 evenly-spaced global frame indices spanning [from_idx, to_idx)."""
    total = to_idx - from_idx
    if total < 4:
        # Repeat last frame if episode is very short (edge case)
        indices = list(range(from_idx, to_idx))
        while len(indices) < 4:
            indices.append(indices[-1])
        return indices
    return [from_idx + int(i * (total - 1) / 3) for i in range(4)]


def _load_frame_via_cv2(dataset, global_idx: int, camera_key: str) -> torch.Tensor:
    """
    Read a video frame directly from disk using cv2.

    Fallback for when dataset[i] silently omits video keys. Constructs the
    video path from dataset.root + LeRobot v2 chunk layout and seeks to the
    correct frame by global dataset index.

    dataset.root is expected to include the repo_id, e.g.:
        /data/lerobot_datasets/fabiangrob/pick_place_mixed_unfiltered
    """
    root = Path(dataset.root)
    # dataset.root may or may not include the repo_id depending on how the
    # dataset was loaded (HF cache includes it; --root does not). Try both.
    candidates = [
        root / "videos" / camera_key,
        root / dataset.repo_id / "videos" / camera_key,
    ]
    cam_dir = next((p for p in candidates if p.is_dir()), None)
    if cam_dir is None:
        raise FileNotFoundError(
            f"Camera video directory not found (tried: "
            + ", ".join(str(p) for p in candidates) + ")"
        )

    chunk_files = sorted(cam_dir.glob("chunk-*/file-000.mp4"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk video files (chunk-*/file-000.mp4) under {cam_dir}")

    # Get frames-per-chunk from first video
    cap0 = cv2.VideoCapture(str(chunk_files[0]))
    chunk_size = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    cap0.release()

    chunk_idx = min(global_idx // chunk_size, len(chunk_files) - 1)
    frame_in_chunk = global_idx % chunk_size
    video_path = chunk_files[chunk_idx]

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_in_chunk)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(
            f"cv2 failed to read frame {frame_in_chunk} (global {global_idx}) from {video_path}"
        )

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame_rgb).permute(2, 0, 1)  # CHW uint8


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a CHW image tensor to a PIL RGB Image.
    Handles float [0,1] and uint8 [0,255] tensors.
    """
    if tensor.dtype in (torch.float32, torch.float16, torch.bfloat16):
        tensor = (tensor.float() * 255).clamp(0, 255).byte()
    # CHW → HWC
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def episode_to_failsense_input(
    dataset,
    episode_idx: int,
    task_description: str,
    top_camera_key: str = "observation.images.top",
    wrist_camera_key: str = "observation.images.wrist",
    target_size: Tuple[int, int] | None = None,
) -> Tuple[List[Image.Image], str]:
    """
    Convert a LeRobot episode to the input format expected by I-FailSense.

    Args:
        dataset:          LeRobotDataset object.
        episode_idx:      Zero-based episode index.
        task_description: Natural-language task description string.
        top_camera_key:   Feature key for the top (exocentric) camera.
        wrist_camera_key: Feature key for the wrist (egocentric) camera.
        target_size:      (H, W) to resize each frame to, or None to keep
                          original resolution (default). The VLM processor
                          handles resizing internally, so resizing here is
                          only needed for visualisation.

    Returns:
        images:           Flat list of 8 PIL Images:
                          [top_t0, top_t1, top_t2, top_t3,
                           wrist_t0, wrist_t1, wrist_t2, wrist_t3]
        task_description: Unchanged pass-through for convenience.

    Raises:
        KeyError: If camera keys are not present in the dataset features.
    """
    from_idx, to_idx = _get_episode_frame_bounds(dataset, episode_idx)
    sample_indices = _sample_4_indices(from_idx, to_idx)

    resize = transforms.Resize(target_size, antialias=True) if target_size is not None else None

    top_frames: List[Image.Image] = []
    wrist_frames: List[Image.Image] = []

    _warned_fallback = False
    for global_idx in sample_indices:
        sample = dataset[global_idx]

        if top_camera_key in sample:
            top_tensor = sample[top_camera_key]
            wrist_tensor = sample[wrist_camera_key]
        else:
            if not _warned_fallback:
                print(
                    f"\n[semantic_adapter] dataset[i] missing video keys "
                    f"(got: {list(sample.keys())}); "
                    "falling back to direct cv2 video reading.",
                    flush=True,
                )
                _warned_fallback = True
            top_tensor = _load_frame_via_cv2(dataset, global_idx, top_camera_key)
            wrist_tensor = _load_frame_via_cv2(dataset, global_idx, wrist_camera_key)

        if resize:
            top_tensor = resize(top_tensor)
            wrist_tensor = resize(wrist_tensor)
        top_frames.append(_tensor_to_pil(top_tensor))
        wrist_frames.append(_tensor_to_pil(wrist_tensor))

    # I-FailSense 2pov order: all top frames first, then all wrist frames
    images = top_frames + wrist_frames
    return images, task_description


def verify_grid_visually(
    images: List[Image.Image],
    output_path: str = "adapter_verification.png",
) -> None:
    """
    Save a debug grid image for visual verification that the adapter output
    is correct: top camera on top row, wrist camera on bottom row,
    timesteps progressing left to right.

    Call this on a few episodes before running any evaluation.
    """
    from PIL import ImageDraw, ImageFont

    assert len(images) == 8, f"Expected 8 images, got {len(images)}"
