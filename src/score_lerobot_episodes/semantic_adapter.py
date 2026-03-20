"""
LeRobot episode → I-FailSense input adapter.

Converts a single episode from a LeRobot dataset into the flat list of 8 PIL
images that I-FailSense expects:
    [top_t0, top_t1, top_t2, top_t3, wrist_t0, wrist_t1, wrist_t2, wrist_t3]

Camera order matches the 2pov convention used in FailSense-Calvin-2p-3b:
images_1 (top) first, then images_2 (wrist), each with 4 evenly-spaced frames.
"""

from __future__ import annotations

from typing import List, Tuple

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

    for global_idx in sample_indices:
        sample = dataset[global_idx]
        if global_idx == sample_indices[0]:
            print(f"\nDEBUG sample keys: {list(sample.keys())}", flush=True)
        top_tensor = resize(sample[top_camera_key]) if resize else sample[top_camera_key]
        wrist_tensor = resize(sample[wrist_camera_key]) if resize else sample[wrist_camera_key]
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
