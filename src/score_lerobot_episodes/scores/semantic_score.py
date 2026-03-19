"""
SemanticScorer: I-FailSense integration for score_lerobot_episodes.

Scores an episode on semantic quality (right object, task completed correctly).
Operates directly on the LeRobot dataset object, NOT on video files.
Does not fit into the existing DatasetScorer interface—integrate as a separate
per-episode pass in the scoring pipeline.

Usage:
    scorer = SemanticScorer(
        task_description="pick up the orange cube ...",
        vlm_model_id="ACIDE/FailSense-Calvin-2p-3b",
        fs_weights_path=None,  # None = VLM-only baseline
        device="cuda",
    )
    score = scorer.score_episode(dataset, episode_idx=0)
    # score is float in [0, 1], higher = more likely semantically correct
"""

from __future__ import annotations

from typing import Optional

import torch
from i_failsense.model import FailSense, process_input


class SemanticScorer:
    """
    Wraps I-FailSense (FailSense model) for per-episode semantic scoring.

    Two modes:
    - VLM-only (fs_weights_path=None): decodes the last autoregressive token
      from the PaliGemma2 backbone ("success" or "fail"). This is the
      pre-fine-tuning baseline, equivalent to the paper's "LoRA-only" baseline.
    - Full model (fs_weights_path set): uses FS block classifiers averaged
      over multiple internal VLM layers for a continuous success probability.
    """

    def __init__(
        self,
        task_description: str,
        vlm_model_id: str = "ACIDE/FailSense-Calvin-2p-3b",
        fs_weights_path: Optional[str] = None,
        device: str = "cuda",
        top_camera_key: str = "observation.images.top",
        wrist_camera_key: str = "observation.images.wrist",
    ):
        """
        Args:
            task_description: Natural-language task string passed to the VLM.
            vlm_model_id:     HuggingFace model ID for the FailSense VLM adapter.
            fs_weights_path:  Path to trained FS block checkpoint (.pt file).
                              If None, runs in VLM-only mode (no FS blocks).
            device:           "cuda", "cpu", or "mps".
            top_camera_key:   Dataset feature key for top camera.
            wrist_camera_key: Dataset feature key for wrist camera.
        """
        self.task_description = task_description
        self.top_camera_key = top_camera_key
        self.wrist_camera_key = wrist_camera_key
        self.fs_weights_path = fs_weights_path
        self.mode = "vlm_only" if fs_weights_path is None else "full"
        self._process_input = process_input

        print(f"[SemanticScorer] Loading model: {vlm_model_id}")
        print(f"[SemanticScorer] Mode: {self.mode}")
        print("[SemanticScorer] Note: model loads in float32 then casts to bfloat16.")
        print("                 If VRAM < 12 GB this may OOM during load.")
        print("                 Workaround: ensure no other large tensors are on GPU.")

        self.model = FailSense(vlm_model_id, device=device)

        # Cast to bfloat16 to stay within 10 GB VRAM on RTX 3080.
        # The model is loaded in float32 by I-FailSense (hardcoded). We cast
        # post-hoc. Accept minor precision difference vs native bfloat16 load.
        print("[SemanticScorer] Casting VLM backbone to bfloat16...")
        self.model.vlm_model = self.model.vlm_model.to(torch.bfloat16)

        if fs_weights_path is not None:
            print(f"[SemanticScorer] Loading FS block weights: {fs_weights_path}")
            self.model.load_classifier(fs_weights_path)

        self.model.eval()
        self._device = self.model.device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_episode(self, dataset, episode_idx: int) -> float:
        """
        Score one episode for semantic quality.

        Returns a float in [0, 1]:
            1.0 = model is confident the task was completed correctly.
            0.0 = model is confident this is a semantic failure.
            0.5 = indeterminate (VLM-only mode with unexpected output token).
        """
        from score_lerobot_episodes.semantic_adapter import episode_to_failsense_input  # noqa: PLC0415

        images, task_desc = episode_to_failsense_input(
            dataset,
            episode_idx,
            self.task_description,
            self.top_camera_key,
            self.wrist_camera_key,
        )
        text_prompt = self._process_input(images, task_desc)

        if self.mode == "vlm_only":
            return self._vlm_only_score(images, text_prompt)
        else:
            return self._full_model_score(images, text_prompt)

    # ------------------------------------------------------------------
    # Internal scoring methods
    # ------------------------------------------------------------------

    def _vlm_only_score(self, images: list, text_prompt: str) -> float:
        """
        VLM-only baseline: decode the last autoregressive token from the
        PaliGemma2 backbone. No FS block classifiers used.

        Returns 1.0 for "success", 0.0 for "fail", 0.5 if indeterminate.
        """
        # Wrap in batch dimension: [[pil1..pil8]] and ["<image>...<image> evaluate en task"]
        images_batch = [images]
        text_batch = [text_prompt]

        model_inputs = self.model.processor(
            text=text_batch,
            images=images_batch,
            return_tensors="pt",
            padding=True,
        ).to(self._device)

        # Clear stale hook features and free fragmented GPU memory before forward pass
        self.model.layer_features.clear()
        if self._device.type == "cuda":
            torch.cuda.empty_cache()

        with torch.no_grad():
            output = self.model.vlm_model(**model_inputs)
            last_token_logits = output.logits[:, -1, :]  # [1, vocab_size]

        predicted_id = torch.argmax(last_token_logits, dim=-1)
        decoded = self.model.processor.decode(
            [predicted_id.item()], skip_special_tokens=True
        ).strip().lower()

        if decoded in ("success", "1", "pass"):
            return 1.0
        elif decoded in ("fail", "0", "failure"):
            return 0.0
        else:
            print(f"  [VLM-only] Unexpected output token: '{decoded}' — returning 0.5")
            return 0.5

    def _full_model_score(self, images: list, text_prompt: str) -> float:
        """
        Full model: average probability from FS block classifiers (no VLM voting).
        Returns a float in [0, 1].
        """
        images_batch = [images]
        text_batch = [text_prompt]

        # voting=False: FS classifiers only, no VLM text output
        _, avg_probs = self.model.predict(images_batch, text_batch, voting=False)
        return float(avg_probs.cpu().item())

    def cleanup(self) -> None:
        """Release GPU memory and remove forward hooks."""
        self.model.cleanup()
