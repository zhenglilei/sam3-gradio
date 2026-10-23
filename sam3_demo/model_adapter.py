"""Inference loader for the GE2 layout LoRA checkpoint format."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Keep the same projection and scaling as ge2_layout_autoresearch."""

    def __init__(self, base: nn.Linear, rank: int) -> None:
        super().__init__()
        self.base = base
        self.scaling = 2.0  # GE2 training uses alpha = rank * 2.
        self.lora_a = nn.Linear(base.in_features, rank, bias=False).to(
            device=base.weight.device, dtype=torch.float32
        )
        self.lora_b = nn.Linear(rank, base.out_features, bias=False).to(
            device=base.weight.device, dtype=torch.float32
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_b(self.lora_a(x.float())) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)


def load_ge2_adapter(model: nn.Module, path: str) -> dict:
    checkpoint = Path(path).expanduser().resolve()
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
        handle.seek(0)
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if payload.get("format") != "ge2_layout_lora_v1":
        raise ValueError("Expected a ge2_layout_lora_v1 adapter checkpoint")
    rank = payload["config"]["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("GE2 LoRA rank must be a positive integer")

    tracker = model.inst_interactive_predictor.model
    decoder = tracker.sam_mask_decoder
    replaced = 0
    targets = {"q_proj", "k_proj", "v_proj", "out_proj"}
    for module in list(decoder.transformer.modules()):
        for name, child in list(module.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank))
                replaced += 1
    if not replaced:
        raise ValueError("No PVS TwoWayTransformer projections matched the GE2 adapter")

    # These are full submodule states, including unchanged base projections.
    # Partial/loosely matched loading would silently lose the trained heads.
    decoder.load_state_dict(payload["decoder"], strict=True)
    tracker.sam_prompt_encoder.load_state_dict(payload["prompt_encoder"], strict=True)
    decoder.dynamic_multimask_via_stability = payload["preprocessing"][
        "dynamic_multimask_via_stability"
    ]
    model.requires_grad_(False)
    model.eval()
    torch.clear_autocast_cache()
    return {
        "format": payload["format"],
        "sha256": digest,
        "rank": rank,
        "projections": replaced,
        "step": payload["step"],
        "decoder_tensors": len(payload["decoder"]),
        "prompt_encoder_tensors": len(payload["prompt_encoder"]),
    }
