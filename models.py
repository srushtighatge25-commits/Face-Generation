"""Shared checkpoint-compatible DCGAN model, without web-server startup."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class Generator(nn.Module):
    """DCGAN generator used to create the supplied checkpoint."""

    def __init__(self, latent_size: int = 100, feature_maps: int = 64) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.main = nn.Sequential(
            # (latent_size) x 1 x 1 -> (feature_maps * 8) x 4 x 4
            nn.ConvTranspose2d(latent_size, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            # -> (feature_maps * 4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            # -> (feature_maps * 2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            # -> feature_maps x 32 x 32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            # -> 3 x 64 x 64, normalized to [-1, 1]
            nn.ConvTranspose2d(feature_maps, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.main(noise)


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Accept a plain state dict or common training-checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ("generator_state_dict", "state_dict", "model_state_dict", "generator"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint does not contain a generator state dictionary.")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = key
        for prefix in ("module.", "generator.", "netG.", "_orig_mod."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        state_dict[clean_key] = value

    if "main.0.weight" not in state_dict:
        raise KeyError("Expected generator layer 'main.0.weight' was not found.")
    return state_dict


def load_generator(model_path: Path | str, device: torch.device | str = "cpu") -> Generator:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with PyTorch versions before weights_only.
        checkpoint = torch.load(model_path, map_location=device)

    state_dict = _extract_state_dict(checkpoint)
    first_layer = state_dict["main.0.weight"]
    latent_size = int(first_layer.shape[0])
    feature_maps = int(first_layer.shape[1]) // 8

    model = Generator(latent_size=latent_size, feature_maps=feature_maps).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

