from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image


@dataclass
class ImageData:
    rgb: np.ndarray  # (H, W, 3) uint8
    path: str = ""


def load_image(path: str) -> ImageData:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    return ImageData(rgb=arr, path=path)


def choose_keyframes(paths: Sequence[str], k: int = 5) -> List[str]:
    """Choose up to k representative frames uniformly."""
    if not paths:
        return []
    if len(paths) <= k:
        return list(paths)
    idxs = np.linspace(0, len(paths) - 1, num=k, dtype=int)
    return [paths[i] for i in idxs.tolist()]


def relative_luminance_from_rgb(rgb: np.ndarray) -> float:
    """Paper Eq.(1): relative luminance (0..1).

    L = 1/255 * (0.2126 R + 0.7152 G + 0.0722 B)
    where R,G,B are 0..255.
    """
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Expected RGB image array of shape (H, W, 3).")

    R = float(rgb[..., 0].mean())
    G = float(rgb[..., 1].mean())
    B = float(rgb[..., 2].mean())
    L = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255.0
    return float(max(0.0, min(1.0, L)))


def smoothed_brightness(image_paths: Sequence[str], k: int = 5) -> float:
    keyframes = choose_keyframes(image_paths, k=k)
    if not keyframes:
        return 0.0
    vals: List[float] = []
    for p in keyframes:
        img = load_image(p)
        vals.append(relative_luminance_from_rgb(img.rgb))
    return float(np.mean(vals)) if vals else 0.0


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def blur_score_variance_of_laplacian(rgb: np.ndarray) -> float:
    """Optional image quality metric: variance of Laplacian (higher = sharper).

    This is NOT in the excerpt; included as optional debugging/extension and is not used
    by default thresholds.
    """
    gray = _to_grayscale(rgb)
    # 3x3 Laplacian kernel
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # Convolution (valid)
    H, W = gray.shape
    if H < 3 or W < 3:
        return 0.0
    out = np.zeros((H - 2, W - 2), dtype=np.float32)
    for i in range(H - 2):
        for j in range(W - 2):
            patch = gray[i : i + 3, j : j + 3]
            out[i, j] = float(np.sum(patch * k))
    return float(np.var(out))
