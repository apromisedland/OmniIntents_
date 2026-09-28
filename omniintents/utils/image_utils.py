"""Paper luminance formula and vectorized image utilities."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image

from ..errors import ValidationError


@dataclass
class ImageData:
    rgb: np.ndarray
    path: str = ""


def load_image(path: str) -> ImageData:
    try:
        with Image.open(path) as image:
            return ImageData(np.asarray(image.convert("RGB")), path)
    except OSError as error:
        raise ValidationError("Unreadable image input") from error


def choose_keyframes(paths: Sequence[str], k: int = 5) -> list[str]:
    if k <= 0 or not paths:
        return []
    indices = np.linspace(0, len(paths) - 1, min(len(paths), k), dtype=int)
    return [paths[index] for index in indices]


def relative_luminance_from_rgb(rgb: np.ndarray) -> float:
    values = np.asarray(rgb, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3 or not values.size:
        raise ValidationError("Expected a non-empty RGB array")
    if not np.all(np.isfinite(values)) or values.min() < 0 or values.max() > 255:
        raise ValidationError("RGB values must be finite and in 0..255")
    return float(np.mean(values @ np.array([0.2126, 0.7152, 0.0722])) / 255)


def smoothed_brightness(image_paths: Sequence[str], k: int = 5) -> float | None:
    paths = choose_keyframes(image_paths, k)
    return float(np.mean([relative_luminance_from_rgb(load_image(path).rgb) for path in paths])) if paths else None


def blur_score_variance_of_laplacian(rgb: np.ndarray) -> float:
    gray = np.asarray(rgb, dtype=np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    if min(gray.shape) < 3:
        return 0.0
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
        + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    return float(np.var(laplacian))
