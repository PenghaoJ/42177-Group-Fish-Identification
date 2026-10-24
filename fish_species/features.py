"""Identical colour/texture extraction during training and website inference.

Only the resized photograph contributes features: letterbox padding is excluded.
No segmentation annotations, filename, fish ID or shape descriptors are inputs.
"""
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray, rgb2hsv, rgb2lab
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


@dataclass(frozen=True)
class FeatureConfig:
    image_size: int = 224
    hs_bins: tuple = (18, 8)
    value_bins: int = 16
    lbp_settings: tuple = ((8, 1), (16, 2))
    glcm_levels: int = 16
    glcm_distances: tuple = (1, 3)
    use_foreground_mask: bool = False
    version: str = "hsv-lab-lbp-glcm-v1"

    def to_dict(self):
        return asdict(self)


def _root_histogram(histogram):
    histogram = np.asarray(histogram, dtype=np.float64).ravel()
    return np.sqrt(histogram / max(float(histogram.sum()), 1.0))


def extract_features(image, config=None):
    """Return 221 colour/texture features for one PIL image.

    ImageOps.contain gives the valid rectangle of the equivalent 224x224
    letterbox. Features operate on that rectangle, so padding never contributes.
    GLCM directions are pooled to reduce sensitivity to fish orientation.
    """
    config = config or FeatureConfig()
    if config.use_foreground_mask:
        raise ValueError("This deployable baseline does not use ground-truth masks")
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = ImageOps.contain(
        image, (config.image_size, config.image_size), Image.Resampling.LANCZOS
    )
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    hsv = rgb2hsv(rgb)
    colour = [
        _root_histogram(np.histogram2d(
            hsv[..., 0].ravel(), hsv[..., 1].ravel(),
            bins=config.hs_bins, range=((0, 1), (0, 1)),
        )[0]),
        _root_histogram(np.histogram(
            hsv[..., 2], bins=config.value_bins, range=(0, 1)
        )[0]),
    ]
    lab = rgb2lab(rgb)
    for channel, divisor in enumerate((100.0, 128.0, 128.0)):
        values = lab[..., channel] / divisor
        mean, std = float(values.mean()), float(values.std())
        skew = float(np.mean((values - mean) ** 3) / (std ** 3 + 1e-10))
        colour.append(np.array([mean, std, skew]))

    gray = np.clip(np.rint(rgb2gray(rgb) * 255.0), 0, 255).astype(np.uint8)
    texture = []
    for points, radius in config.lbp_settings:
        lbp = local_binary_pattern(gray, points, radius, method="uniform")
        # Exclude the descriptor's boundary, including very narrow-image cases.
        if min(gray.shape) > 2 * radius:
            lbp = lbp[radius:-radius, radius:-radius]
        texture.append(_root_histogram(np.histogram(
            lbp, bins=points + 2, range=(0, points + 2)
        )[0]))
    quantized = (gray.astype(np.uint16) * config.glcm_levels // 256).astype(np.uint8)
    glcm = graycomatrix(
        quantized, distances=list(config.glcm_distances),
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=config.glcm_levels, symmetric=True, normed=True,
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"):
        values = graycoprops(glcm, prop)
        texture.extend([values.mean(axis=1), values.std(axis=1)])
    features = np.concatenate(colour + texture).astype(np.float64)
    if features.shape != (221,) or not np.isfinite(features).all():
        raise ValueError("Invalid colour/texture feature vector")
    return features
