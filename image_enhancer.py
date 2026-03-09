"""
image_enhancer.py
Adaptive Medical Image Enhancement Pipeline.
Performs noise estimation, auto-parameter tuning, and applies
CLAHE, Gaussian, and Median filters adaptively.
"""

import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.ndimage import uniform_filter
import os
import uuid


def estimate_noise(image):
    """Estimate noise level using Laplacian variance method."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_level = laplacian.var()
    return float(noise_level)


def compute_entropy(image):
    """Compute image entropy to measure information content."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    ent = entropy(gray, disk(5))
    return float(np.mean(ent))


def auto_clahe_clip_limit(noise_level):
    """Auto-determine CLAHE clip limit based on noise level."""
    if noise_level < 100:
        return 4.0
    elif noise_level < 500:
        return 3.0
    elif noise_level < 1500:
        return 2.0
    else:
        return 1.5


def auto_kernel_size(noise_level):
    """Auto-determine filter kernel size based on noise level."""
    if noise_level < 100:
        return 3
    elif noise_level < 500:
        return 5
    elif noise_level < 1500:
        return 7
    else:
        return 9


def apply_clahe(image, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a, b))
        result = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        result = clahe.apply(image)
    return result


def apply_gaussian_filter(image, kernel_size=5):
    """Apply Gaussian blur filter."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_median_filter(image, kernel_size=5):
    """Apply Median filter for salt-and-pepper noise removal."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def select_optimal_filter(image, noise_level):
    """Select optimal filter based on noise level and entropy."""
    img_entropy = compute_entropy(image)

    if noise_level > 1000:
        return 'median'
    elif img_entropy < 3.0:
        return 'clahe'
    elif noise_level > 300:
        return 'gaussian'
    else:
        return 'clahe'


def enhance_image(image_path, output_dir='static/uploads',
                  use_clahe=True, use_gaussian=False, use_median=False,
                  auto_mode=True):
    """
    Full adaptive enhancement pipeline.

    Args:
        image_path: Path to input image.
        output_dir: Directory to save enhanced image.
        use_clahe: Apply CLAHE filter.
        use_gaussian: Apply Gaussian filter.
        use_median: Apply Median filter.
        auto_mode: Auto-select filters based on noise analysis.

    Returns:
        dict with enhanced_path, metrics, and filter info.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original = image.copy()
    noise_level = estimate_noise(image)
    img_entropy = compute_entropy(image)

    clip_limit = auto_clahe_clip_limit(noise_level)
    kernel_size = auto_kernel_size(noise_level)

    filters_applied = []

    if auto_mode:
        optimal = select_optimal_filter(image, noise_level)
        if optimal == 'clahe':
            image = apply_clahe(image, clip_limit=clip_limit)
            filters_applied.append('CLAHE')
        elif optimal == 'gaussian':
            image = apply_gaussian_filter(image, kernel_size=kernel_size)
            filters_applied.append('Gaussian')
        elif optimal == 'median':
            image = apply_median_filter(image, kernel_size=kernel_size)
            filters_applied.append('Median')

        image = apply_clahe(image, clip_limit=clip_limit)
        if 'CLAHE' not in filters_applied:
            filters_applied.append('CLAHE')
    else:
        if use_median:
            image = apply_median_filter(image, kernel_size=kernel_size)
            filters_applied.append('Median')
        if use_gaussian:
            image = apply_gaussian_filter(image, kernel_size=kernel_size)
            filters_applied.append('Gaussian')
        if use_clahe:
            image = apply_clahe(image, clip_limit=clip_limit)
            filters_applied.append('CLAHE')

    enhanced_noise = estimate_noise(image)
    enhanced_entropy = compute_entropy(image)

    original_brightness = float(np.mean(original))
    enhanced_brightness = float(np.mean(image))
    brightness_improvement = float(
        ((enhanced_brightness - original_brightness) / (original_brightness + 1e-6)) * 100
    )

    noise_reduction = float(
        ((noise_level - enhanced_noise) / (noise_level + 1e-6)) * 100
    )

    original_contrast = float(np.std(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original))
    enhanced_contrast = float(np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image))
    contrast_improvement = float(
        ((enhanced_contrast - original_contrast) / (original_contrast + 1e-6)) * 100
    )

    quality_score = min(100.0, max(0.0,
        50.0 + (contrast_improvement * 0.3) + (noise_reduction * 0.2) + (brightness_improvement * 0.1)
    ))

    os.makedirs(output_dir, exist_ok=True)
    filename = f"enhanced_{uuid.uuid4().hex[:8]}.png"
    enhanced_path = os.path.join(output_dir, filename)
    cv2.imwrite(enhanced_path, image)

    metrics = {
        'noise_level_original': round(noise_level, 2),
        'noise_level_enhanced': round(enhanced_noise, 2),
        'noise_reduction_pct': round(noise_reduction, 2),
        'entropy_original': round(img_entropy, 2),
        'entropy_enhanced': round(enhanced_entropy, 2),
        'brightness_improvement_pct': round(brightness_improvement, 2),
        'contrast_improvement_pct': round(contrast_improvement, 2),
        'quality_score': round(quality_score, 2),
        'clip_limit': clip_limit,
        'kernel_size': kernel_size,
        'filters_applied': filters_applied,
        'auto_mode': auto_mode
    }

    return {
        'enhanced_path': enhanced_path,
        'metrics': metrics
    }
