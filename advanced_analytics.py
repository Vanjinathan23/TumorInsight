"""
advanced_analytics.py
Advanced Analytical Features Module.
Provides Image Quality Assessment, Tumor Area Percentage,
Region Suspicion Ranking, and Explainable AI Insight Generation.
All metrics are computed from the actual uploaded scan — no mock values.
"""

import cv2
import numpy as np


# ─── 1. IMAGE QUALITY ASSESSMENT ────────────────────────────────

def assess_image_quality(image_path):
    """
    Analyze the uploaded scan and compute image quality metrics
    BEFORE enhancement.

    Computes:
        - Noise level via pixel standard deviation
        - Blur detection via Laplacian variance
        - Contrast score via intensity range

    Returns:
        dict with image_quality_score (0-100) and quality_label.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- Noise level (pixel standard deviation) ---
    noise_level = float(np.std(gray))

    # --- Blur detection (Laplacian variance) ---
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian.var())

    # --- Contrast score (intensity range normalized to 0-1) ---
    min_intensity = float(np.min(gray))
    max_intensity = float(np.max(gray))
    contrast_score = float((max_intensity - min_intensity) / 255.0)

    # --- Combine into overall quality score (0-100) ---
    # Normalize each component to 0-1 range, then weight

    # Noise: lower std (< 10) = uniform/poor, moderate (20-60) = good,
    # very high (> 80) = noisy/poor. Peak quality around 30-50.
    if noise_level < 10:
        noise_quality = noise_level / 10.0 * 0.4  # very uniform = low quality
    elif noise_level <= 60:
        noise_quality = 0.4 + (noise_level - 10) / 50.0 * 0.6  # sweet spot
    else:
        noise_quality = max(0.0, 1.0 - (noise_level - 60) / 100.0)  # too noisy

    # Blur: higher Laplacian variance = sharper = better quality
    # Typical range: 10-5000+
    blur_quality = min(1.0, blur_score / 1000.0)

    # Contrast: higher range = better quality
    contrast_quality = contrast_score  # already 0-1

    # Weighted combination
    quality_score = (
        noise_quality * 0.30 +
        blur_quality * 0.40 +
        contrast_quality * 0.30
    ) * 100.0

    quality_score = round(min(100.0, max(0.0, quality_score)), 2)

    # Label
    if quality_score < 35:
        quality_label = "Poor"
    elif quality_score < 65:
        quality_label = "Moderate"
    else:
        quality_label = "Good"

    return {
        'image_quality_score': quality_score,
        'quality_label': quality_label,
        'quality_details': {
            'noise_level': round(noise_level, 2),
            'blur_score': round(blur_score, 2),
            'contrast_score': round(contrast_score, 4),
            'noise_quality': round(noise_quality, 4),
            'blur_quality': round(blur_quality, 4),
            'contrast_quality': round(contrast_quality, 4)
        }
    }


# ─── 2. TUMOR AREA PERCENTAGE ───────────────────────────────────

def compute_tumor_area_percentage(regions, image_area):
    """
    Compute the percentage of the image occupied by tumor regions
    AFTER segmentation.

    Formula:
        tumor_area_percentage = (total_tumor_pixels / total_image_pixels) * 100

    Args:
        regions: list of region dicts from ai_analysis (each has 'area' key).
        image_area: total image pixel count (height * width).

    Returns:
        dict with tumor_area_percentage.
    """
    if not regions or image_area <= 0:
        return {'tumor_area_percentage': 0.0}

    total_tumor_pixels = sum(r.get('area', 0) for r in regions)
    percentage = (total_tumor_pixels / image_area) * 100.0
    percentage = round(float(percentage), 4)

    return {'tumor_area_percentage': percentage}


# ─── 3. REGION SUSPICION RANKING ────────────────────────────────

def rank_regions(regions, image_path):
    """
    Rank detected regions by suspicion level.

    For each region computes:
        - Normalized area
        - Irregularity index (1 - circularity)
        - Edge strength (Canny edge density within bounding box)

    Rank score formula:
        rank_score = 0.4 * normalized_area + 0.3 * irregularity + 0.3 * edge_density

    Labels:
        - Most Suspicious  (rank_score >= 0.6)
        - Moderate Concern  (rank_score >= 0.3)
        - Low Concern       (rank_score < 0.3)

    Returns:
        dict with region_ranking list.
    """
    if not regions:
        return {'region_ranking': []}

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    edges = cv2.Canny(gray, 50, 150)
    image_area = gray.shape[0] * gray.shape[1]

    # Find max area for normalization
    areas = [r.get('area', 0) for r in regions]
    max_area = max(areas) if areas else 1

    ranked = []
    for r in regions:
        region_id = r.get('region_id', 0)
        area = r.get('area', 0)

        # Normalized area (relative to largest region, capped at 1.0)
        normalized_area = min(1.0, area / (max_area + 1e-6))

        # Irregularity index: 1 - circularity (more irregular = higher)
        circularity = r.get('circularity', 1.0)
        irregularity = 1.0 - min(1.0, circularity)

        # Edge density within the bounding box
        bb = r.get('bounding_box', {})
        x = bb.get('x', 0)
        y = bb.get('y', 0)
        w = bb.get('w', 1)
        h = bb.get('h', 1)

        roi_edges = edges[y:y+h, x:x+w]
        roi_area = w * h
        if roi_area > 0 and roi_edges.size > 0:
            edge_density = float(np.sum(roi_edges > 0)) / roi_area
            edge_density = min(1.0, edge_density * 3.0)  # scale up
        else:
            edge_density = 0.0

        # Weighted rank score
        rank_score = (
            0.4 * normalized_area +
            0.3 * irregularity +
            0.3 * edge_density
        )
        rank_score = round(float(min(1.0, max(0.0, rank_score))), 4)

        # Label
        if rank_score >= 0.6:
            label = "Most Suspicious"
        elif rank_score >= 0.3:
            label = "Moderate Concern"
        else:
            label = "Low Concern"

        ranked.append({
            'region_id': int(region_id),
            'area': int(area),
            'rank_score': rank_score,
            'label': label,
            'details': {
                'normalized_area': round(float(normalized_area), 4),
                'irregularity': round(float(irregularity), 4),
                'edge_density': round(float(edge_density), 4)
            }
        })

    # Sort by rank_score descending (most suspicious first)
    ranked.sort(key=lambda x: x['rank_score'], reverse=True)

    return {'region_ranking': ranked}


# ─── 4. EXPLAINABLE AI INSIGHT GENERATOR ────────────────────────

def generate_ai_insight(region_count, regions, tumor_area_percentage,
                        risk_level, confidence_score, quality_label,
                        image_quality_score):
    """
    Generate a natural language summary using all computed metrics.

    Includes:
        - Number of regions detected
        - Largest region percentage
        - Risk level
        - Confidence score
        - Image quality comment

    Returns:
        dict with ai_insight string.
    """
    if region_count == 0:
        insight = (
            f"No significant tumor regions detected in this scan. "
            f"Image quality rated {quality_label} (score: {image_quality_score}). "
            f"Overall risk classified as {risk_level} with "
            f"{round(confidence_score * 100, 1)}% confidence. "
            f"Routine follow-up recommended as per standard protocol."
        )
        return {'ai_insight': insight}

    # Find the largest region's area percentage
    if regions:
        total_image_area = 0
        max_region_area = 0
        for r in regions:
            area = r.get('area', 0)
            if area > max_region_area:
                max_region_area = area
            area_ratio = r.get('area_ratio', 0)
            if area_ratio > 0 and area > 0:
                total_image_area = area / area_ratio

        if total_image_area > 0:
            largest_pct = round((max_region_area / total_image_area) * 100, 1)
        else:
            largest_pct = round(tumor_area_percentage, 1)
    else:
        largest_pct = 0.0

    # Region descriptor
    if region_count == 1:
        region_phrase = "One region"
    elif region_count == 2:
        region_phrase = "Two regions"
    elif region_count == 3:
        region_phrase = "Three regions"
    else:
        region_phrase = f"{region_count} regions"

    # Check irregularity across regions
    avg_circularity = np.mean([r.get('circularity', 1.0) for r in regions]) if regions else 1.0
    if avg_circularity < 0.5:
        shape_desc = "irregular"
    elif avg_circularity < 0.75:
        shape_desc = "moderately irregular"
    else:
        shape_desc = "relatively regular"

    # Build the insight
    insight = (
        f"{region_phrase} detected with {shape_desc} morphology. "
        f"Largest region occupies {largest_pct}% of scan area. "
        f"Image quality rated {quality_label} (score: {image_quality_score}). "
        f"Overall risk classified as {risk_level} with "
        f"{round(confidence_score * 100, 1)}% confidence."
    )

    # Append recommendation based on risk
    if risk_level == 'High':
        insight += " Immediate clinical review is strongly recommended."
    elif risk_level == 'Medium':
        insight += " Further diagnostic evaluation is advised."
    else:
        insight += " Routine monitoring suggested per clinical guidelines."

    return {'ai_insight': insight}
