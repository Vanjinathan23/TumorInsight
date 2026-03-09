"""
risk_engine.py
Tumor Risk Scoring Engine.
Computes weighted risk scores based on area, irregularity,
edge density, and texture variation. Generates natural language insights.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os


def compute_normalized_area(regions, image_area):
    """Compute normalized area weight from detected regions."""
    if not regions:
        return 0.0
    total_area = sum(r['area'] for r in regions)
    ratio = total_area / (image_area + 1e-6)
    normalized = min(1.0, ratio * 20)
    return float(normalized)


def compute_irregularity(regions):
    """Compute irregularity score from circularity and compactness."""
    if not regions:
        return 0.0
    irregularities = []
    for r in regions:
        circularity = r.get('circularity', 1.0)
        compactness = r.get('compactness', 1.0)
        irregularity = (1.0 - circularity) * 0.6 + min(1.0, compactness / 5.0) * 0.4
        irregularities.append(irregularity)
    return float(np.mean(irregularities))


def compute_edge_density(image_path, regions):
    """Compute edge density within detected regions."""
    image = cv2.imread(image_path)
    if image is None:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)

    total_edge_pixels = 0
    total_area = 0

    for r in regions:
        bb = r['bounding_box']
        x, y, w, h = bb['x'], bb['y'], bb['w'], bb['h']
        roi = edges[y:y+h, x:x+w]
        total_edge_pixels += np.sum(roi > 0)
        total_area += w * h

    if total_area == 0:
        return 0.0

    density = total_edge_pixels / (total_area + 1e-6)
    normalized = min(1.0, density * 3)
    return float(normalized)


def compute_texture_variation(image_path, regions):
    """Compute texture variation using GLCM properties."""
    image = cv2.imread(image_path)
    if image is None:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    if not regions:
        return 0.0

    variations = []
    for r in regions:
        bb = r['bounding_box']
        x, y, w, h = bb['x'], bb['y'], bb['w'], bb['h']
        roi = gray[y:y+h, x:x+w]

        if roi.size == 0 or roi.shape[0] < 2 or roi.shape[1] < 2:
            continue

        roi_quantized = (roi // 16).astype(np.uint8)

        try:
            glcm = graycomatrix(roi_quantized, distances=[1], angles=[0],
                                levels=16, symmetric=True, normed=True)
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

            variation = (contrast / 50.0) * 0.6 + (1.0 - homogeneity) * 0.4
            variation = min(1.0, variation)
            variations.append(variation)
        except Exception:
            std_val = np.std(roi) / 128.0
            variations.append(min(1.0, float(std_val)))

    if not variations:
        return 0.0

    return float(np.mean(variations))


def compute_risk_score(image_path, regions, image_area):
    """
    Compute overall tumor risk score using weighted formula.

    Weights:
        - Area:          0.35
        - Irregularity:  0.25
        - Edge density:  0.25
        - Texture:       0.15

    Returns:
        dict with risk_score, risk_level, confidence_score,
        severity_index, component_scores.
    """
    area_weight = compute_normalized_area(regions, image_area)
    irregularity_weight = compute_irregularity(regions)
    edge_density_weight = compute_edge_density(image_path, regions)
    texture_weight = compute_texture_variation(image_path, regions)

    risk_score = (
        area_weight * 0.35 +
        irregularity_weight * 0.25 +
        edge_density_weight * 0.25 +
        texture_weight * 0.15
    )

    risk_score = round(float(min(1.0, max(0.0, risk_score))), 4)

    if risk_score < 0.35:
        risk_level = 'Low'
    elif risk_score < 0.65:
        risk_level = 'Medium'
    else:
        risk_level = 'High'

    base_confidence = 0.5
    region_count_boost = min(0.2, len(regions) * 0.05)
    area_boost = min(0.15, area_weight * 0.15)
    texture_boost = min(0.15, texture_weight * 0.15)
    confidence_score = min(0.99, base_confidence + region_count_boost + area_boost + texture_boost)
    confidence_score = round(float(confidence_score), 4)

    severity_components = []
    for r in regions:
        region_severity = (
            r.get('area_ratio', 0) * 100 * 0.4 +
            (1 - r.get('circularity', 1)) * 0.3 +
            r.get('compactness', 1) / 5.0 * 0.3
        )
        severity_components.append(round(float(min(1.0, region_severity)), 4))

    severity_index = round(float(np.mean(severity_components)), 4) if severity_components else 0.0

    return {
        'risk_score': risk_score,
        'risk_level': risk_level,
        'confidence_score': confidence_score,
        'severity_index': severity_index,
        'component_scores': {
            'normalized_area': round(area_weight, 4),
            'irregularity': round(irregularity_weight, 4),
            'edge_density': round(edge_density_weight, 4),
            'texture_variation': round(texture_weight, 4)
        },
        'region_severities': severity_components,
        'total_regions': len(regions)
    }


def generate_insights(risk_data, regions):
    """Generate natural language summary from risk analysis."""
    n_regions = risk_data['total_regions']
    risk_level = risk_data['risk_level']
    confidence = risk_data['confidence_score']
    risk_score = risk_data['risk_score']
    components = risk_data['component_scores']

    if n_regions == 0:
        return (
            "No significant regions of interest were detected in this scan. "
            "The image appears within normal parameters. "
            "Recommend routine follow-up as per standard protocol."
        )

    region_word = "region" if n_regions == 1 else "regions"

    irregularity_desc = "regular"
    if components['irregularity'] > 0.6:
        irregularity_desc = "highly irregular"
    elif components['irregularity'] > 0.3:
        irregularity_desc = "moderately irregular"
    elif components['irregularity'] > 0.15:
        irregularity_desc = "slightly irregular"

    boundary_desc = "smooth boundaries"
    if components['edge_density'] > 0.6:
        boundary_desc = "significant boundary distortion"
    elif components['edge_density'] > 0.3:
        boundary_desc = "moderate boundary distortion"
    elif components['edge_density'] > 0.15:
        boundary_desc = "mild boundary variation"

    texture_desc = ""
    if components['texture_variation'] > 0.5:
        texture_desc = " Texture analysis reveals notable heterogeneity within the detected regions."
    elif components['texture_variation'] > 0.25:
        texture_desc = " Texture patterns show moderate variation within the regions."

    total_area = sum(r['area'] for r in regions)

    first_region = regions[0] if regions else {}
    tumor_type_str = first_region.get('tumor_type', 'unspecified tumor pattern')

    insight = (
        f"Diagnostic analysis identified {n_regions} {irregularity_desc} {region_word} consistent with **{tumor_type_str}**. "
        f"Boundary analysis indicates {boundary_desc}. "
        f"Total affected area: {total_area:,} pixels. "
        f"Estimated risk level is {risk_level} with {confidence*100:.0f}% confidence "
        f"(risk score: {risk_score:.2f}).{texture_desc} "
    )

    if risk_level == 'High':
        insight += "Immediate clinical review is strongly recommended."
    elif risk_level == 'Medium':
        insight += "Further diagnostic evaluation is recommended."
    else:
        insight += "Routine monitoring is suggested as per clinical guidelines."

    return insight
