"""
ai_analysis.py
Tumor Detection & Segmentation Module.
Uses classical CV (Otsu, Canny, Contours) with morphological operations
to detect and segment tumor-like regions.
"""

import cv2
import numpy as np
from scipy import ndimage
import os
import uuid


def preprocess_for_detection(image):
    """Convert image to grayscale and apply preprocessing."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def otsu_threshold(blurred):
    """Apply Otsu's thresholding for automatic segmentation."""
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def morphological_cleanup(binary_mask, kernel_size=5):
    """Apply morphological operations to clean up segmentation mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.erode(cleaned, small_kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, small_kernel, iterations=1)

    return cleaned


def extract_contours(binary_mask, min_area=500):
    """Extract and filter contours based on minimum area."""
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]
    filtered.sort(key=cv2.contourArea, reverse=True)
    return filtered


def compute_region_stats(contour, image_shape):
    """Compute statistics for a detected region."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)

    M = cv2.moments(contour)
    if M['m00'] > 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = x + w // 2, y + h // 2

    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    image_area = image_shape[0] * image_shape[1]
    area_ratio = area / (image_area + 1e-6)

    aspect_ratio = float(w) / (h + 1e-6)

    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    return {
        'area': int(area),
        'perimeter': round(float(perimeter), 2),
        'bounding_box': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
        'centroid': {'x': int(cx), 'y': int(cy)},
        'circularity': round(float(circularity), 4),
        'solidity': round(float(solidity), 4),
        'area_ratio': round(float(area_ratio), 6),
        'aspect_ratio': round(float(aspect_ratio), 4),
        'compactness': round(float(compactness), 4)
    }


def classify_tumor(stats):
    """
    Classify tumor type based on morphological statistics.
    Note: In a production environment, this would use a trained CNN/ML model.
    """
    circularity = stats['circularity']
    solidity = stats['solidity']
    area_ratio = stats['area_ratio']

    if circularity > 0.75 and solidity > 0.9:
        tumor_type = "Meningioma"
        description = "Well-defined, typically benign tumor growing from the meninges. Smooth boundaries detected."
    elif area_ratio < 0.005 and circularity > 0.6:
        tumor_type = "Pituitary Tumor"
        description = "Small, localized growth consistent with pituitary adenoma characteristics."
    else:
        tumor_type = "Suspected Glioma"
        description = "Infiltrative growth pattern with irregular boundaries. High-priority for neurological review."

    return tumor_type, description


def generate_mask(image_shape, contours):
    """Generate a binary segmentation mask from contours."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    return mask


def generate_overlay(image, contours, regions):
    """Generate visualization overlay with contours and bounding boxes."""
    overlay = image.copy()

    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (0, 0, 255), -1)
    overlay = cv2.addWeighted(overlay, 0.7, mask, 0.3, 0)

    for i, contour in enumerate(contours):
        # Highlight the region with a glow effect
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 3)

        bb = regions[i]['bounding_box']
        cv2.rectangle(
            overlay,
            (bb['x'], bb['y']),
            (bb['x'] + bb['w'], bb['y'] + bb['h']),
            (0, 255, 0), 1
        )

        label = f"{regions[i]['tumor_type']}"
        cv2.putText(
            overlay, label,
            (bb['x'], bb['y'] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
        )

    return overlay


def apply_canny_edges(gray, low_threshold=50, high_threshold=150):
    """Apply Canny edge detection."""
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges


def analyze_image(image_path, output_dir='static/uploads',
                  min_area=500, confidence_threshold=0.3):
    """
    Full tumor detection and segmentation pipeline.

    Args:
        image_path: Path to the (enhanced) image.
        output_dir: Directory to save outputs.
        min_area: Minimum contour area to consider.
        confidence_threshold: Minimum confidence to report a region.

    Returns:
        dict with regions, mask_path, overlay_path, edge_path, statistics.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray, blurred = preprocess_for_detection(image)

    binary = otsu_threshold(blurred)

    cleaned = morphological_cleanup(binary)

    contours = extract_contours(cleaned, min_area=min_area)

    regions = []
    for i, contour in enumerate(contours):
        stats = compute_region_stats(contour, image.shape)
        confidence = min(1.0, stats['area_ratio'] * 50 + stats['circularity'] * 0.3 + stats['solidity'] * 0.2)
        confidence = round(float(confidence), 4)
        stats['confidence'] = confidence
        stats['region_id'] = i + 1

        if confidence >= confidence_threshold:
            t_type, t_desc = classify_tumor(stats)
            stats['tumor_type'] = t_type
            stats['explanation'] = t_desc
            regions.append(stats)

    valid_contours = [contours[i] for i in range(len(contours))
                      if i < len(regions)]

    mask = generate_mask(image.shape, valid_contours)
    overlay = generate_overlay(image, valid_contours, regions)
    edges = apply_canny_edges(gray)

    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]

    mask_path = os.path.join(output_dir, f"mask_{uid}.png")
    overlay_path = os.path.join(output_dir, f"overlay_{uid}.png")
    edge_path = os.path.join(output_dir, f"edges_{uid}.png")

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(edge_path, edges)

    total_tumor_area = sum(r['area'] for r in regions)
    image_area = image.shape[0] * image.shape[1]
    coverage = round(float(total_tumor_area / (image_area + 1e-6) * 100), 2)

    summary = {
        'total_regions': len(regions),
        'total_tumor_area_px': total_tumor_area,
        'image_area_px': image_area,
        'tumor_coverage_pct': coverage,
        'image_dimensions': {
            'width': image.shape[1],
            'height': image.shape[0]
        }
    }

    return {
        'regions': regions,
        'summary': summary,
        'mask_path': mask_path,
        'overlay_path': overlay_path,
        'edge_path': edge_path
    }
