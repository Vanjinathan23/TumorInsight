"""
visualization_engine.py
Visualization Engine for Medical Image Analysis.
Generates heatmaps, contour overlays, and split-view comparisons.
"""

import cv2
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os
import uuid


def generate_heatmap(mask_path, output_dir='static/uploads'):
    """
    Generate a heatmap visualization from a segmentation mask.

    Args:
        mask_path: Path to the binary mask image.
        output_dir: Directory to save the heatmap.

    Returns:
        Path to the saved heatmap image.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    blurred = cv2.GaussianBlur(mask, (21, 21), 0)

    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    heatmap_path = os.path.join(output_dir, f"heatmap_{uid}.png")
    cv2.imwrite(heatmap_path, heatmap)

    return heatmap_path


def generate_heatmap_overlay(image_path, mask_path, output_dir='static/uploads', alpha=0.5):
    """
    Generate a heatmap overlaid on the original image.

    Args:
        image_path: Path to the original image.
        mask_path: Path to the binary mask.
        output_dir: Directory to save the result.
        alpha: Blending factor.

    Returns:
        Path to the saved overlay image.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("Could not read image or mask")

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    blurred = cv2.GaussianBlur(mask_resized, (21, 21), 0)
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    overlay_path = os.path.join(output_dir, f"heatmap_overlay_{uid}.png")
    cv2.imwrite(overlay_path, overlay)

    return overlay_path


def generate_contour_overlay(image_path, mask_path, output_dir='static/uploads'):
    """
    Generate contour overlay visualization on the original image.

    Args:
        image_path: Path to the original image.
        mask_path: Path to the binary mask.
        output_dir: Directory to save the result.

    Returns:
        Path to the saved contour overlay image.
    """
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        raise ValueError("Could not read image or mask")

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    contours, _ = cv2.findContours(
        mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        area = cv2.contourArea(contour)
        label = f"Area: {area}px"
        cv2.putText(overlay, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    contour_path = os.path.join(output_dir, f"contour_overlay_{uid}.png")
    cv2.imwrite(contour_path, overlay)

    return contour_path


def generate_split_view(original_path, enhanced_path, output_dir='static/uploads'):
    """
    Generate a side-by-side split view of original vs enhanced images.

    Args:
        original_path: Path to the original image.
        enhanced_path: Path to the enhanced image.
        output_dir: Directory to save the result.

    Returns:
        Path to the saved split-view image.
    """
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)

    if original is None or enhanced is None:
        raise ValueError("Could not read one or both images")

    h = max(original.shape[0], enhanced.shape[0])
    w1 = original.shape[1]
    w2 = enhanced.shape[1]

    original_resized = cv2.resize(original, (w1, h))
    enhanced_resized = cv2.resize(enhanced, (w2, h))

    divider_width = 4
    divider = np.ones((h, divider_width, 3), dtype=np.uint8) * 200

    split_view = np.hstack([original_resized, divider, enhanced_resized])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(split_view, "Original", (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(split_view, "Enhanced", (w1 + divider_width + 10, 30), font, 0.8, (255, 255, 255), 2)

    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    split_path = os.path.join(output_dir, f"split_view_{uid}.png")
    cv2.imwrite(split_path, split_view)

    return split_path


def generate_plotly_heatmap(mask_path, output_dir='static/uploads'):
    """
    Generate an interactive Plotly heatmap as HTML.

    Args:
        mask_path: Path to the binary mask.
        output_dir: Directory to save the HTML file.

    Returns:
        Path to the saved HTML file.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    blurred = cv2.GaussianBlur(mask, (21, 21), 0).astype(float)

    fig = go.Figure(data=go.Heatmap(
        z=np.flipud(blurred),
        colorscale='Hot',
        showscale=True,
        colorbar=dict(title='Intensity')
    ))

    fig.update_layout(
        title='Tumor Region Heatmap',
        xaxis_title='X',
        yaxis_title='Y',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=700,
        height=500,
        margin=dict(l=50, r=50, t=60, b=50)
    )

    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    html_path = os.path.join(output_dir, f"plotly_heatmap_{uid}.html")
    fig.write_html(html_path, include_plotlyjs='cdn')

    return html_path


def generate_all_visualizations(original_path, enhanced_path, mask_path,
                                output_dir='static/uploads'):
    """
    Generate all visualization outputs.

    Returns:
        dict with paths to all generated visualizations.
    """
    heatmap = generate_heatmap(mask_path, output_dir)
    heatmap_overlay = generate_heatmap_overlay(original_path, mask_path, output_dir)
    contour_overlay = generate_contour_overlay(original_path, mask_path, output_dir)
    split_view = generate_split_view(original_path, enhanced_path, output_dir)
    plotly_heatmap = generate_plotly_heatmap(mask_path, output_dir)

    return {
        'heatmap_path': heatmap,
        'heatmap_overlay_path': heatmap_overlay,
        'contour_overlay_path': contour_overlay,
        'split_view_path': split_view,
        'plotly_heatmap_path': plotly_heatmap
    }
