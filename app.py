"""
app.py
Main Flask Application Server.
Provides all API endpoints and dashboard routes for the
Adaptive AI-Based Medical Image Enhancement and Tumor Risk Analysis System.
"""

import os
import uuid
import json
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, send_file,
    redirect, url_for, session
)
import numpy as np
import cv2

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from image_enhancer import enhance_image
from ai_analysis import analyze_image
from risk_engine import compute_risk_score, generate_insights
from visualization_engine import generate_all_visualizations
from report_generator import generate_report
from advanced_analytics import (
    assess_image_quality,
    compute_tumor_area_percentage,
    rank_regions,
    generate_ai_insight
)

app = Flask(__name__)
app.secret_key = 'medical-imaging-secret-key-2024'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
EXPORT_FOLDER = 'exports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'dicom'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

scan_store = {}


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = NumpyEncoder


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_dicom_to_png(dicom_path, output_dir):
    """Convert a DICOM file to PNG format."""
    if not HAS_PYDICOM:
        raise ImportError("pydicom is required for DICOM file support")

    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(float)

    pixel_array = ((pixel_array - pixel_array.min()) /
                   (pixel_array.max() - pixel_array.min() + 1e-6) * 255)
    pixel_array = pixel_array.astype(np.uint8)

    uid = uuid.uuid4().hex[:8]
    png_path = os.path.join(output_dir, f"dicom_converted_{uid}.png")
    cv2.imwrite(png_path, pixel_array)
    return png_path


def get_scan_stats():
    """Get aggregate statistics from all scans."""
    total = len(scan_store)
    risk_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Pending': 0}
    
    for s in scan_store.values():
        lvl = s.get('risk', {}).get('risk_level', 'Pending')
        if lvl in risk_counts:
            risk_counts[lvl] += 1
        else:
            risk_counts['Pending'] += 1

    risk_alerts = risk_counts['Medium'] + risk_counts['High']
    
    quality_scores = [
        s.get('enhancement_metrics', {}).get('quality_score', 0)
        for s in scan_store.values()
        if s.get('enhancement_metrics')
    ]
    avg_quality = round(np.mean(quality_scores), 1) if quality_scores else 0

    return {
        'total_scans': total,
        'risk_alerts': risk_alerts,
        'avg_quality': float(avg_quality),
        'risk_distribution': risk_counts
    }

# ─── Dashboard Routes ───────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('intro_page'))


@app.route('/intro')
def intro_page():
    return render_template('intro.html')


@app.route('/dashboard/overview')
def overview():
    stats = get_scan_stats()
    return render_template('dashboards/overview.html',
                           stats=stats, scans=scan_store)


@app.route('/dashboard/upload')
def upload_page():
    return render_template('dashboards/upload.html')


@app.route('/dashboard/detection')
def detection_page():
    scan_id = request.args.get('scan_id', '')
    scan = scan_store.get(scan_id, {})
    return render_template('dashboards/detection.html',
                           scan=scan, scan_id=scan_id)


@app.route('/dashboard/risk')
def risk_page():
    scan_id = request.args.get('scan_id', '')
    scan = scan_store.get(scan_id, {})
    return render_template('dashboards/risk.html',
                           scan=scan, scan_id=scan_id)


@app.route('/dashboard/comparison')
def comparison_page():
    scan_id = request.args.get('scan_id', '')
    scan = scan_store.get(scan_id, {})
    return render_template('dashboards/comparison.html',
                           scan=scan, scan_id=scan_id)


@app.route('/dashboard/export')
def export_page():
    scan_id = request.args.get('scan_id', '')
    scan = scan_store.get(scan_id, {})
    return render_template('dashboards/export.html',
                           scan=scan, scan_id=scan_id)


# ─── API Endpoints ──────────────────────────────────────────────

def is_brain_scan(image_path):
    """
    Heuristic to check if the uploaded image is likely a brain scan (MRI/CT).
    Checks for:
    1. Grayscale-like intensity distribution (low saturation if RGB).
    2. Dark background (high percentage of low-intensity pixels).
    3. Centralized structure (standard deviation of intensities).
    """
    image = cv2.imread(image_path)
    if image is None:
        return False

    # Check color saturation (brain scans are usually grayscale)
    if len(image.shape) == 3:
        # Convert to HSV and check saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        if avg_saturation > 50: # Arbitrary threshold for "too colorful"
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Check for dark background
    # Legitimate brain scans usually have > 40% black background
    black_pixels = np.sum(gray < 30)
    total_pixels = gray.size
    black_ratio = black_pixels / total_pixels
    
    if black_ratio < 0.2: # Too much light / busy for a typical MRI
        return False

    # Check for centralized structure (Standard deviation of pixel intensities)
    # Extremely low std dev means a flat/random image.
    std_dev = np.std(gray)
    if std_dev < 15: # Too flat/uniform
        return False

    return True


@app.route('/upload', methods=['POST'])
def upload():
    """Handle image upload (DICOM, PNG, JPG)."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Use PNG, JPG, or DICOM.'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    uid = uuid.uuid4().hex[:8]
    filename = f"upload_{uid}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    if ext in ('dcm', 'dicom'):
        try:
            filepath = convert_dicom_to_png(filepath, UPLOAD_FOLDER)
        except Exception as e:
            return jsonify({'error': f'DICOM conversion failed: {str(e)}'}), 500

    # Validate if it's a brain scan
    if not is_brain_scan(filepath):
        # Remove the invalid file
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({
            'error': 'Scan not found',
            'message': 'Upload only supporting images'
        }), 400

    # Compute initial image quality
    try:
        quality_res = assess_image_quality(filepath)
    except Exception as e:
        quality_res = {'image_quality_score': 0, 'quality_label': 'Unknown', 'error': str(e)}

    scan_id = uid
    scan_store[scan_id] = {
        'scan_id': scan_id,
        'original_path': filepath,
        'filename': file.filename,
        'uploaded_at': datetime.now().isoformat(),
        'status': 'uploaded',
        'image_quality_score': quality_res['image_quality_score'],
        'quality_label': quality_res['quality_label']
    }

    return jsonify({
        'success': True,
        'scan_id': scan_id,
        'filename': file.filename,
        'image_url': '/' + filepath.replace('\\', '/'),
        'image_quality_score': quality_res['image_quality_score'],
        'quality_label': quality_res['quality_label'],
        'message': 'Image uploaded successfully'
    })


@app.route('/enhance', methods=['POST'])
def enhance():
    """Run adaptive image enhancement pipeline."""
    data = request.get_json() or {}
    scan_id = data.get('scan_id', '')

    if scan_id not in scan_store:
        return jsonify({'error': 'Scan not found'}), 404

    scan = scan_store[scan_id]
    image_path = scan['original_path']

    use_clahe = data.get('use_clahe', True)
    use_gaussian = data.get('use_gaussian', False)
    use_median = data.get('use_median', False)
    auto_mode = data.get('auto_mode', True)

    try:
        result = enhance_image(
            image_path,
            output_dir=UPLOAD_FOLDER,
            use_clahe=use_clahe,
            use_gaussian=use_gaussian,
            use_median=use_median,
            auto_mode=auto_mode
        )

        scan_store[scan_id]['enhanced_path'] = result['enhanced_path']
        scan_store[scan_id]['enhancement_metrics'] = result['metrics']
        scan_store[scan_id]['status'] = 'enhanced'

        response = {
            'success': True,
            'scan_id': scan_id,
            'enhanced_url': '/' + result['enhanced_path'].replace('\\', '/'),
            'metrics': result['metrics'],
            'message': 'Image enhanced successfully'
        }
        return json.dumps(response, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({'error': f'Enhancement failed: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Run tumor detection and segmentation."""
    data = request.get_json() or {}
    scan_id = data.get('scan_id', '')

    if scan_id not in scan_store:
        return jsonify({'error': 'Scan not found'}), 404

    scan = scan_store[scan_id]
    image_path = scan.get('enhanced_path', scan['original_path'])
    min_area = data.get('min_area', 500)

    try:
        result = analyze_image(
            image_path,
            output_dir=UPLOAD_FOLDER,
            min_area=min_area
        )

        scan_store[scan_id]['regions'] = result['regions']
        scan_store[scan_id]['summary'] = result['summary']
        scan_store[scan_id]['mask_path'] = result['mask_path']
        scan_store[scan_id]['overlay_path'] = result['overlay_path']
        scan_store[scan_id]['edge_path'] = result['edge_path']
        
        # New advanced analytics
        image_area = result['summary']['image_area_px']
        area_perc_res = compute_tumor_area_percentage(result['regions'], image_area)
        ranking_res = rank_regions(result['regions'], image_path)
        
        scan_store[scan_id]['tumor_area_percentage'] = area_perc_res['tumor_area_percentage']
        scan_store[scan_id]['region_ranking'] = ranking_res['region_ranking']
        scan_store[scan_id]['status'] = 'analyzed'

        response = {
            'success': True,
            'scan_id': scan_id,
            'regions': result['regions'],
            'summary': result['summary'],
            'tumor_area_percentage': area_perc_res['tumor_area_percentage'],
            'region_ranking': ranking_res['region_ranking'],
            'overlay_url': '/' + result['overlay_path'].replace('\\', '/'),
            'mask_url': '/' + result['mask_path'].replace('\\', '/'),
            'edge_url': '/' + result['edge_path'].replace('\\', '/'),
            'message': 'Analysis complete'
        }
        return json.dumps(response, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/risk', methods=['POST'])
def risk():
    """Compute tumor risk score."""
    data = request.get_json() or {}
    scan_id = data.get('scan_id', '')

    if scan_id not in scan_store:
        return jsonify({'error': 'Scan not found'}), 404

    scan = scan_store[scan_id]
    regions = scan.get('regions', [])
    image_path = scan.get('enhanced_path', scan['original_path'])
    image_area = scan.get('summary', {}).get('image_area_px', 1)

    try:
        risk_result = compute_risk_score(image_path, regions, image_area)
        insights = generate_insights(risk_result, regions)

        # New advanced analytics: Generate AI Insight
        ai_insight_res = generate_ai_insight(
            len(regions),
            regions,
            scan.get('tumor_area_percentage', 0),
            risk_result['risk_level'],
            risk_result['confidence_score'],
            scan.get('quality_label', 'Unknown'),
            scan.get('image_quality_score', 0)
        )

        scan_store[scan_id]['risk'] = risk_result
        scan_store[scan_id]['insights'] = insights  # existing
        scan_store[scan_id]['ai_insight'] = ai_insight_res['ai_insight']
        scan_store[scan_id]['status'] = 'risk_computed'

        response = {
            'success': True,
            'scan_id': scan_id,
            'risk': risk_result,
            'insights': insights,
            'ai_insight': ai_insight_res['ai_insight'],
            'message': 'Risk analysis complete'
        }
        return json.dumps(response, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({'error': f'Risk computation failed: {str(e)}'}), 500


@app.route('/visualize', methods=['POST'])
def visualize():
    """Generate all visualizations."""
    data = request.get_json() or {}
    scan_id = data.get('scan_id', '')

    if scan_id not in scan_store:
        return jsonify({'error': 'Scan not found'}), 404

    scan = scan_store[scan_id]
    original_path = scan.get('original_path', '')
    enhanced_path = scan.get('enhanced_path', original_path)
    mask_path = scan.get('mask_path', '')

    if not mask_path:
        return jsonify({'error': 'No analysis mask found. Run analysis first.'}), 400

    try:
        viz_result = generate_all_visualizations(
            original_path, enhanced_path, mask_path,
            output_dir=UPLOAD_FOLDER
        )

        scan_store[scan_id]['visualizations'] = viz_result
        scan_store[scan_id]['status'] = 'visualized'

        response = {
            'success': True,
            'scan_id': scan_id,
            'heatmap_url': '/' + viz_result['heatmap_path'].replace('\\', '/'),
            'heatmap_overlay_url': '/' + viz_result['heatmap_overlay_path'].replace('\\', '/'),
            'contour_overlay_url': '/' + viz_result['contour_overlay_path'].replace('\\', '/'),
            'split_view_url': '/' + viz_result['split_view_path'].replace('\\', '/'),
            'plotly_heatmap_url': '/' + viz_result['plotly_heatmap_path'].replace('\\', '/'),
            'message': 'Visualizations generated'
        }
        return json.dumps(response, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({'error': f'Visualization failed: {str(e)}'}), 500


@app.route('/export', methods=['POST'])
def export():
    """Generate and download reports."""
    data = request.get_json() or {}
    scan_id = data.get('scan_id', '')
    report_format = data.get('format', 'pdf')

    if scan_id not in scan_store:
        return jsonify({'error': 'Scan not found'}), 404

    scan = scan_store[scan_id]

    try:
        report_path = generate_report(scan, report_format, EXPORT_FOLDER)

        mime_types = {
            'pdf': 'application/pdf',
            'html': 'text/html',
            'json': 'application/json'
        }

        return send_file(
            report_path,
            as_attachment=True,
            download_name=f"medical_report.{report_format}",
            mimetype=mime_types.get(report_format, 'application/octet-stream')
        )

    except Exception as e:
        return jsonify({'error': f'Export failed: {str(e)}'}), 500


@app.route('/api/scan/<scan_id>')
def get_scan(scan_id):
    """Get full scan data."""
    if scan_id not in scan_store:
        return jsonify({'error': 'Scan not found'}), 404

    scan = scan_store[scan_id]
    response = json.dumps({'success': True, 'scan': scan}, cls=NumpyEncoder)
    return response, 200, {'Content-Type': 'application/json'}


@app.route('/api/scans')
def get_all_scans():
    """Get list of all scans."""
    scans_list = []
    for sid, s in scan_store.items():
        scans_list.append({
            'scan_id': sid,
            'filename': s.get('filename', ''),
            'uploaded_at': s.get('uploaded_at', ''),
            'status': s.get('status', ''),
            'risk_level': s.get('risk', {}).get('risk_level', 'N/A')
        })
    return jsonify({'success': True, 'scans': scans_list})


@app.route('/api/stats')
def get_stats():
    """Get aggregate dashboard stats."""
    stats = get_scan_stats()
    return jsonify({'success': True, 'stats': stats})


@app.route('/api/run-full-pipeline', methods=['POST'])
def run_full_pipeline():
    """Run the complete analysis pipeline on a scan."""
    data = request.get_json() or {}
    scan_id = data.get('scan_id', '')

    if scan_id not in scan_store:
        return jsonify({'error': 'Scan not found'}), 404

    scan = scan_store[scan_id]
    start_time = datetime.now()
    results = {'scan_id': scan_id}

    try:
        # Step 1: Enhance
        enh = enhance_image(scan['original_path'], output_dir=UPLOAD_FOLDER, auto_mode=True)
        scan_store[scan_id]['enhanced_path'] = enh['enhanced_path']
        scan_store[scan_id]['enhancement_metrics'] = enh['metrics']

        # Step 2: Analyze
        ana = analyze_image(enh['enhanced_path'], output_dir=UPLOAD_FOLDER)
        scan_store[scan_id].update({
            'regions': ana['regions'],
            'summary': ana['summary'],
            'mask_path': ana['mask_path'],
            'overlay_path': ana['overlay_path'],
            'edge_path': ana['edge_path']
        })

        # Step 3: Risk
        risk_res = compute_risk_score(
            enh['enhanced_path'],
            ana['regions'],
            ana['summary']['image_area_px']
        )
        insights = generate_insights(risk_res, ana['regions'])
        scan_store[scan_id]['risk'] = risk_res
        scan_store[scan_id]['insights'] = insights

        # Step 4: Visualize
        viz = generate_all_visualizations(
            scan['original_path'],
            enh['enhanced_path'],
            ana['mask_path'],
            output_dir=UPLOAD_FOLDER
        )
        scan_store[scan_id]['visualizations'] = viz

        # Step 5: Advanced Analytics Integration
        # Image Quality
        quality_res = assess_image_quality(scan['original_path'])
        scan_store[scan_id].update({
            'image_quality_score': quality_res['image_quality_score'],
            'quality_label': quality_res['quality_label']
        })

        # Tumor Area Percentage & Region Ranking
        area_perc_res = compute_tumor_area_percentage(ana['regions'], ana['summary']['image_area_px'])
        ranking_res = rank_regions(ana['regions'], enh['enhanced_path'])
        
        scan_store[scan_id].update({
            'tumor_area_percentage': area_perc_res['tumor_area_percentage'],
            'region_ranking': ranking_res['region_ranking']
        })

        # Explainable AI Insight
        ai_insight_res = generate_ai_insight(
            ana['summary']['total_regions'],
            ana['regions'],
            area_perc_res['tumor_area_percentage'],
            risk_res['risk_level'],
            risk_res['confidence_score'],
            quality_res['quality_label'],
            quality_res['image_quality_score']
        )
        scan_store[scan_id]['ai_insight'] = ai_insight_res['ai_insight']

        scan_store[scan_id]['status'] = 'complete'

        results.update({
            'success': True,
            'message': 'Full pipeline completed',
            'confidence_score': risk_res['confidence_score'],
            'risk_level': risk_res['risk_level'],
            'region_count': ana['summary']['total_regions'],
            'tumor_percentage': area_perc_res['tumor_area_percentage'], # Backward compatibility if needed
            'tumor_area_percentage': area_perc_res['tumor_area_percentage'],
            'image_quality_score': quality_res['image_quality_score'],
            'quality_label': quality_res['quality_label'],
            'region_ranking': ranking_res['region_ranking'],
            'ai_insight': ai_insight_res['ai_insight'],
            'processing_time': round(float((datetime.now() - start_time).total_seconds()), 3),
            'enhanced_image': '/' + enh['enhanced_path'].replace('\\', '/'),
            'segmented_image': '/' + ana['mask_path'].replace('\\', '/'),
            'highlighted_image': '/' + ana['overlay_path'].replace('\\', '/'),
            'heatmap_image': '/' + viz['heatmap_overlay_path'].replace('\\', '/')
        })

        return json.dumps(results, cls=NumpyEncoder), 200, {'Content-Type': 'application/json'}

    except Exception as e:
        return jsonify({'error': f'Pipeline failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
