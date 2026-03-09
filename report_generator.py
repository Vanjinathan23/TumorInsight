"""
report_generator.py
Medical Report Generator.
Generates structured medical reports in PDF, HTML, and JSON formats.
"""

import json
import os
import uuid
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from jinja2 import Template


SYSTEM_VERSION = "1.0.0"


def generate_pdf_report(data, output_dir='exports'):
    """
    Generate a professional PDF medical report.

    Args:
        data: dict containing scan results, images, metrics, risk scores, insights.
        output_dir: Directory to save the PDF.

    Returns:
        Path to the generated PDF.
    """
    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    pdf_path = os.path.join(output_dir, f"medical_report_{uid}.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=30, leftMargin=30,
        topMargin=30, bottomMargin=30
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='ReportTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=20,
        textColor=colors.HexColor('#e85d3b')
    ))
    styles.add(ParagraphStyle(
        name='SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#333333')
    ))
    styles.add(ParagraphStyle(
        name='BodyText2',
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
        leading=14
    ))
    styles.add(ParagraphStyle(
        name='MetaText',
        fontName='Helvetica',
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=10
    ))

    elements = []

    elements.append(Paragraph("Medical Image Analysis Report", styles['ReportTitle']))
    elements.append(Paragraph(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System v{SYSTEM_VERSION}",
        styles['MetaText']
    ))
    elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#e85d3b')))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Scan Overview", styles['SectionHeader']))
    summary = data.get('summary', {})
    overview_data = [
        ['Total Regions Detected', str(summary.get('total_regions', 0))],
        ['Image Dimensions', f"{summary.get('image_dimensions', {}).get('width', 'N/A')} x {summary.get('image_dimensions', {}).get('height', 'N/A')}"],
        ['Tumor Coverage', f"{summary.get('tumor_coverage_pct', 0)}%"],
        ['Total Tumor Area', f"{summary.get('total_tumor_area_px', 0):,} px"],
    ]
    overview_table = Table(overview_data, colWidths=[200, 300])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(overview_table)
    elements.append(Spacer(1, 12))

    def add_image_safe(path, label, width=3*inch):
        if path and os.path.exists(path):
            elements.append(Paragraph(label, styles['SectionHeader']))
            try:
                img = RLImage(path, width=width, height=width * 0.75)
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception:
                elements.append(Paragraph(f"[Image could not be loaded: {path}]", styles['BodyText2']))

    add_image_safe(data.get('original_path'), "Original Scan")
    add_image_safe(data.get('enhanced_path'), "Enhanced Image")
    add_image_safe(data.get('mask_path'), "Segmentation Mask")
    add_image_safe(data.get('overlay_path'), "Detection Overlay")

    elements.append(Paragraph("Risk Assessment", styles['SectionHeader']))
    risk = data.get('risk', {})
    risk_data_table = [
        ['Risk Score', str(risk.get('risk_score', 'N/A'))],
        ['Risk Level', str(risk.get('risk_level', 'N/A'))],
        ['Confidence Score', f"{risk.get('confidence_score', 0) * 100:.1f}%"],
        ['Severity Index', str(risk.get('severity_index', 'N/A'))],
    ]
    risk_table = Table(risk_data_table, colWidths=[200, 300])
    risk_level_color = colors.HexColor('#27ae60')
    if risk.get('risk_level') == 'High':
        risk_level_color = colors.HexColor('#e74c3c')
    elif risk.get('risk_level') == 'Medium':
        risk_level_color = colors.HexColor('#f39c12')

    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('TEXTCOLOR', (1, 1), (1, 1), risk_level_color),
        ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold'),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 12))

    components = risk.get('component_scores', {})
    if components:
        elements.append(Paragraph("Risk Component Breakdown", styles['SectionHeader']))
        comp_data = [
            ['Component', 'Score'],
            ['Normalized Area', str(components.get('normalized_area', 'N/A'))],
            ['Irregularity', str(components.get('irregularity', 'N/A'))],
            ['Edge Density', str(components.get('edge_density', 'N/A'))],
            ['Texture Variation', str(components.get('texture_variation', 'N/A'))],
        ]
        comp_table = Table(comp_data, colWidths=[200, 300])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e85d3b')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f5f5f5')),
        ]))
        elements.append(comp_table)
        elements.append(Spacer(1, 12))

    insights = data.get('insights', '')
    if insights:
        elements.append(Paragraph("AI-Generated Insights", styles['SectionHeader']))
        elements.append(Paragraph(insights, styles['BodyText2']))
        elements.append(Spacer(1, 12))

    enhancement = data.get('enhancement_metrics', {})
    if enhancement:
        elements.append(Paragraph("Enhancement Metrics", styles['SectionHeader']))
        enh_data = [
            ['Metric', 'Value'],
            ['Noise Reduction', f"{enhancement.get('noise_reduction_pct', 0)}%"],
            ['Contrast Improvement', f"{enhancement.get('contrast_improvement_pct', 0)}%"],
            ['Quality Score', f"{enhancement.get('quality_score', 0)}"],
            ['Filters Applied', ', '.join(enhancement.get('filters_applied', []))],
        ]
        enh_table = Table(enh_data, colWidths=[200, 300])
        enh_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f5f5f5')),
        ]))
        elements.append(enh_table)
        elements.append(Spacer(1, 12))

    elements.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        f"This report was auto-generated by the Adaptive AI Medical Image Analysis System v{SYSTEM_VERSION}. "
        "It is intended for clinical decision support and should not replace professional medical judgment.",
        styles['MetaText']
    ))

    doc.build(elements)
    return pdf_path


def generate_html_report(data, output_dir='exports'):
    """
    Generate an HTML medical report.

    Args:
        data: dict containing scan results.
        output_dir: Directory to save the HTML.

    Returns:
        Path to the generated HTML.
    """
    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    html_path = os.path.join(output_dir, f"medical_report_{uid}.html")

    template_str = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Image Analysis Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            padding: 40px;
        }
        .report-container {
            max-width: 900px;
            margin: 0 auto;
            background: rgba(30, 30, 50, 0.95);
            border-radius: 16px;
            padding: 40px;
            border: 1px solid rgba(232, 93, 59, 0.3);
        }
        .report-header {
            text-align: center;
            border-bottom: 3px solid #e85d3b;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .report-header h1 {
            color: #e85d3b;
            font-size: 28px;
            margin-bottom: 8px;
        }
        .report-header .meta {
            color: #888;
            font-size: 12px;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #e85d3b;
            font-size: 18px;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 1px solid rgba(232, 93, 59, 0.2);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
        .metric-item {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px 16px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .metric-item .label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }
        .metric-item .value {
            font-size: 20px;
            font-weight: 700;
            color: #fff;
            margin-top: 4px;
        }
        .risk-badge {
            display: inline-block;
            padding: 4px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 14px;
        }
        .risk-low { background: rgba(39,174,96,0.2); color: #27ae60; }
        .risk-medium { background: rgba(243,156,18,0.2); color: #f39c12; }
        .risk-high { background: rgba(231,76,60,0.2); color: #e74c3c; }
        .insight-box {
            background: rgba(232,93,59,0.08);
            border: 1px solid rgba(232,93,59,0.2);
            border-radius: 8px;
            padding: 16px;
            line-height: 1.6;
            font-size: 14px;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 16px;
        }
        .image-card {
            text-align: center;
        }
        .image-card img {
            max-width: 100%;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .image-card .caption {
            margin-top: 6px;
            font-size: 12px;
            color: #888;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding-top: 16px;
            border-top: 1px solid rgba(255,255,255,0.1);
            font-size: 11px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="report-container">
        <div class="report-header">
            <h1>Medical Image Analysis Report</h1>
            <p class="meta">Generated: {{ timestamp }} | System v{{ version }}</p>
        </div>

        <div class="section">
            <h2>Scan Overview</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="label">Regions Detected</div>
                    <div class="value">{{ summary.total_regions }}</div>
                </div>
                <div class="metric-item">
                    <div class="label">Tumor Coverage</div>
                    <div class="value">{{ summary.tumor_coverage_pct }}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Image Dimensions</div>
                    <div class="value">{{ summary.image_dimensions.width }} × {{ summary.image_dimensions.height }}</div>
                </div>
                <div class="metric-item">
                    <div class="label">Total Tumor Area</div>
                    <div class="value">{{ summary.total_tumor_area_px }} px</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Risk Assessment</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="label">Risk Score</div>
                    <div class="value">{{ risk.risk_score }}</div>
                </div>
                <div class="metric-item">
                    <div class="label">Risk Level</div>
                    <div class="value">
                        <span class="risk-badge risk-{{ risk.risk_level|lower }}">{{ risk.risk_level }}</span>
                    </div>
                </div>
                <div class="metric-item">
                    <div class="label">Confidence</div>
                    <div class="value">{{ (risk.confidence_score * 100)|round(1) }}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Severity Index</div>
                    <div class="value">{{ risk.severity_index }}</div>
                </div>
            </div>
        </div>

        {% if insights %}
        <div class="section">
            <h2>AI-Generated Insights</h2>
            <div class="insight-box">{{ insights }}</div>
        </div>
        {% endif %}

        {% if enhancement_metrics %}
        <div class="section">
            <h2>Enhancement Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="label">Noise Reduction</div>
                    <div class="value">{{ enhancement_metrics.noise_reduction_pct }}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Contrast Improvement</div>
                    <div class="value">{{ enhancement_metrics.contrast_improvement_pct }}%</div>
                </div>
                <div class="metric-item">
                    <div class="label">Quality Score</div>
                    <div class="value">{{ enhancement_metrics.quality_score }}</div>
                </div>
                <div class="metric-item">
                    <div class="label">Filters Applied</div>
                    <div class="value">{{ enhancement_metrics.filters_applied|join(', ') }}</div>
                </div>
            </div>
        </div>
        {% endif %}

        <div class="footer">
            <p>Auto-generated by Adaptive AI Medical Image Analysis System v{{ version }}</p>
            <p>This report is for clinical decision support only.</p>
        </div>
    </div>
</body>
</html>"""

    template = Template(template_str)
    html_content = template.render(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        version=SYSTEM_VERSION,
        summary=data.get('summary', {}),
        risk=data.get('risk', {}),
        insights=data.get('insights', ''),
        enhancement_metrics=data.get('enhancement_metrics', {}),
        regions=data.get('regions', [])
    )

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return html_path


def generate_json_report(data, output_dir='exports'):
    """
    Generate a structured JSON report.

    Args:
        data: dict containing scan results.
        output_dir: Directory to save the JSON.

    Returns:
        Path to the generated JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    json_path = os.path.join(output_dir, f"medical_report_{uid}.json")

    report = {
        'report_metadata': {
            'generated_at': datetime.now().isoformat(),
            'system_version': SYSTEM_VERSION,
            'report_type': 'medical_image_analysis'
        },
        'scan_summary': data.get('summary', {}),
        'regions': data.get('regions', []),
        'risk_assessment': data.get('risk', {}),
        'insights': data.get('insights', ''),
        'enhancement_metrics': data.get('enhancement_metrics', {}),
        'image_paths': {
            'original': data.get('original_path', ''),
            'enhanced': data.get('enhanced_path', ''),
            'mask': data.get('mask_path', ''),
            'overlay': data.get('overlay_path', '')
        }
    }

    def convert_numpy(obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    report = convert_numpy(report)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    return json_path


def generate_report(data, report_format='pdf', output_dir='exports'):
    """
    Generate a report in the specified format.

    Args:
        data: dict containing all analysis data.
        report_format: 'pdf', 'html', or 'json'.
        output_dir: Directory to save the report.

    Returns:
        Path to the generated report.
    """
    if report_format == 'pdf':
        return generate_pdf_report(data, output_dir)
    elif report_format == 'html':
        return generate_html_report(data, output_dir)
    elif report_format == 'json':
        return generate_json_report(data, output_dir)
    else:
        raise ValueError(f"Unsupported report format: {report_format}")
