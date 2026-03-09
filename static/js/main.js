/* ═══════════════════════════════════════════════════════════════
   TumorInsight — Main JavaScript
   ═══════════════════════════════════════════════════════════════ */

// ─── Theme Toggle ───────────────────────────────────────────
function initTheme() {
    const saved = localStorage.getItem('tumorinsight-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeIcon(saved);
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('tumorinsight-theme', next);
    updateThemeIcon(next);
}

function updateThemeIcon(theme) {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.innerHTML = theme === 'dark'
        ? '<i data-lucide="sun"></i>'
        : '<i data-lucide="moon"></i>';
    if (typeof lucide !== 'undefined') lucide.createIcons();
}

// ─── Arc Navigation ──────────────────────────────────────────
function initArcNav() {
    const toggle = document.getElementById('arcToggle');
    const menu = document.getElementById('arcMenu');
    if (!toggle || !menu) return;

    toggle.addEventListener('click', () => {
        toggle.classList.toggle('active');
        menu.classList.toggle('open');
    });

    document.addEventListener('click', (e) => {
        if (!e.target.closest('.arc-nav')) {
            toggle.classList.remove('active');
            menu.classList.remove('open');
        }
    });
}

// ─── Toast Notifications ────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const icons = { success: '✓', error: '✕', info: 'ℹ' };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ─── Loading Overlay ────────────────────────────────────────
function showLoading(text = 'Processing...', sub = 'Please wait while the AI analyzes your image') {
    const overlay = document.getElementById('loadingOverlay');
    const textEl = document.getElementById('loadingText');
    const subEl = document.getElementById('loadingSub');
    if (overlay) {
        overlay.classList.add('active');
        if (textEl) textEl.textContent = text;
        if (subEl) subEl.textContent = sub;
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.remove('active');
}

// ─── Drag & Drop Upload ─────────────────────────────────────
function initUploadZone() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');
    const preview = document.getElementById('filePreview');
    if (!zone || !input) return;

    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFileSelect(files[0]);
    });

    input.addEventListener('change', () => {
        if (input.files.length > 0) handleFileSelect(input.files[0]);
    });
}

let selectedFile = null;

function handleFileSelect(file) {
    const allowed = ['image/png', 'image/jpeg', 'image/jpg', 'application/dicom'];
    const ext = file.name.split('.').pop().toLowerCase();

    if (!allowed.includes(file.type) && !['dcm', 'dicom', 'png', 'jpg', 'jpeg'].includes(ext)) {
        showToast('Unsupported file format. Use PNG, JPG, or DICOM.', 'error');
        return;
    }

    selectedFile = file;

    const preview = document.getElementById('filePreview');
    const previewImg = document.getElementById('previewImage');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');

    if (preview) {
        preview.classList.add('active');

        if (['png', 'jpg', 'jpeg'].includes(ext) && previewImg) {
            const reader = new FileReader();
            reader.onload = (e) => { previewImg.src = e.target.result; };
            reader.readAsDataURL(file);
        } else if (previewImg) {
            previewImg.src = '';
            previewImg.alt = 'DICOM file';
        }

        if (fileName) fileName.textContent = file.name;
        if (fileSize) fileSize.textContent = formatFileSize(file.size);
    }

    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) uploadBtn.disabled = false;

    // Clear previous validation errors
    const errorContainer = document.getElementById('validationError');
    if (errorContainer) {
        errorContainer.style.display = 'none';
        errorContainer.classList.remove('active');
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ─── Upload File ────────────────────────────────────────────
async function uploadFile() {
    if (!selectedFile) {
        showToast('Please select a file first', 'error');
        return;
    }

    showLoading('Uploading Image...', 'Preparing your medical scan for analysis');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.success) {
            sessionStorage.setItem('currentScanId', data.scan_id);
            sessionStorage.setItem('originalUrl', data.image_url);
            showToast('Image uploaded successfully!', 'success');
            hideLoading();

            // Auto-run full pipeline
            await runFullPipeline(data.scan_id);
        } else {
            hideLoading();
            if (data.error === 'Scan not found') {
                const errorContainer = document.getElementById('validationError');
                const errorTitle = document.getElementById('errorTitle');
                const errorMessage = document.getElementById('errorMessage');

                if (errorContainer) {
                    if (errorTitle) errorTitle.textContent = data.error;
                    if (errorMessage) errorMessage.textContent = data.message || 'Upload only supporting images';
                    errorContainer.style.display = 'block';
                    errorContainer.classList.add('active');
                }
            } else {
                showToast(data.error || 'Upload failed', 'error');
            }
        }
    } catch (err) {
        hideLoading();
        showToast('Upload failed: ' + err.message, 'error');
    }
}

// ─── Full Pipeline ──────────────────────────────────────────
async function runFullPipeline(scanId) {
    showLoading('Running AI Pipeline...', 'Enhancing → Detecting → Analyzing → Visualizing');

    try {
        const res = await fetch('/api/run-full-pipeline', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scan_id: scanId })
        });
        const data = await res.json();

        if (data.success) {
            // Store results
            sessionStorage.setItem('pipelineResult', JSON.stringify(data));
            sessionStorage.setItem('enhancedUrl', data.enhanced_url);
            sessionStorage.setItem('overlayUrl', data.overlay_url);
            sessionStorage.setItem('maskUrl', data.mask_url);

            hideLoading();
            showToast('Full analysis pipeline complete!', 'success');

            // Navigate to detection page
            setTimeout(() => {
                window.location.href = `/dashboard/detection?scan_id=${scanId}`;
            }, 1000);
        } else {
            hideLoading();
            showToast(data.error || 'Pipeline failed', 'error');
        }
    } catch (err) {
        hideLoading();
        showToast('Pipeline error: ' + err.message, 'error');
    }
}

// ─── Individual Pipeline Steps ──────────────────────────────
async function runEnhancement(scanId) {
    showLoading('Enhancing Image...', 'Applying adaptive filters');
    try {
        const res = await fetch('/enhance', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scan_id: scanId, auto_mode: true })
        });
        const data = await res.json();
        hideLoading();
        if (data.success) {
            showToast('Enhancement complete', 'success');
            return data;
        } else {
            showToast(data.error, 'error');
            return null;
        }
    } catch (err) {
        hideLoading();
        showToast('Enhancement failed', 'error');
        return null;
    }
}

async function runAnalysis(scanId) {
    showLoading('Detecting Tumors...', 'Running segmentation and contour analysis');
    try {
        const res = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scan_id: scanId })
        });
        const data = await res.json();
        hideLoading();
        if (data.success) {
            showToast('Detection complete', 'success');
            return data;
        } else {
            showToast(data.error, 'error');
            return null;
        }
    } catch (err) {
        hideLoading();
        showToast('Analysis failed', 'error');
        return null;
    }
}

async function runRiskAnalysis(scanId) {
    showLoading('Computing Risk...', 'Calculating severity and confidence scores');
    try {
        const res = await fetch('/risk', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scan_id: scanId })
        });
        const data = await res.json();
        hideLoading();
        if (data.success) {
            showToast('Risk analysis complete', 'success');
            return data;
        } else {
            showToast(data.error, 'error');
            return null;
        }
    } catch (err) {
        hideLoading();
        showToast('Risk computation failed', 'error');
        return null;
    }
}

// ─── Export Functions ───────────────────────────────────────
async function exportReport(format) {
    const scanId = getCurrentScanId();
    if (!scanId) {
        showToast('No scan data available. Upload an image first.', 'error');
        return;
    }

    showLoading(`Generating ${format.toUpperCase()} Report...`, 'Compiling analysis data');

    try {
        const res = await fetch('/export', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ scan_id: scanId, format: format })
        });

        if (res.ok) {
            const blob = await res.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medical_report.${format}`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            hideLoading();
            showToast(`${format.toUpperCase()} report downloaded!`, 'success');
        } else {
            const data = await res.json();
            hideLoading();
            showToast(data.error || 'Export failed', 'error');
        }
    } catch (err) {
        hideLoading();
        showToast('Export failed: ' + err.message, 'error');
    }
}

// ─── Image Comparison Slider ────────────────────────────────
function initComparisonSlider() {
    const container = document.getElementById('comparisonContainer');
    if (!container) return;

    const slider = container.querySelector('.comparison-slider');
    const original = container.querySelector('.comparison-original');
    if (!slider || !original) return;

    let isDragging = false;

    function updateSlider(x) {
        const rect = container.getBoundingClientRect();
        let pos = ((x - rect.left) / rect.width) * 100;
        pos = Math.max(5, Math.min(95, pos));
        original.style.width = pos + '%';
        slider.style.left = pos + '%';
    }

    slider.addEventListener('mousedown', () => isDragging = true);
    container.addEventListener('mousedown', () => isDragging = true);

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            e.preventDefault();
            updateSlider(e.clientX);
        }
    });

    document.addEventListener('mouseup', () => isDragging = false);

    container.addEventListener('touchstart', () => isDragging = true);
    container.addEventListener('touchmove', (e) => {
        if (isDragging) {
            updateSlider(e.touches[0].clientX);
        }
    });
    container.addEventListener('touchend', () => isDragging = false);
}

// ─── Risk Gauge Animation ───────────────────────────────────
function animateGauge(score, elementId = 'riskGauge') {
    requestAnimationFrame(() => {
        const gauge = document.getElementById(elementId);
        if (!gauge) return;

        const fill = gauge.querySelector('.gauge-fill');
        const valueEl = document.getElementById('gaugeValue') ||
            gauge.querySelector('.gauge-value') ||
            gauge.closest('.risk-gauge-wrapper')?.querySelector('.gauge-value');

        if (!fill) return;

        // Ensure score is a number, clamp 0 to 1
        const nScore = Math.min(Math.max(parseFloat(score) || 0, 0), 1);
        console.log(`[Gauge] Animating to score: ${nScore * 100}%`);

        // Use native path length for pixel-perfect offset
        const circumference = fill.getTotalLength() || (Math.PI * 90);
        const offset = circumference - (nScore * circumference);

        // Reset and prepare
        fill.style.transition = 'none';
        fill.style.strokeDasharray = `${circumference} ${circumference}`;
        fill.style.strokeDashoffset = circumference;

        // Determine clinical color
        let color = 'var(--risk-low)';
        if (nScore >= 0.7) color = 'var(--risk-high)';
        else if (nScore >= 0.4) color = 'var(--risk-medium)';

        fill.style.stroke = color;

        // Trigger animation after layout
        setTimeout(() => {
            fill.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
            fill.style.strokeDashoffset = offset;
        }, 50);

        // Counter logic with cubic ease-out
        if (valueEl) {
            const target = Math.round(nScore * 100);
            if (target === 0) {
                valueEl.textContent = '0%';
                return;
            }

            const duration = 1500;
            const startTime = performance.now();

            function updateCounter(now) {
                const elapsed = now - startTime;
                const progress = Math.min(elapsed / duration, 1);

                // Ease out back
                const eased = 1 - Math.pow(1 - progress, 3);
                const val = Math.round(eased * target);

                valueEl.textContent = val + '%';

                if (progress < 1) {
                    requestAnimationFrame(updateCounter);
                }
            }
            requestAnimationFrame(updateCounter);
        }
    });
}

// ─── Load Scan Data ─────────────────────────────────────────
function loadScanData() {
    const data = sessionStorage.getItem('pipelineResult');
    if (data) {
        try {
            return JSON.parse(data);
        } catch (e) {
            return null;
        }
    }
    return null;
}

function getCurrentScanId() {
    // Check URL first
    const params = new URLSearchParams(window.location.search);
    const urlScanId = params.get('scan_id');
    if (urlScanId) return urlScanId;
    return sessionStorage.getItem('currentScanId');
}

// ─── Fetch Scan from Server ─────────────────────────────────
async function fetchScanData(scanId) {
    try {
        const res = await fetch(`/api/scan/${scanId}`);
        const data = await res.json();
        if (data.success) return data.scan;
    } catch (e) { /* ignore */ }
    return null;
}

// ─── Initialize ─────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initArcNav();
    initUploadZone();
    initComparisonSlider();

    const themeBtn = document.getElementById('themeToggle');
    if (themeBtn) themeBtn.addEventListener('click', toggleTheme);
});
