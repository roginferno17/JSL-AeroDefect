/**
 * JSL-AeroDefect Pro: Client Application Controller
 * Handles real-time inspection, tab switching, Chart.js FFT visualization,
 * 2D Coil Digital Twin canvas rendering, and ASTM Certificate generation.
 */

// Application State
const state = {
  currentTab: 'tab-inspect',
  currentSample: 'sample_sc.jpg',
  selectedGrade: '304',
  lineSpeedMps: 8.5,
  isCamSimRunning: false,
  camSimTimer: null,
  latestInspection: null,
  fftChartInstance: null,
  coilSimData: null,
  isWebcamActive: false,
  webcamStream: null,
  webcamInterval: null,
  auditHistory: JSON.parse(localStorage.getItem('jsl_inspection_history') || '[]'),
};

// DOM Elements
const el = {
  // Navigation
  navButtons: document.querySelectorAll('.nav-btn'),
  tabPanes: document.querySelectorAll('.tab-pane'),
  gradeSelect: document.getElementById('grade-select'),
  lineSpeedSlider: document.getElementById('line-speed-slider'),
  
  // Telemetry
  telemLatency: document.getElementById('telem-latency'),
  telemFps: document.getElementById('telem-fps'),
  telemSpeed: document.getElementById('telem-speed'),
  
  // Sidebar Info
  infoGradeName: document.getElementById('info-grade-name'),
  infoGradeDesc: document.getElementById('info-grade-desc'),
  infoMaxDsi: document.getElementById('info-max-dsi'),
  infoFinish: document.getElementById('info-finish'),
  
  // Tab 1: Live Inspection
  presetButtons: document.querySelectorAll('.preset-btn'),
  fileUploadInput: document.getElementById('file-upload-input'),
  btnToggleWebcam: document.getElementById('btn-toggle-webcam'),
  webcamBtnLabel: document.getElementById('webcam-btn-label'),
  btnCaptureWebcam: document.getElementById('btn-capture-webcam'),
  webcamVideo: document.getElementById('webcam-video'),
  webcamCanvas: document.getElementById('webcam-canvas'),
  btnToggleCamSim: document.getElementById('btn-toggle-cam-sim'),
  simBtnLabel: document.getElementById('sim-btn-label'),
  camScanline: document.getElementById('cam-scanline'),
  feedStatusTag: document.getElementById('feed-status-tag'),
  annotatedPreviewImg: document.getElementById('annotated-preview-img'),
  
  statPrimaryType: document.getElementById('stat-primary-type'),
  statPrimaryConf: document.getElementById('stat-primary-conf'),
  statDefectCount: document.getElementById('stat-defect-count'),
  statMaxDsi: document.getElementById('stat-max-dsi'),
  statLatency: document.getElementById('stat-latency'),
  
  dispBadge: document.getElementById('disp-badge'),
  dispSummary: document.getElementById('disp-summary'),
  dispViolations: document.getElementById('disp-violations'),
  defectsScrollList: document.getElementById('defects-scroll-list'),
  
  periodicityTeaser: document.getElementById('periodicity-teaser'),
  teaserSub: document.getElementById('teaser-sub'),
  btnJumpPeriodicity: document.getElementById('btn-jump-periodicity'),
  
  // Tab 2: Roll Mark Root Cause
  heroUrgencyBadge: document.getElementById('hero-urgency-badge'),
  heroStandName: document.getElementById('hero-stand-name'),
  heroActionText: document.getElementById('hero-action-text'),
  heroPeriod: document.getElementById('hero-period'),
  heroDiameter: document.getElementById('hero-diameter'),
  heroConf: document.getElementById('hero-conf'),
  fftCanvas: document.getElementById('fftChart'),
  
  // Tab 3: Coil Twin
  twinPrimeYield: document.getElementById('twin-prime-yield'),
  twinSavingsLakhs: document.getElementById('twin-savings-lakhs'),
  twinAnnualCrores: document.getElementById('twin-annual-crores'),
  btnReSimulateCoil: document.getElementById('btn-re-simulate-coil'),
  stripMapCanvas: document.getElementById('stripMapCanvas'),
  prescText: document.getElementById('presc-text'),
  
  // Tab 4: Certificate
  certDocNo: document.getElementById('cert-doc-no'),
  certDocDate: document.getElementById('cert-doc-date'),
  certCoilId: document.getElementById('cert-coil-id'),
  certGrade: document.getElementById('cert-grade'),
  certFinish: document.getElementById('cert-finish'),
  certValDensity: document.getElementById('cert-val-density'),
  certValInclusions: document.getElementById('cert-val-inclusions'),
  certValDsi: document.getElementById('cert-val-dsi'),
  certFinalVerdict: document.getElementById('cert-final-verdict'),
  certFinalNotes: document.getElementById('cert-final-notes'),

  // Tab 5: History Audit Log
  btnExportCsv: document.getElementById('btn-export-csv'),
  btnClearHistory: document.getElementById('btn-clear-history'),
  kpiTotalInspections: document.getElementById('kpi-total-inspections'),
  kpiPrimeRate: document.getElementById('kpi-prime-rate'),
  kpiAvgDsi: document.getElementById('kpi-avg-dsi'),
  kpiScrapCount: document.getElementById('kpi-scrap-count'),
  historyTableBody: document.getElementById('history-table-body'),
};

// Grade metadata dictionary for dynamic UI feedback
const GRADE_DESCRIPTIONS = {
  '304': {
    name: 'SS 304 (Standard Austenitic)',
    desc: '18/8 Chrome-Nickel. Sensitive to visual scratches and roll marks. High demand in food processing and architecture.',
    maxDsi: '40.0',
    finish: 'Bright Annealed (BA) / 2B'
  },
  '316L': {
    name: 'SS 316L (Pharma / Marine)',
    desc: 'Molybdenum-bearing grade. Zero-tolerance for inclusions or pitting micro-cavities (initiates crevice corrosion).',
    maxDsi: '30.0',
    finish: 'No. 4 Sanitary / 2B'
  },
  '430': {
    name: 'SS 430 (Ferritic Automotive)',
    desc: 'Straight Chromium alloy. Moderate tolerance for light cosmetic marks; strict rejection on edge cracks (breaks during stamping).',
    maxDsi: '55.0',
    finish: '2B / Mill No. 1'
  },
  '201': {
    name: 'SS 201 (Utensils & Hollowware)',
    desc: 'High formability cookware alloy. Permissive tolerance matrix for non-structural superficial marks.',
    maxDsi: '65.0',
    finish: '2B Cold Rolled'
  }
};

// ================= Tab Navigation =================
function switchTab(tabId) {
  state.currentTab = tabId;
  el.navButtons.forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });
  el.tabPanes.forEach(pane => {
    pane.classList.toggle('active', pane.id === tabId);
  });

  if (tabId === 'tab-periodicity') {
    initOrUpdateFFTChart();
  } else if (tabId === 'tab-coil-twin') {
    loadCoilSimulation();
  } else if (tabId === 'tab-certificate') {
    updateCertificateView();
  } else if (tabId === 'tab-history') {
    renderHistoryView();
  }
}

el.navButtons.forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

el.btnJumpPeriodicity?.addEventListener('click', () => {
  switchTab('tab-periodicity');
});

// ================= Steel Grade & Speed Controls =================
el.gradeSelect.addEventListener('change', (e) => {
  state.selectedGrade = e.target.value;
  const info = GRADE_DESCRIPTIONS[state.selectedGrade] || GRADE_DESCRIPTIONS['304'];
  el.infoGradeName.textContent = info.name;
  el.infoGradeDesc.textContent = info.desc;
  el.infoMaxDsi.textContent = info.maxDsi;
  el.infoFinish.textContent = info.finish;

  // Re-inspect current sample under new grade tolerances
  inspectSample(state.currentSample);
});

el.lineSpeedSlider.addEventListener('input', (e) => {
  state.lineSpeedMps = parseFloat(e.target.value);
  el.telemSpeed.textContent = `${state.lineSpeedMps.toFixed(1)} m/s`;
  
  // Calculate simulated camera frame rate based on speed (higher line speed requires higher FPS)
  const simulatedFps = Math.min(120, Math.round(state.lineSpeedMps * 6.5 + (Math.random() * 2 - 1)));
  el.telemFps.textContent = `${simulatedFps} FPS`;
});

// ================= Preset & File Upload & Webcam Inspection =================
el.presetButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    stopWebcam();
    el.presetButtons.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const sample = btn.dataset.sample;
    state.currentSample = sample;
    inspectSample(sample);
  });
});

el.fileUploadInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files[0]) {
    stopWebcam();
    const file = e.target.files[0];
    uploadAndInspect(file);
  }
});

// Webcam Controls
if (el.btnToggleWebcam) {
  el.btnToggleWebcam.addEventListener('click', toggleWebcam);
}
if (el.btnCaptureWebcam) {
  el.btnCaptureWebcam.addEventListener('click', captureAndInspectWebcamFrame);
}

async function toggleWebcam() {
  if (state.isWebcamActive) {
    stopWebcam();
  } else {
    await startWebcam();
  }
}

async function startWebcam() {
  // Stop simulation if running
  if (state.isCamSimRunning) {
    el.btnToggleCamSim.click();
  }
  
  try {
    el.feedStatusTag.textContent = 'CONNECTING LAPTOP WEBCAM...';
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    });
    state.webcamStream = stream;
    el.webcamVideo.srcObject = stream;
    await el.webcamVideo.play();
    
    state.isWebcamActive = true;
    el.webcamVideo.classList.remove('hidden');
    el.btnToggleWebcam.classList.add('active');
    el.webcamBtnLabel.textContent = 'Turn Off Webcam';
    el.btnCaptureWebcam.classList.remove('hidden');
    el.feedStatusTag.textContent = 'LIVE WEBCAM STREAM (Click "Capture Frame" to inspect)';
    el.camScanline.classList.remove('hidden');

    // Auto-inspect first frame after short delay
    setTimeout(() => {
      captureAndInspectWebcamFrame();
    }, 1000);
  } catch (err) {
    console.error('Webcam access error:', err);
    el.feedStatusTag.textContent = `WEBCAM ERROR: ${err.message || 'Permission denied'}`;
    alert(`Could not access camera: ${err.message}. Please check browser camera permissions.`);
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach(track => track.stop());
    state.webcamStream = null;
  }
  state.isWebcamActive = false;
  if (el.webcamVideo) {
    el.webcamVideo.srcObject = null;
    el.webcamVideo.classList.add('hidden');
  }
  if (el.btnToggleWebcam) {
    el.btnToggleWebcam.classList.remove('active');
    el.webcamBtnLabel.textContent = 'Open Laptop Webcam';
  }
  if (el.btnCaptureWebcam) {
    el.btnCaptureWebcam.classList.add('hidden');
  }
  if (el.camScanline) {
    el.camScanline.classList.add('hidden');
  }
}

async function captureAndInspectWebcamFrame() {
  if (!state.isWebcamActive || !el.webcamVideo.videoWidth) {
    return;
  }
  
  el.feedStatusTag.textContent = 'ANALYZING LIVE WEBCAM FRAME...';
  const canvas = el.webcamCanvas;
  canvas.width = el.webcamVideo.videoWidth;
  canvas.height = el.webcamVideo.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(el.webcamVideo, 0, 0, canvas.width, canvas.height);

  canvas.toBlob(async (blob) => {
    if (!blob) return;
    const file = new File([blob], 'webcam_frame.jpg', { type: 'image/jpeg' });
    await uploadAndInspect(file);
    // After inspection, keep video hidden temporarily or re-show with analysis
    el.webcamVideo.classList.add('hidden');
    el.feedStatusTag.textContent = 'WEBCAM INSPECTED: (Click "Capture Frame" or "Open Laptop Webcam" to refresh)';
  }, 'image/jpeg', 0.92);
}

async function inspectSample(sampleFilename) {
  el.feedStatusTag.textContent = `INSPECTING: ${sampleFilename}...`;
  const formData = new FormData();
  formData.append('sample_name', sampleFilename);
  formData.append('grade', state.selectedGrade);
  formData.append('confidence_threshold', 0.30);

  try {
    const resp = await fetch('/api/inspect', {
      method: 'POST',
      body: formData
    });
    if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
    const data = await resp.json();
    renderInspectionResults(data, sampleFilename);
  } catch (err) {
    console.error('Inspection failed:', err);
    el.feedStatusTag.textContent = `ERROR: ${err.message}`;
  }
}

async function uploadAndInspect(file) {
  el.feedStatusTag.textContent = `ANALYZING UPLOAD: ${file.name}...`;
  const formData = new FormData();
  formData.append('file', file);
  formData.append('grade', state.selectedGrade);
  formData.append('confidence_threshold', 0.30);

  try {
    const resp = await fetch('/api/inspect', {
      method: 'POST',
      body: formData
    });
    if (!resp.ok) throw new Error(`HTTP error ${resp.status}`);
    const data = await resp.json();
    renderInspectionResults(data, file.name);
  } catch (err) {
    console.error('Upload inspection failed:', err);
    el.feedStatusTag.textContent = `ERROR: ${err.message}`;
  }
}

function renderInspectionResults(data, sourceName) {
  state.latestInspection = data;
  const results = data.results;
  const disp = data.disposition;

  // Update Visual Canvas
  el.annotatedPreviewImg.src = data.annotated_image_b64;
  el.feedStatusTag.textContent = `ACTIVE: ${sourceName} (${results.defect_count} defects)`;

  // Update Primary Defect Type and Peak Confidence Score
  if (el.statPrimaryType) {
    el.statPrimaryType.textContent = results.primary_defect_type || (results.defect_count === 0 ? 'CLEAN PRIME' : 'UNCLASSIFIED');
  }
  if (el.statPrimaryConf) {
    el.statPrimaryConf.textContent = results.primary_confidence_pct || (results.defect_count === 0 ? '100.0%' : '--');
  }

  // Update Telemetry
  const latency = data.inference_latency_ms;
  el.statLatency.textContent = `${latency.toFixed(1)} ms`;
  el.telemLatency.textContent = `${latency.toFixed(1)} ms`;
  
  el.statDefectCount.textContent = results.defect_count;
  el.statMaxDsi.textContent = results.max_dsi.toFixed(1);
  el.statMaxDsi.className = `stat-num ${results.max_dsi > 50 ? 'text-rose' : results.max_dsi > 25 ? 'text-amber' : 'text-emerald'}`;

  // Update Disposition Verdict Box
  el.dispBadge.textContent = disp.overall_disposition.replace(/_/g, ' ');
  el.dispBadge.style.background = disp.badge_color;
  el.dispSummary.textContent = disp.summary;

  // Render violations list
  el.dispViolations.innerHTML = '';
  if (disp.violation_details && disp.violation_details.length > 0) {
    disp.violation_details.forEach(viol => {
      const tag = document.createElement('div');
      tag.className = 'violation-tag';
      tag.textContent = viol;
      el.dispViolations.appendChild(tag);
    });
  }

  // Render Defect Scroll List with prominent Defect Type & Confidence Score
  el.defectsScrollList.innerHTML = '';
  if (results.detections.length === 0) {
    el.defectsScrollList.innerHTML = `
      <div class="empty-state">
        ✨ <strong>PRIME SURFACE CONFORMITY</strong><br>
        No surface defects detected. Strip strictly satisfies ASTM A240 specification for Grade ${state.selectedGrade}.
      </div>`;
  } else {
    results.detections.forEach(det => {
      const card = document.createElement('div');
      card.className = 'defect-card-item';
      card.style.borderLeftColor = det.color;
      const confPct = (det.confidence * 100).toFixed(1);
      
      card.innerHTML = `
        <div class="defect-header-row">
          <div class="defect-type-pill" style="background:${det.color}22; border:1px solid ${det.color}; color:${det.color}">
            DEFECT TYPE: <strong>${det.defect_type || det.class_name}</strong>
          </div>
          <div class="defect-conf-pill">
            CONFIDENCE: <strong class="text-emerald">${confPct}%</strong>
          </div>
        </div>

        <!-- Visual Confidence Meter Bar -->
        <div class="conf-meter-wrapper">
          <div class="conf-meter-bar" style="width:${confPct}%; background:${det.confidence > 0.88 ? '#10b981' : '#f59e0b'}"></div>
        </div>

        <div class="defect-item-top" style="margin-top:6px;">
          <span class="defect-item-name">${det.id} &bull; ${det.class_name}</span>
          <span class="defect-item-dsi" style="color:${det.dsi > 60 ? '#ef4444' : det.dsi > 35 ? '#f59e0b' : '#10b981'}">
            DSI: ${det.dsi} &bull; ${det.severity}
          </span>
        </div>
        <div class="defect-item-origin">${det.description}</div>
        <div class="defect-item-metrics">
          <span>Area: <strong>${det.area_px} px (${det.area_pct}%)</strong></span>
          <span>Aspect Ratio: <strong>${det.aspect_ratio}</strong></span>
          <span>Origin: <strong>${det.origin}</strong></span>
        </div>
      `;
      el.defectsScrollList.appendChild(card);
    });
  }

  // Check if Roll Marks were detected -> show Root Cause Banner
  const hasRollMarks = results.detections.some(d => d.class_id === 'RM');
  if (hasRollMarks && data.periodicity_analysis) {
    el.periodicityTeaser.classList.remove('hidden');
    const p = data.periodicity_analysis;
    el.teaserSub.innerHTML = `Repeating wavelength: <strong>${p.dominant_period_mm}mm</strong> &bull; Diagnosed stand: <strong>${p.identified_stand}</strong>`;
    updatePeriodicityView(p);
  } else {
    el.periodicityTeaser.classList.add('hidden');
  }

  // Record into persistent Historical Audit Trail
  recordToAuditHistory(data, sourceName);
}

// ================= Live High-Speed Camera Simulator =================
const SIM_SAMPLES = [
  'sample_sc.jpg',
  'sample_rm.jpg',
  'sample_rs.jpg',
  'sample_ec.jpg',
  'clean_prime_steel.jpg',
  'sample_in.jpg',
  'sample_ps.jpg',
  'clean_prime_steel.jpg',
  'sample_pa.jpg',
  'sample_cr.jpg'
];
let simIndex = 0;

el.btnToggleCamSim.addEventListener('click', () => {
  state.isCamSimRunning = !state.isCamSimRunning;
  
  if (state.isCamSimRunning) {
    el.btnToggleCamSim.classList.add('running');
    el.simBtnLabel.textContent = 'Stop High-Speed Cam Sim';
    el.camScanline.classList.remove('hidden');
    
    // Cycle frames
    state.camSimTimer = setInterval(() => {
      simIndex = (simIndex + 1) % SIM_SAMPLES.length;
      const nextSample = SIM_SAMPLES[simIndex];
      state.currentSample = nextSample;
      inspectSample(nextSample);
    }, 1800);
  } else {
    el.btnToggleCamSim.classList.remove('running');
    el.simBtnLabel.textContent = 'Start High-Speed Cam Sim';
    el.camScanline.classList.add('hidden');
    if (state.camSimTimer) clearInterval(state.camSimTimer);
  }
});

// ================= Tab 2: Roll-Mark Root Cause & FFT Chart =================
function updatePeriodicityView(pData) {
  if (!pData) return;
  el.heroStandName.textContent = `${pData.identified_stand} Work Roll Fault`;
  el.heroPeriod.textContent = `${pData.dominant_period_mm} mm`;
  el.heroDiameter.textContent = `${pData.estimated_roll_diameter_mm} mm`;
  el.heroConf.textContent = `${pData.confidence_pct}%`;
  el.heroUrgencyBadge.textContent = pData.urgency + ' URGENCY';
  el.heroUrgencyBadge.className = `hero-badge ${pData.urgency === 'CRITICAL' ? 'danger' : 'warning'}`;
  el.heroActionText.innerHTML = `
    Dominant spatial period of <strong>${pData.dominant_period_mm} mm</strong> corresponds precisely to a work roll diameter of <strong>${pData.estimated_roll_diameter_mm} mm</strong> with 3.5% forward slip.<br>
    Recommended Action: <em>${pData.maintenance_action}</em>
  `;

  // Highlight matched stand item in list
  document.querySelectorAll('.stand-item').forEach(item => {
    item.classList.remove('fault');
    const pill = item.querySelector('.stand-pill');
    pill.className = 'stand-pill safe';
    pill.textContent = 'Normal';
  });

  const standKey = pData.identified_stand.includes('F1') ? 'F1' :
                   pData.identified_stand.includes('F2') ? 'F2' :
                   pData.identified_stand.includes('F3') ? 'F3' :
                   pData.identified_stand.includes('F4') ? 'F4' :
                   pData.identified_stand.includes('F5') ? 'F5' : 'Bridle';

  const matchedElem = document.querySelector(`.stand-item[data-stand="${standKey}"]`);
  if (matchedElem) {
    matchedElem.classList.add('fault');
    const pill = matchedElem.querySelector('.stand-pill');
    pill.className = 'stand-pill danger';
    pill.textContent = 'FAULT DETECTED';
  }

  // Update FFT Spectrum Chart
  if (pData.spectrum && pData.spectrum.length > 0) {
    renderFFTChart(pData.spectrum, pData.dominant_period_mm);
  }
}

function initOrUpdateFFTChart() {
  if (state.latestInspection?.periodicity_analysis) {
    updatePeriodicityView(state.latestInspection.periodicity_analysis);
  } else {
    // Default demonstration data for Stand F3
    const demoSpectrum = [
      { wavelength_mm: 700, power: 0.05 },
      { wavelength_mm: 800, power: 0.12 },
      { wavelength_mm: 900, power: 0.18 },
      { wavelength_mm: 1000, power: 0.42 },
      { wavelength_mm: 1068, power: 1.00 }, // Peak at Stand F3
      { wavelength_mm: 1150, power: 0.35 },
      { wavelength_mm: 1250, power: 0.15 },
      { wavelength_mm: 1350, power: 0.08 },
      { wavelength_mm: 1500, power: 0.03 },
    ];
    renderFFTChart(demoSpectrum, 1068.1);
  }
}

function renderFFTChart(spectrumData, peakWavelength) {
  if (!el.fftCanvas) return;
  const ctx = el.fftCanvas.getContext('2d');
  
  const labels = spectrumData.map(d => `${d.wavelength_mm} mm`);
  const values = spectrumData.map(d => d.power);

  if (state.fftChartInstance) {
    state.fftChartInstance.destroy();
  }

  state.fftChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Normalized Fourier Power |P(f)|²',
        data: values,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.15)',
        fill: true,
        tension: 0.35,
        borderWidth: 2,
        pointBackgroundColor: values.map(v => v > 0.8 ? '#ff4d4d' : '#f59e0b'),
        pointRadius: values.map(v => v > 0.8 ? 6 : 3),
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => `Wavelength: ${items[0].label}`,
            label: (item) => `Spectral Intensity: ${(item.raw * 100).toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { color: 'rgba(255, 255, 255, 0.05)' },
          ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } },
          min: 0,
          max: 1.1
        }
      }
    }
  });
}

// ================= Tab 3: Coil Digital Twin Canvas =================
async function loadCoilSimulation() {
  try {
    const resp = await fetch('/api/coil/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        coil_id: 'JSL-CR-2026-9842',
        grade: state.selectedGrade,
        total_length_m: 1500.0,
        strip_width_mm: 1250.0,
        thickness_mm: 1.5
      })
    });
    if (!resp.ok) throw new Error('Simulation failed');
    const data = await resp.json();
    state.coilSimData = data;
    renderCoilTwin(data);
  } catch (err) {
    console.error('Coil simulation failed:', err);
  }
}

el.btnReSimulateCoil?.addEventListener('click', loadCoilSimulation);

function renderCoilTwin(data) {
  // Update ROI Header
  el.twinPrimeYield.textContent = `${data.yield_impact.prime_yield_pct}%`;
  el.twinSavingsLakhs.textContent = `₹ ${data.economic_roi.money_saved_per_coil_lakhs} Lakhs`;
  el.twinAnnualCrores.textContent = `₹ ${data.economic_roi.annual_savings_potential_crores} Crores`;
  el.prescText.innerHTML = `${data.smart_slitting_recommendation.slitting_pattern}`;

  // Draw 2D Strip Canvas
  const canvas = el.stripMapCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  // Background steel strip
  ctx.fillStyle = '#111827';
  ctx.fillRect(0, 0, w, h);

  // Strip grain lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
  for (let y = 10; y < h; y += 15) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Draw Heatmap Grid
  const grid = data.heatmap_grid;
  if (grid && grid.matrix) {
    const cellW = w / grid.length_bins;
    const cellH = h / grid.width_bins;

    for (let row = 0; row < grid.width_bins; row++) {
      for (let col = 0; col < grid.length_bins; col++) {
        const count = grid.matrix[row][col];
        if (count > 0) {
          const alpha = Math.min(0.85, count * 0.28);
          ctx.fillStyle = count >= 3 ? `rgba(239, 68, 68, ${alpha})` : `rgba(245, 158, 11, ${alpha})`;
          ctx.fillRect(col * cellW, row * cellH, cellW, cellH);
        }
      }
    }
  }

  // Draw Slitting & Trimming Cut Lines
  const rec = data.smart_slitting_recommendation;
  
  // Head crop cut line
  const headCutX = (rec.head_crop_shear_m / 1500.0) * w;
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 2;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(headCutX, 0);
  ctx.lineTo(headCutX, h);
  ctx.stroke();

  // Tail crop cut line
  const tailCutX = w - (rec.tail_crop_shear_m / 1500.0) * w;
  ctx.beginPath();
  ctx.moveTo(tailCutX, 0);
  ctx.lineTo(tailCutX, h);
  ctx.stroke();

  // Drive side trim line (bottom 65mm out of 1250mm)
  const driveTrimY = (rec.drive_side_trim_mm / 1250.0) * h;
  ctx.beginPath();
  ctx.moveTo(headCutX, driveTrimY);
  ctx.lineTo(tailCutX, driveTrimY);
  ctx.stroke();
  ctx.setLineDash([]); // Reset dash

  // Prime Salvage Zone Highlight
  ctx.fillStyle = 'rgba(16, 185, 129, 0.08)';
  ctx.fillRect(headCutX, driveTrimY, tailCutX - headCutX, h - driveTrimY);
}

// ================= Tab 4: ASTM Quality Certificate =================
function updateCertificateView() {
  const now = new Date();
  el.certDocDate.textContent = now.toISOString().split('T')[0];
  el.certGrade.textContent = GRADE_DESCRIPTIONS[state.selectedGrade]?.name || 'Austenitic 304';
  el.certFinish.textContent = GRADE_DESCRIPTIONS[state.selectedGrade]?.finish || '2B / BA';

  if (state.latestInspection) {
    const res = state.latestInspection.results;
    const disp = state.latestInspection.disposition;

    el.certValDsi.textContent = res.max_dsi.toFixed(1);
    el.certValDensity.textContent = `${(res.defect_count / 15.0).toFixed(3)} defects / m²`;
    
    const inclusionsCount = res.detections.filter(d => d.class_id === 'IN' || d.class_id === 'PS').length;
    el.certValInclusions.textContent = `${inclusionsCount} Found`;

    el.certFinalVerdict.textContent = disp.overall_disposition.replace(/_/g, ' ');
    el.certFinalVerdict.style.color = disp.badge_color;
    el.certFinalNotes.textContent = disp.summary;
  }
}

// ================= Tab 5: Historical Inspection Audit Trail & CSV =================
function recordToAuditHistory(data, sourceName) {
  const res = data.results;
  const disp = data.disposition;
  const now = new Date();
  
  const record = {
    id: `INS-${Date.now().toString().slice(-6)}`,
    timestamp: now.toLocaleTimeString() + ' ' + now.toLocaleDateString(),
    isoTimestamp: now.toISOString(),
    source: sourceName || 'Live Camera',
    grade: state.selectedGrade,
    primaryDefect: res.primary_defect_type || (res.defect_count === 0 ? 'CLEAN PRIME' : 'UNCLASSIFIED'),
    confidence: res.primary_confidence_pct || '100%',
    defectCount: res.defect_count,
    maxDsi: res.max_dsi.toFixed(1),
    latencyMs: data.inference_latency_ms.toFixed(1),
    disposition: disp.overall_disposition.replace(/_/g, ' '),
    dispositionColor: disp.badge_color,
    summary: disp.summary
  };

  state.auditHistory.unshift(record);
  // Keep last 150 records
  if (state.auditHistory.length > 150) {
    state.auditHistory.pop();
  }
  
  try {
    localStorage.setItem('jsl_inspection_history', JSON.stringify(state.auditHistory));
  } catch (e) {
    console.warn('LocalStorage save failed:', e);
  }

  // If currently on Tab 5, update table live
  if (state.currentTab === 'tab-history') {
    renderHistoryView();
  }
}

function renderHistoryView() {
  const history = state.auditHistory;
  
  // Calculate KPIs
  const total = history.length;
  const primeCount = history.filter(h => h.disposition.includes('PRIME')).length;
  const scrapCount = history.filter(h => h.disposition.includes('SCRAP')).length;
  const primeRate = total > 0 ? Math.round((primeCount / total) * 100) : 0;
  const avgDsi = total > 0 ? (history.reduce((acc, h) => acc + parseFloat(h.maxDsi || 0), 0) / total).toFixed(1) : '0.0';

  if (el.kpiTotalInspections) el.kpiTotalInspections.textContent = total;
  if (el.kpiPrimeRate) el.kpiPrimeRate.textContent = `${primeRate}%`;
  if (el.kpiAvgDsi) el.kpiAvgDsi.textContent = avgDsi;
  if (el.kpiScrapCount) el.kpiScrapCount.textContent = scrapCount;

  // Render Table
  if (!el.historyTableBody) return;
  
  if (total === 0) {
    el.historyTableBody.innerHTML = `
      <tr class="empty-history-row">
        <td colspan="10" style="text-align: center; padding: 36px 0; color: var(--text-dim);">
          📋 No inspections performed yet. Run Live Inspection, upload an image, or use the webcam to generate entries.
        </td>
      </tr>`;
    return;
  }

  el.historyTableBody.innerHTML = history.map((rec, idx) => {
    let dispBadgeClass = 'badge-disposition';
    let badgeStyle = `background: ${rec.dispositionColor}22; border: 1px solid ${rec.dispositionColor}; color: ${rec.dispositionColor};`;
    
    return `
      <tr>
        <td style="color: var(--text-dim); font-family: monospace;">#${total - idx}</td>
        <td style="color: var(--text-muted);">${rec.timestamp}</td>
        <td><strong>${rec.source}</strong></td>
        <td><span style="background: rgba(255,107,53,0.12); padding: 2px 6px; border-radius: 4px; color: var(--jsl-orange); font-weight:700;">SS ${rec.grade}</span></td>
        <td><span style="font-weight:600;">${rec.primaryDefect}</span></td>
        <td style="text-align:center;"><strong>${rec.defectCount}</strong></td>
        <td><strong style="color: ${rec.maxDsi > 50 ? '#ef4444' : rec.maxDsi > 25 ? '#f59e0b' : '#10b981'}">${rec.maxDsi}</strong></td>
        <td style="color: var(--text-dim);">${rec.latencyMs} ms</td>
        <td><span class="${dispBadgeClass}" style="${badgeStyle}">${rec.disposition}</span></td>
        <td>
          <button class="btn-row-cert" onclick="viewHistoricalCert('${rec.id}')">View Cert</button>
        </td>
      </tr>
    `;
  }).join('');
}

// Global hook for row-click
window.viewHistoricalCert = function(recordId) {
  const rec = state.auditHistory.find(h => h.id === recordId);
  if (rec) {
    switchTab('tab-certificate');
  }
};

function exportHistoryToCSV() {
  if (state.auditHistory.length === 0) {
    alert('No inspection records to export yet. Perform some inspections first!');
    return;
  }

  const headers = [
    'Inspection ID',
    'Timestamp',
    'Source / File Name',
    'Steel Grade',
    'Primary Defect Type',
    'Confidence Score',
    'Defects Count',
    'Max DSI Severity',
    'Inference Latency (ms)',
    'Disposition Verdict',
    'Quality Summary'
  ];

  const rows = state.auditHistory.map(rec => [
    rec.id,
    rec.isoTimestamp || rec.timestamp,
    `"${rec.source}"`,
    `SS ${rec.grade}`,
    `"${rec.primaryDefect}"`,
    rec.confidence,
    rec.defectCount,
    rec.maxDsi,
    rec.latencyMs,
    `"${rec.disposition}"`,
    `"${(rec.summary || '').replace(/"/g, '""')}"`
  ]);

  const csvContent = [
    headers.join(','),
    ...rows.map(e => e.join(','))
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  const dateStr = new Date().toISOString().split('T')[0];
  link.setAttribute('download', `JSL_AeroDefect_Audit_Log_${dateStr}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function clearInspectionHistory() {
  if (state.auditHistory.length === 0) return;
  if (confirm('Are you sure you want to clear all historical inspection records?')) {
    state.auditHistory = [];
    localStorage.removeItem('jsl_inspection_history');
    renderHistoryView();
  }
}

// Attach export & clear buttons
if (el.btnExportCsv) {
  el.btnExportCsv.addEventListener('click', exportHistoryToCSV);
}
if (el.btnClearHistory) {
  el.btnClearHistory.addEventListener('click', clearInspectionHistory);
}

// Initial Boot
window.addEventListener('DOMContentLoaded', () => {
  // Start with default sample
  inspectSample('sample_sc.jpg');
});

