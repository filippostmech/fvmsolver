const API_BASE = '';
let currentRunId = null;
let pollInterval = null;
let pressureHistory = [];
let latestDiag = null;
let latestFrame = null;

const fieldCanvas = document.getElementById('field-canvas');
const fieldCtx = fieldCanvas.getContext('2d');
const colorbarCanvas = document.getElementById('colorbar-canvas');
const colorbarCtx = colorbarCanvas.getContext('2d');

const btnRun = document.getElementById('btn-run');
const btnPause = document.getElementById('btn-pause');
const btnStop = document.getElementById('btn-stop');
const statusText = document.getElementById('status-text');
const progressFill = document.getElementById('progress-fill');
const diagContainer = document.getElementById('diagnostics-container');
const fieldSelect = document.getElementById('field-select');
const simInfoLeft = document.getElementById('sim-info-left');
const simInfoRight = document.getElementById('sim-info-right');

btnRun.addEventListener('click', startSimulation);
btnPause.addEventListener('click', pauseSimulation);
btnStop.addEventListener('click', stopSimulation);
fieldSelect.addEventListener('change', () => { if (latestFrame) renderField(latestFrame); });
document.getElementById('show-contour').addEventListener('change', () => { if (latestFrame) renderField(latestFrame); });
document.getElementById('show-nozzle').addEventListener('change', () => { if (latestFrame) renderField(latestFrame); });

const FIELD_INFO = {
    alpha: { title: 'VOF - Polymer Volume Fraction', unit: '', fmt: v => v.toFixed(3) },
    u_mag: { title: 'Velocity Magnitude', unit: 'm/s', fmt: v => engFormat(v) },
    T: { title: 'Temperature', unit: 'K', fmt: v => v.toFixed(1) },
    p: { title: 'Pressure', unit: 'Pa', fmt: v => engFormat(v) },
    eta: { title: 'Viscosity', unit: 'Pa\u00B7s', fmt: v => engFormat(v) },
};

function engFormat(val) {
    const abs = Math.abs(val);
    if (abs === 0) return '0';
    if (abs >= 1e6) return (val / 1e6).toFixed(2) + ' M';
    if (abs >= 1e3) return (val / 1e3).toFixed(2) + ' k';
    if (abs >= 1) return val.toFixed(2);
    if (abs >= 1e-3) return (val * 1e3).toFixed(2) + ' m';
    if (abs >= 1e-6) return (val * 1e6).toFixed(2) + ' \u00B5';
    return val.toExponential(2);
}

function engFormatPa(val) {
    const abs = Math.abs(val);
    if (abs >= 1e6) return (val / 1e6).toFixed(2) + ' MPa';
    if (abs >= 1e3) return (val / 1e3).toFixed(1) + ' kPa';
    return val.toFixed(1) + ' Pa';
}

function autoCalculateSteps() {
    const gapMM = parseFloat(document.getElementById('nozzle_bed_gap').value);
    const flowRateMM3 = parseFloat(document.getElementById('flow_rate').value);
    const diamMM = parseFloat(document.getElementById('nozzle_diameter').value);
    const dt = parseFloat(document.getElementById('dt').value);

    const gapM = gapMM * 1e-3;
    const flowRateM3 = flowRateMM3 * 1e-9;
    const radiusM = (diamMM * 1e-3) / 2;
    const area = Math.PI * radiusM * radiusM;
    const uMean = flowRateM3 / area;
    const uMax = 2.0 * uMean;

    const tTravel = gapM / uMax;
    const tTotal = tTravel * 1.3;
    const steps = Math.ceil(tTotal / dt);
    const rounded = Math.ceil(steps / 500) * 500;

    document.getElementById('n_steps').value = rounded;
    const fpu = Math.max(1, Math.ceil(rounded / 100));
    document.getElementById('frames_per_update').value = fpu;
}

document.getElementById('nozzle_bed_gap').addEventListener('change', autoCalculateSteps);
document.getElementById('flow_rate').addEventListener('change', autoCalculateSteps);
document.getElementById('nozzle_diameter').addEventListener('change', autoCalculateSteps);
document.getElementById('dt').addEventListener('change', autoCalculateSteps);
autoCalculateSteps();

function getConfig() {
    const gapMM = parseFloat(document.getElementById('nozzle_bed_gap').value);
    return {
        nozzle_diameter: parseFloat(document.getElementById('nozzle_diameter').value) * 1e-3,
        nozzle_length: parseFloat(document.getElementById('nozzle_length').value) * 1e-3,
        nozzle_bed_gap: gapMM * 1e-3,
        flow_rate: parseFloat(document.getElementById('flow_rate').value) * 1e-9,
        T_nozzle: parseFloat(document.getElementById('T_nozzle').value) + 273.15,
        T_ambient: parseFloat(document.getElementById('T_ambient').value) + 273.15,
        h_conv: parseFloat(document.getElementById('h_conv').value),
        eta_0: parseFloat(document.getElementById('eta_0').value),
        eta_inf: parseFloat(document.getElementById('eta_inf').value),
        lambda_cy: parseFloat(document.getElementById('lambda_cy').value),
        a_cy: parseFloat(document.getElementById('a_cy').value),
        n_cy: parseFloat(document.getElementById('n_cy').value),
        E_a: parseFloat(document.getElementById('E_a').value),
        sigma: parseFloat(document.getElementById('sigma').value),
        nr: parseInt(document.getElementById('nr').value),
        nz: parseInt(document.getElementById('nz').value),
        dt: parseFloat(document.getElementById('dt').value),
        n_steps: parseInt(document.getElementById('n_steps').value),
        frames_per_update: parseInt(document.getElementById('frames_per_update').value),
    };
}

async function startSimulation() {
    const config = getConfig();
    pressureHistory = [];
    latestDiag = null;
    latestFrame = null;

    const diam = document.getElementById('nozzle_diameter').value;
    const flow = document.getElementById('flow_rate').value;
    const temp = document.getElementById('T_nozzle').value;
    simInfoLeft.textContent = `TPU extrusion: ${diam} mm nozzle, ${flow} mm\u00B3/s, ${temp}\u00B0C`;
    simInfoRight.textContent = '';

    try {
        const resp = await fetch(`${API_BASE}/api/simulate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const data = await resp.json();
        currentRunId = data.run_id;

        btnRun.disabled = true;
        btnPause.disabled = false;
        btnStop.disabled = false;
        statusText.textContent = 'Running...';
        statusText.style.color = '#81c784';

        startPolling();
    } catch (e) {
        statusText.textContent = 'Error: ' + e.message;
        statusText.style.color = '#ef5350';
    }
}

async function pauseSimulation() {
    if (!currentRunId) return;
    try {
        const resp = await fetch(`${API_BASE}/api/pause/${currentRunId}`, { method: 'POST' });
        const data = await resp.json();
        btnPause.textContent = data.paused ? 'Resume' : 'Pause';
        statusText.textContent = data.paused ? 'Paused' : 'Running...';
    } catch (e) {}
}

async function stopSimulation() {
    if (!currentRunId) return;
    try {
        await fetch(`${API_BASE}/api/stop/${currentRunId}`, { method: 'POST' });
        statusText.textContent = 'Stopping...';
    } catch (e) {}
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(pollStatus, 500);
}

async function pollStatus() {
    if (!currentRunId) return;

    try {
        const resp = await fetch(`${API_BASE}/api/latest/${currentRunId}`);
        const data = await resp.json();

        if (data.frame) {
            latestFrame = data.frame;
            latestDiag = data.frame.diagnostics;
            renderField(data.frame);
            updateDiagnostics(data.frame.diagnostics);
            updatePlots(data.frame.diagnostics);
            updateSimInfo(data.frame.diagnostics);
        }

        const pct = data.total_steps > 0 ? (data.current_step / data.total_steps * 100) : 0;
        progressFill.style.width = pct + '%';
        statusText.textContent = `${data.status} (step ${data.current_step}/${data.total_steps})`;

        if (data.status === 'completed' || data.status === 'error' || data.status === 'stopped') {
            clearInterval(pollInterval);
            pollInterval = null;
            btnRun.disabled = false;
            btnPause.disabled = true;
            btnStop.disabled = true;
            btnPause.textContent = 'Pause';

            if (data.status === 'error') {
                statusText.style.color = '#ef5350';
            } else {
                statusText.style.color = '#4fc3f7';
            }
        }
    } catch (e) {}
}

function updateSimInfo(diag) {
    if (!diag || diag.step === undefined) return;
    const timeStr = diag.time < 1e-3
        ? (diag.time * 1e6).toFixed(1) + ' \u00B5s'
        : diag.time < 1
            ? (diag.time * 1e3).toFixed(2) + ' ms'
            : diag.time.toFixed(4) + ' s';
    simInfoRight.textContent = `Step ${diag.step} | t = ${timeStr} | CFL = ${diag.cfl.toFixed(4)}`;
}

function colormapRainbow(t) {
    t = Math.max(0, Math.min(1, t));

    let r, g, b;
    if (t < 0.2) {
        const s = t / 0.2;
        r = Math.floor(15 + s * 45);
        g = Math.floor(10 + s * 50);
        b = Math.floor(80 + s * 140);
    } else if (t < 0.4) {
        const s = (t - 0.2) / 0.2;
        r = Math.floor(60 - s * 30);
        g = Math.floor(60 + s * 140);
        b = Math.floor(220 - s * 40);
    } else if (t < 0.6) {
        const s = (t - 0.4) / 0.2;
        r = Math.floor(30 + s * 80);
        g = Math.floor(200 + s * 50);
        b = Math.floor(180 - s * 120);
    } else if (t < 0.8) {
        const s = (t - 0.6) / 0.2;
        r = Math.floor(110 + s * 145);
        g = Math.floor(250 - s * 50);
        b = Math.floor(60 - s * 40);
    } else {
        const s = (t - 0.8) / 0.2;
        r = 255;
        g = Math.floor(200 - s * 170);
        b = Math.floor(20 + s * 20);
    }
    return `rgb(${r},${g},${b})`;
}

function colormapVOF(t) {
    t = Math.max(0, Math.min(1, t));
    const r = Math.floor(210 * t + 30 * (1 - t));
    const g = Math.floor(40 * t + 50 * (1 - t));
    const b = Math.floor(40 * t + 210 * (1 - t));
    return `rgb(${r},${g},${b})`;
}

function colormap(t, fieldName) {
    if (fieldName === 'alpha') {
        return colormapVOF(t);
    }
    return colormapRainbow(t);
}

function renderField(frame) {
    const canvas = fieldCanvas;
    const ctx = fieldCtx;
    const fieldName = fieldSelect.value;
    const data = frame[fieldName];
    if (!data) return;

    const nr = frame.nr;
    const nz = frame.nz;
    const cw = canvas.width;
    const ch = canvas.height;

    ctx.fillStyle = '#0a1520';
    ctx.fillRect(0, 0, cw, ch);

    let fmin = Infinity, fmax = -Infinity;
    for (let i = 0; i < nr; i++) {
        for (let j = 0; j < nz; j++) {
            const v = data[i][j];
            if (isFinite(v)) {
                if (v < fmin) fmin = v;
                if (v > fmax) fmax = v;
            }
        }
    }
    if (!isFinite(fmin) || !isFinite(fmax)) {
        fmin = 0; fmax = 1;
    }
    if (Math.abs(fmax - fmin) < 1e-20) {
        fmax = fmin + 1;
    }

    const margin = { top: 50, bottom: 50, left: 50, right: 20 };
    const plotW = cw - margin.left - margin.right;
    const plotH = ch - margin.top - margin.bottom;
    const halfW = plotW / 2;

    const cellW = halfW / nr;
    const cellH = plotH / nz;

    const centerX = margin.left + halfW;

    for (let i = 0; i < nr; i++) {
        for (let j = 0; j < nz; j++) {
            const v = data[i][j];
            const t = (v - fmin) / (fmax - fmin);
            const col = colormap(t, fieldName);

            const x_right = centerX + i * cellW;
            const x_left = centerX - (i + 1) * cellW;
            const y = margin.top + (nz - 1 - j) * cellH;

            ctx.fillStyle = col;
            ctx.fillRect(x_right, y, cellW + 0.5, cellH + 0.5);
            ctx.fillRect(x_left, y, cellW + 0.5, cellH + 0.5);
        }
    }

    const rMax = frame.r_centers[nr - 1] + (frame.r_centers[1] - frame.r_centers[0]) * 0.5;
    const zMin = frame.z_centers[0];
    const zMax = frame.z_centers[nz - 1];
    const zRange = zMax - zMin;

    const showNozzle = document.getElementById('show-nozzle').checked;

    if (showNozzle) {
        const nozzleRFrac = frame.nozzle_radius / rMax;
        const nozzleRPx = nozzleRFrac * halfW;
        const nozzleZEnd = frame.nozzle_z_end;
        const zFrac = (nozzleZEnd - zMin) / zRange;
        const nozzleYPx = margin.top + (1 - zFrac) * plotH;

        const wallThick = 8;

        ctx.fillStyle = '#5c6b7a';
        ctx.fillRect(centerX + nozzleRPx, margin.top, wallThick, nozzleYPx - margin.top);
        ctx.fillRect(centerX - nozzleRPx - wallThick, margin.top, wallThick, nozzleYPx - margin.top);

        ctx.fillStyle = '#6d7e8f';
        ctx.fillRect(centerX + nozzleRPx, margin.top, wallThick, 4);
        ctx.fillRect(centerX - nozzleRPx - wallThick, margin.top, wallThick, 4);

        const outerExtend = 30;
        ctx.fillStyle = '#5c6b7a';
        ctx.fillRect(centerX + nozzleRPx + wallThick, nozzleYPx - 6, outerExtend, 6);
        ctx.fillRect(centerX - nozzleRPx - wallThick - outerExtend, nozzleYPx - 6, outerExtend, 6);

        ctx.strokeStyle = '#8899aa';
        ctx.lineWidth = 1;
        ctx.strokeRect(centerX + nozzleRPx, margin.top, wallThick, nozzleYPx - margin.top);
        ctx.strokeRect(centerX - nozzleRPx - wallThick, margin.top, wallThick, nozzleYPx - margin.top);

        ctx.fillStyle = '#8899aa';
        ctx.font = 'bold 13px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('NOZZLE', centerX, margin.top + 20);

        const arrowY1 = margin.top + 30;
        const arrowY2 = margin.top + 55;
        const arrowX = centerX;
        ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY1);
        ctx.lineTo(arrowX, arrowY2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(arrowX - 5, arrowY2 - 6);
        ctx.lineTo(arrowX, arrowY2);
        ctx.lineTo(arrowX + 5, arrowY2 - 6);
        ctx.stroke();

        ctx.font = '11px sans-serif';
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillText('flow', arrowX + 18, (arrowY1 + arrowY2) / 2 + 4);

        ctx.fillStyle = 'rgba(255, 171, 64, 0.8)';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText('Nozzle Exit', centerX + nozzleRPx + wallThick + outerExtend - 2, nozzleYPx - 10);

        ctx.fillStyle = 'rgba(120, 180, 255, 0.6)';
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'left';
        const extrudateY = nozzleYPx + (margin.top + plotH - nozzleYPx) * 0.3;
        ctx.fillText('Extrudate', centerX + nozzleRPx + wallThick + 6, extrudateY);

        ctx.fillStyle = 'rgba(100, 160, 220, 0.4)';
        ctx.textAlign = 'right';
        ctx.fillText('Air', centerX - nozzleRPx - wallThick - outerExtend + 5, extrudateY);
    }

    const bedY = margin.top + plotH;
    const bedH = 10;
    const grad = ctx.createLinearGradient(0, bedY, 0, bedY + bedH);
    grad.addColorStop(0, '#607d8b');
    grad.addColorStop(1, '#37474f');
    ctx.fillStyle = grad;
    ctx.fillRect(margin.left - 10, bedY, plotW + 20, bedH);
    ctx.strokeStyle = '#90a4ae';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left - 10, bedY, plotW + 20, bedH);

    ctx.fillStyle = '#90a4ae';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('PRINT BED', centerX, bedY + bedH + 16);

    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(centerX, margin.top);
    ctx.lineTo(centerX, bedY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.save();
    ctx.translate(centerX + 4, bedY - 14);
    ctx.font = '10px sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('axis of symmetry', 0, 0);
    ctx.restore();

    if (document.getElementById('show-contour').checked && frame.contour_r && frame.contour_r.length > 0) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2.5;
        ctx.shadowColor = 'rgba(255,255,255,0.4)';
        ctx.shadowBlur = 4;

        const points = [];
        for (let k = 0; k < frame.contour_r.length; k++) {
            const rFrac = frame.contour_r[k] / rMax;
            const zFrac = (frame.contour_z[k] - zMin) / zRange;
            const px = centerX + rFrac * halfW;
            const py = margin.top + (1 - zFrac) * plotH;
            points.push({ x: px, y: py, xm: centerX - rFrac * halfW });
        }

        points.sort((a, b) => a.y - b.y);

        function drawSmoothCurve(ctx, pts, getX) {
            if (pts.length < 2) return;
            ctx.beginPath();
            ctx.moveTo(getX(pts[0]), pts[0].y);

            if (pts.length === 2) {
                ctx.lineTo(getX(pts[1]), pts[1].y);
            } else {
                for (let k = 0; k < pts.length - 1; k++) {
                    const p0 = k > 0 ? pts[k - 1] : pts[k];
                    const p1 = pts[k];
                    const p2 = pts[k + 1];
                    const p3 = k < pts.length - 2 ? pts[k + 2] : pts[k + 1];

                    const cp1x = getX(p1) + (getX(p2) - getX(p0)) / 6;
                    const cp1y = p1.y + (p2.y - p0.y) / 6;
                    const cp2x = getX(p2) - (getX(p3) - getX(p1)) / 6;
                    const cp2y = p2.y - (p3.y - p1.y) / 6;

                    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, getX(p2), p2.y);
                }
            }
            ctx.stroke();
        }

        drawSmoothCurve(ctx, points, p => p.x);
        drawSmoothCurve(ctx, points, p => p.xm);

        ctx.shadowBlur = 0;
    }

    const rMaxMM = rMax * 1000;
    const zRangeMM = zRange * 1000;

    ctx.fillStyle = '#b0bec5';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('r (mm)', centerX + halfW / 2, ch - 8);

    ctx.save();
    ctx.translate(14, margin.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#b0bec5';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('z (mm)', 0, 0);
    ctx.restore();

    ctx.font = '11px sans-serif';
    ctx.fillStyle = '#7a8fa6';
    ctx.textAlign = 'center';

    const nTicksR = 4;
    for (let i = 0; i <= nTicksR; i++) {
        const frac = i / nTicksR;
        const px_r = centerX + frac * halfW;
        const px_l = centerX - frac * halfW;
        const val = (frac * rMaxMM).toFixed(2);

        ctx.fillText(val, px_r, margin.top + plotH + 28);
        if (i > 0) {
            ctx.fillText('-' + val, px_l, margin.top + plotH + 28);
        }

        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 0.5;
        if (i > 0 && i < nTicksR) {
            ctx.beginPath();
            ctx.moveTo(px_r, margin.top);
            ctx.lineTo(px_r, margin.top + plotH);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(px_l, margin.top);
            ctx.lineTo(px_l, margin.top + plotH);
            ctx.stroke();
        }
    }

    ctx.textAlign = 'right';
    const nTicksZ = 5;
    for (let i = 0; i <= nTicksZ; i++) {
        const frac = i / nTicksZ;
        const py = margin.top + frac * plotH;
        const zVal = ((1 - frac) * zRangeMM + zMin * 1000).toFixed(2);

        ctx.fillText(zVal, margin.left - 6, py + 4);

        if (i > 0 && i < nTicksZ) {
            ctx.strokeStyle = 'rgba(255,255,255,0.07)';
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(margin.left, py);
            ctx.lineTo(margin.left + plotW, py);
            ctx.stroke();
        }
    }

    const scaleBarLen = 0.1e-3;
    const scaleBarPx = (scaleBarLen / rMax) * halfW;
    const scaleBarX = margin.left + 10;
    const scaleBarY = margin.top + plotH - 20;
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.fillRect(scaleBarX, scaleBarY, scaleBarPx, 3);
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText((scaleBarLen * 1000).toFixed(1) + ' mm', scaleBarX, scaleBarY - 5);

    renderColorbar(fmin, fmax, fieldName);
}

function renderColorbar(fmin, fmax, fieldName) {
    const ctx = colorbarCtx;
    const cw = colorbarCanvas.width;
    const ch = colorbarCanvas.height;
    const marginTop = 50;
    const marginBot = 50;

    ctx.fillStyle = '#0f1923';
    ctx.fillRect(0, 0, cw, ch);

    const barW = 22;
    const barH = ch - marginTop - marginBot;
    const barX = 8;

    for (let y = 0; y < barH; y++) {
        const t = 1 - y / barH;
        ctx.fillStyle = colormap(t, fieldName);
        ctx.fillRect(barX, marginTop + y, barW, 1);
    }

    ctx.strokeStyle = '#5c6b7a';
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, marginTop, barW, barH);

    const info = FIELD_INFO[fieldName] || { title: fieldName, unit: '', fmt: v => v.toFixed(3) };

    ctx.fillStyle = '#4fc3f7';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'left';

    const titleLines = info.title.split(' - ');
    titleLines.forEach((line, idx) => {
        ctx.fillText(line, barX, marginTop - 12 - (titleLines.length - 1 - idx) * 14);
    });

    ctx.fillStyle = '#b0bec5';
    ctx.font = '12px monospace';

    const nTicks = 6;
    for (let i = 0; i <= nTicks; i++) {
        const frac = i / nTicks;
        const val = fmax - frac * (fmax - fmin);
        const y = marginTop + frac * barH;

        ctx.strokeStyle = '#5c6b7a';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(barX + barW, y);
        ctx.lineTo(barX + barW + 4, y);
        ctx.stroke();

        let txt;
        const abs = Math.abs(val);
        if (abs === 0) {
            txt = '0';
        } else if (abs >= 1e6) {
            txt = (val / 1e6).toFixed(1) + 'M';
        } else if (abs >= 1e3) {
            txt = (val / 1e3).toFixed(1) + 'k';
        } else if (abs >= 1) {
            txt = val.toFixed(2);
        } else if (abs >= 0.01) {
            txt = val.toFixed(3);
        } else {
            txt = val.toExponential(1);
        }
        ctx.fillText(txt, barX + barW + 7, y + 4);
    }

    if (info.unit) {
        ctx.fillStyle = '#7a8fa6';
        ctx.font = '11px sans-serif';
        ctx.fillText('[' + info.unit + ']', barX, ch - marginBot + 20);
    }
}

function updateDiagnostics(diag) {
    if (!diag) return;

    const cflClass = diag.cfl < 0.5 ? 'good' : diag.cfl < 1.0 ? 'warn' : 'bad';
    const dpStr = engFormatPa(diag.pressure_drop);
    const tMin = (diag.T_min - 273.15).toFixed(0);
    const tMax = (diag.T_max - 273.15).toFixed(0);

    diagContainer.innerHTML = `
        <div class="diag-grid">
            <div class="diag-item">
                <span class="diag-label">CFL</span>
                <span class="diag-value ${cflClass}">${diag.cfl.toFixed(4)}</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">\u0394P</span>
                <span class="diag-value neutral">${dpStr}</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">|u| max</span>
                <span class="diag-value neutral">${engFormat(diag.u_max)} m/s</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">Mass</span>
                <span class="diag-value neutral">${diag.mass_polymer.toExponential(2)} kg</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">T range</span>
                <span class="diag-value neutral">${tMin}\u2013${tMax} \u00B0C</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">\u03B7 range</span>
                <span class="diag-value neutral">${engFormat(diag.eta_min)}\u2013${engFormat(diag.eta_max)}</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">\u03B1 range</span>
                <span class="diag-value neutral">${diag.alpha_min.toFixed(3)}\u2013${diag.alpha_max.toFixed(3)}</span>
            </div>
            <div class="diag-item">
                <span class="diag-label">dt used</span>
                <span class="diag-value neutral">${diag.dt_used.toExponential(2)} s</span>
            </div>
        </div>
    `;
}

function updatePlots(diag) {
    if (!diag) return;
    pressureHistory.push({ time: diag.time, dp: diag.pressure_drop });

    drawLinePlot('plot-pressure', pressureHistory.map(p => p.time), pressureHistory.map(p => p.dp), 'Time (s)', 'Pressure Drop', '#4fc3f7', 'Pa');
    drawLinePlot('plot-swell', null, diag.swell_ratios, 'z index', 'Swell Ratio', '#ff7043', '');

    if (diag.centerline_T) {
        const temps = diag.centerline_T.map(t => t - 273.15);
        drawLinePlot('plot-temperature', null, temps, 'z index', 'Temperature', '#66bb6a', '\u00B0C');
    }
}

function drawLinePlot(canvasId, xVals, yVals, xlabel, ylabel, color, unit) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width;
    const ch = canvas.height;

    ctx.fillStyle = '#0d1b2a';
    ctx.fillRect(0, 0, cw, ch);

    if (!yVals || yVals.length === 0) return;

    const margin = { top: 20, right: 20, bottom: 35, left: 70 };
    const pw = cw - margin.left - margin.right;
    const ph = ch - margin.top - margin.bottom;

    if (!xVals) {
        xVals = [];
        for (let i = 0; i < yVals.length; i++) xVals.push(i);
    }

    let xmin = Math.min(...xVals);
    let xmax = Math.max(...xVals);
    let ymin = Math.min(...yVals);
    let ymax = Math.max(...yVals);

    if (xmax === xmin) xmax = xmin + 1;
    if (ymax === ymin) { ymax = ymin + 1; ymin = ymin - 1; }

    const pad = (ymax - ymin) * 0.08;
    ymin -= pad;
    ymax += pad;

    function tx(x) { return margin.left + ((x - xmin) / (xmax - xmin)) * pw; }
    function ty(y) { return margin.top + (1 - (y - ymin) / (ymax - ymin)) * ph; }

    ctx.strokeStyle = '#1e3a5f';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const yv = ymin + (i / 4) * (ymax - ymin);
        const py = ty(yv);

        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(margin.left, py);
        ctx.lineTo(cw - margin.right, py);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#90a4ae';
        ctx.font = '11px monospace';
        ctx.textAlign = 'right';
        let label;
        const abs = Math.abs(yv);
        if (abs >= 1e6) {
            label = (yv / 1e6).toFixed(1) + 'M';
        } else if (abs >= 1e3) {
            label = (yv / 1e3).toFixed(1) + 'k';
        } else if (abs >= 1 || abs === 0) {
            label = yv.toFixed(1);
        } else if (abs >= 0.01) {
            label = yv.toFixed(3);
        } else {
            label = yv.toExponential(1);
        }
        ctx.fillText(label, margin.left - 6, py + 4);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < xVals.length; i++) {
        const px = tx(xVals[i]);
        const py = ty(yVals[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.fillStyle = '#90a4ae';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(xlabel, margin.left + pw / 2, ch - 8);

    ctx.save();
    ctx.translate(14, margin.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#90a4ae';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    const yLabel = unit ? ylabel + ' (' + unit + ')' : ylabel;
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    ctx.strokeStyle = '#2a4a6b';
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, pw, ph);
}

fieldCtx.fillStyle = '#0a1520';
fieldCtx.fillRect(0, 0, fieldCanvas.width, fieldCanvas.height);
fieldCtx.fillStyle = '#4fc3f7';
fieldCtx.font = '16px sans-serif';
fieldCtx.textAlign = 'center';
fieldCtx.fillText('Configure parameters and click Run Simulation', fieldCanvas.width / 2, fieldCanvas.height / 2 - 10);
fieldCtx.fillStyle = '#7a8fa6';
fieldCtx.font = '13px sans-serif';
fieldCtx.fillText('The simulation domain will appear here', fieldCanvas.width / 2, fieldCanvas.height / 2 + 16);
