const API_BASE = '';
let currentRunId = null;
let ws = null;
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
const diagText = document.getElementById('diagnostics-text');
const fieldSelect = document.getElementById('field-select');

btnRun.addEventListener('click', startSimulation);
btnPause.addEventListener('click', pauseSimulation);
btnStop.addEventListener('click', stopSimulation);
fieldSelect.addEventListener('change', () => { if (latestFrame) renderField(latestFrame); });
document.getElementById('show-contour').addEventListener('change', () => { if (latestFrame) renderField(latestFrame); });
document.getElementById('show-nozzle').addEventListener('change', () => { if (latestFrame) renderField(latestFrame); });

function getConfig() {
    return {
        nozzle_diameter: parseFloat(document.getElementById('nozzle_diameter').value) * 1e-3,
        nozzle_length: parseFloat(document.getElementById('nozzle_length').value) * 1e-3,
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

function colormap(val, min_v, max_v) {
    let t = (val - min_v) / Math.max(max_v - min_v, 1e-30);
    t = Math.max(0, Math.min(1, t));

    let r, g, b;
    if (t < 0.25) {
        const s = t / 0.25;
        r = 0; g = Math.floor(s * 255); b = 255;
    } else if (t < 0.5) {
        const s = (t - 0.25) / 0.25;
        r = 0; g = 255; b = Math.floor((1 - s) * 255);
    } else if (t < 0.75) {
        const s = (t - 0.5) / 0.25;
        r = Math.floor(s * 255); g = 255; b = 0;
    } else {
        const s = (t - 0.75) / 0.25;
        r = 255; g = Math.floor((1 - s) * 255); b = 0;
    }
    return `rgb(${r},${g},${b})`;
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
            if (v < fmin) fmin = v;
            if (v > fmax) fmax = v;
        }
    }

    if (Math.abs(fmax - fmin) < 1e-20) {
        fmax = fmin + 1;
    }

    const margin = 30;
    const plotW = cw - 2 * margin;
    const plotH = ch - 2 * margin;
    const halfW = plotW / 2;

    const cellW = halfW / nr;
    const cellH = plotH / nz;

    const centerX = margin + halfW;

    for (let i = 0; i < nr; i++) {
        for (let j = 0; j < nz; j++) {
            const v = data[i][j];
            const col = colormap(v, fmin, fmax);

            const x_right = centerX + i * cellW;
            const x_left = centerX - (i + 1) * cellW;
            const y = margin + (nz - 1 - j) * cellH;

            ctx.fillStyle = col;
            ctx.fillRect(x_right, y, cellW + 0.5, cellH + 0.5);
            ctx.fillRect(x_left, y, cellW + 0.5, cellH + 0.5);
        }
    }

    if (document.getElementById('show-nozzle').checked) {
        const nozzleRFrac = frame.nozzle_radius / (frame.r_centers[nr - 1] + (frame.r_centers[1] - frame.r_centers[0]) * 0.5);
        const nozzleRPx = nozzleRFrac * halfW;
        const nozzleZEnd = frame.nozzle_z_end;
        const zMin = frame.z_centers[0];
        const zMax = frame.z_centers[nz - 1];
        const zFrac = (nozzleZEnd - zMin) / (zMax - zMin);
        const nozzleYPx = margin + (1 - zFrac) * plotH;

        ctx.strokeStyle = '#ffab40';
        ctx.lineWidth = 2;

        ctx.beginPath();
        ctx.moveTo(centerX + nozzleRPx, margin);
        ctx.lineTo(centerX + nozzleRPx, nozzleYPx);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(centerX - nozzleRPx, margin);
        ctx.lineTo(centerX - nozzleRPx, nozzleYPx);
        ctx.stroke();

        ctx.strokeStyle = '#ffab40';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(margin, nozzleYPx);
        ctx.lineTo(cw - margin, nozzleYPx);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    if (document.getElementById('show-contour').checked && frame.contour_r && frame.contour_r.length > 0) {
        const rMax = frame.r_centers[nr - 1] + (frame.r_centers[1] - frame.r_centers[0]) * 0.5;
        const zMin = frame.z_centers[0];
        const zMax = frame.z_centers[nz - 1];
        const zRange = zMax - zMin;

        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;

        const points = [];
        for (let k = 0; k < frame.contour_r.length; k++) {
            const rFrac = frame.contour_r[k] / rMax;
            const zFrac = (frame.contour_z[k] - zMin) / zRange;
            const px = centerX + rFrac * halfW;
            const py = margin + (1 - zFrac) * plotH;
            points.push({ x: px, y: py, xm: centerX - rFrac * halfW });
        }

        points.sort((a, b) => a.y - b.y);

        if (points.length > 1) {
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            for (let k = 1; k < points.length; k++) {
                ctx.lineTo(points[k].x, points[k].y);
            }
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(points[0].xm, points[0].y);
            for (let k = 1; k < points.length; k++) {
                ctx.lineTo(points[k].xm, points[k].y);
            }
            ctx.stroke();
        }
    }

    ctx.fillStyle = '#7a8fa6';
    ctx.font = '10px monospace';
    ctx.fillText('r', cw - margin + 5, ch / 2);
    ctx.fillText('z', cw / 2 - 3, margin - 5);

    renderColorbar(fmin, fmax, fieldName);
}

function renderColorbar(fmin, fmax, label) {
    const ctx = colorbarCtx;
    const cw = colorbarCanvas.width;
    const ch = colorbarCanvas.height;
    const margin = 30;

    ctx.fillStyle = '#0a1520';
    ctx.fillRect(0, 0, cw, ch);

    const barW = 15;
    const barH = ch - 2 * margin;
    const barX = 5;

    for (let y = 0; y < barH; y++) {
        const t = 1 - y / barH;
        const val = fmin + t * (fmax - fmin);
        ctx.fillStyle = colormap(val, fmin, fmax);
        ctx.fillRect(barX, margin + y, barW, 1);
    }

    ctx.strokeStyle = '#2a4a6b';
    ctx.strokeRect(barX, margin, barW, barH);

    ctx.fillStyle = '#9ab';
    ctx.font = '9px monospace';

    const nTicks = 5;
    for (let i = 0; i <= nTicks; i++) {
        const frac = i / nTicks;
        const val = fmax - frac * (fmax - fmin);
        const y = margin + frac * barH;
        let txt;
        if (Math.abs(val) > 1e4 || (Math.abs(val) < 0.01 && val !== 0)) {
            txt = val.toExponential(1);
        } else {
            txt = val.toFixed(2);
        }
        ctx.fillText(txt, barX + barW + 3, y + 3);
    }

    ctx.fillStyle = '#4fc3f7';
    ctx.fillText(label, barX, margin - 8);
}

function updateDiagnostics(diag) {
    if (!diag) return;
    const lines = [
        `Step: ${diag.step}  t=${diag.time.toExponential(3)}s`,
        `CFL: ${diag.cfl.toFixed(4)}`,
        `Cap dt: ${diag.capillary_dt.toExponential(2)}`,
        `alpha: [${diag.alpha_min.toFixed(4)}, ${diag.alpha_max.toFixed(4)}]`,
        `T: [${(diag.T_min-273.15).toFixed(1)}, ${(diag.T_max-273.15).toFixed(1)}] C`,
        `eta: [${diag.eta_min.toExponential(2)}, ${diag.eta_max.toExponential(2)}]`,
        `|u|_max: ${diag.u_max.toExponential(3)} m/s`,
        `p: [${diag.p_min.toExponential(2)}, ${diag.p_max.toExponential(2)}]`,
        `dP: ${diag.pressure_drop.toExponential(3)} Pa`,
        `Mass: ${diag.mass_polymer.toExponential(4)} kg`,
    ];
    diagText.textContent = lines.join('\n');
}

function updatePlots(diag) {
    if (!diag) return;
    pressureHistory.push({ time: diag.time, dp: diag.pressure_drop });

    drawLinePlot('plot-pressure', pressureHistory.map(p => p.time), pressureHistory.map(p => p.dp), 'Time (s)', 'dP (Pa)', '#4fc3f7');
    drawLinePlot('plot-swell', null, diag.swell_ratios, 'z index', 'Swell ratio', '#ff7043');

    if (diag.centerline_T) {
        const temps = diag.centerline_T.map(t => t - 273.15);
        drawLinePlot('plot-temperature', null, temps, 'z index', 'T (C)', '#66bb6a');
    }
}

function drawLinePlot(canvasId, xVals, yVals, xlabel, ylabel, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const cw = canvas.width;
    const ch = canvas.height;

    ctx.fillStyle = '#0d1b2a';
    ctx.fillRect(0, 0, cw, ch);

    if (!yVals || yVals.length === 0) return;

    const margin = { top: 15, right: 15, bottom: 25, left: 55 };
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

    const pad = (ymax - ymin) * 0.05;
    ymin -= pad;
    ymax += pad;

    function tx(x) { return margin.left + ((x - xmin) / (xmax - xmin)) * pw; }
    function ty(y) { return margin.top + (1 - (y - ymin) / (ymax - ymin)) * ph; }

    ctx.strokeStyle = '#1e3a5f';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const yv = ymin + (i / 4) * (ymax - ymin);
        const py = ty(yv);
        ctx.beginPath();
        ctx.moveTo(margin.left, py);
        ctx.lineTo(cw - margin.right, py);
        ctx.stroke();

        ctx.fillStyle = '#7a8fa6';
        ctx.font = '8px monospace';
        let label;
        if (Math.abs(yv) > 1e4 || (Math.abs(yv) < 0.01 && yv !== 0)) {
            label = yv.toExponential(1);
        } else {
            label = yv.toFixed(2);
        }
        ctx.fillText(label, 2, py + 3);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < xVals.length; i++) {
        const px = tx(xVals[i]);
        const py = ty(yVals[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.fillStyle = '#7a8fa6';
    ctx.font = '9px sans-serif';
    ctx.fillText(xlabel, cw / 2 - 15, ch - 3);

    ctx.save();
    ctx.translate(10, ch / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(ylabel, -20, 0);
    ctx.restore();
}

fieldCtx.fillStyle = '#0a1520';
fieldCtx.fillRect(0, 0, fieldCanvas.width, fieldCanvas.height);
fieldCtx.fillStyle = '#4fc3f7';
fieldCtx.font = '14px sans-serif';
fieldCtx.fillText('Configure parameters and click Run', 100, 350);
