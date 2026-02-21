import asyncio
import json
import uuid
import time
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import threading

app = FastAPI()

simulations: Dict[str, dict] = {}


class SimConfig(BaseModel):
    nozzle_diameter: float = 0.4e-3
    nozzle_length: float = 2.0e-3
    nozzle_bed_gap: float = 3.0e-3
    flow_rate: float = 5e-9
    T_nozzle: float = 493.15
    T_ambient: float = 298.15
    nr: int = 15
    nz: int = 30
    dt: float = 1e-5
    h_conv: float = 10.0
    sigma: float = 0.035
    eta_0: float = 3000.0
    eta_inf: float = 50.0
    lambda_cy: float = 0.1
    a_cy: float = 2.0
    n_cy: float = 0.35
    E_a: float = 40000.0
    c_alpha: float = 1.0
    gravity: float = -9.81
    n_steps: int = 8000
    frames_per_update: int = 80
    stretch_type: str = 'uniform'
    stretch_ratio: float = 1.5


def run_simulation_sync(run_id: str, config: dict, n_steps: int, frames_per_update: int):
    from solver.fvm_solver import CFDSolver

    solver_config = {
        'nozzle_diameter': config.get('nozzle_diameter', 0.4e-3),
        'nozzle_length': config.get('nozzle_length', 2.0e-3),
        'domain_z_ext': config.get('nozzle_bed_gap', 3.0e-3),
        'flow_rate': config.get('flow_rate', 5e-9),
        'T_nozzle': config.get('T_nozzle', 493.15),
        'T_ambient': config.get('T_ambient', 298.15),
        'nr': config.get('nr', 30),
        'nz': config.get('nz', 60),
        'dt': config.get('dt', 1e-6),
        'h_conv': config.get('h_conv', 10.0),
        'gravity': config.get('gravity', -9.81),
        'c_alpha': config.get('c_alpha', 1.0),
        'stretch_type': config.get('stretch_type', 'uniform'),
        'stretch_ratio': config.get('stretch_ratio', 1.5),
        'material': {
            'eta_0': config.get('eta_0', 3000.0),
            'eta_inf': config.get('eta_inf', 50.0),
            'lambda_cy': config.get('lambda_cy', 0.1),
            'a_cy': config.get('a_cy', 2.0),
            'n_cy': config.get('n_cy', 0.35),
            'E_a': config.get('E_a', 40000.0),
            'sigma': config.get('sigma', 0.035),
        }
    }

    solver = CFDSolver(solver_config)
    sim_data = simulations[run_id]
    sim_data['solver'] = solver
    sim_data['status'] = 'running'
    sim_data['frames'] = []
    sim_data['diagnostics_history'] = []

    try:
        for step_i in range(n_steps):
            if sim_data.get('stop_requested', False):
                sim_data['status'] = 'stopped'
                return

            while sim_data.get('paused', False):
                time.sleep(0.1)
                if sim_data.get('stop_requested', False):
                    sim_data['status'] = 'stopped'
                    return

            diag = solver.step()
            sim_data['diagnostics_history'].append(diag)
            sim_data['current_step'] = step_i + 1

            if (step_i + 1) % frames_per_update == 0 or step_i == 0:
                frame = solver.get_frame_data()
                frame['diagnostics'] = diag
                sim_data['frames'].append(frame)
                sim_data['latest_frame'] = frame

        sim_data['status'] = 'completed'
    except Exception as e:
        sim_data['status'] = 'error'
        sim_data['error'] = str(e)
        import traceback
        traceback.print_exc()


@app.post("/api/simulate")
async def start_simulation(config: SimConfig):
    run_id = str(uuid.uuid4())[:8]
    sim_config = config.model_dump()

    simulations[run_id] = {
        'config': sim_config,
        'status': 'initializing',
        'frames': [],
        'current_step': 0,
        'total_steps': config.n_steps,
        'paused': False,
        'stop_requested': False,
    }

    thread = threading.Thread(
        target=run_simulation_sync,
        args=(run_id, sim_config, config.n_steps, config.frames_per_update),
        daemon=True
    )
    thread.start()

    return {"run_id": run_id, "status": "started"}


@app.post("/api/pause/{run_id}")
async def pause_simulation(run_id: str):
    if run_id in simulations:
        simulations[run_id]['paused'] = not simulations[run_id].get('paused', False)
        return {"paused": simulations[run_id]['paused']}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/api/stop/{run_id}")
async def stop_simulation(run_id: str):
    if run_id in simulations:
        simulations[run_id]['stop_requested'] = True
        return {"status": "stopping"}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.get("/api/status/{run_id}")
async def get_status(run_id: str):
    if run_id in simulations:
        sim = simulations[run_id]
        return {
            "status": sim['status'],
            "current_step": sim.get('current_step', 0),
            "total_steps": sim.get('total_steps', 0),
            "paused": sim.get('paused', False),
            "error": sim.get('error', None),
            "num_frames": len(sim.get('frames', [])),
        }
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.get("/api/frame/{run_id}/{frame_idx}")
async def get_frame(run_id: str, frame_idx: int):
    if run_id in simulations:
        frames = simulations[run_id].get('frames', [])
        if 0 <= frame_idx < len(frames):
            return frames[frame_idx]
        return JSONResponse(status_code=404, content={"error": "frame not found"})
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.get("/api/latest/{run_id}")
async def get_latest(run_id: str):
    if run_id in simulations:
        sim = simulations[run_id]
        latest = sim.get('latest_frame', None)
        if latest:
            return {
                'frame': latest,
                'status': sim['status'],
                'current_step': sim.get('current_step', 0),
                'total_steps': sim.get('total_steps', 0),
            }
        return {
            'frame': None,
            'status': sim['status'],
            'current_step': sim.get('current_step', 0),
            'total_steps': sim.get('total_steps', 0),
        }
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.get("/api/diagnostics/{run_id}")
async def get_diagnostics(run_id: str):
    if run_id in simulations:
        hist = simulations[run_id].get('diagnostics_history', [])
        return {"diagnostics": hist[-100:]}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: str):
    await websocket.accept()

    if run_id not in simulations:
        await websocket.send_json({"error": "run not found"})
        await websocket.close()
        return

    last_frame_idx = -1
    try:
        while True:
            sim = simulations.get(run_id)
            if not sim:
                break

            frames = sim.get('frames', [])
            current_len = len(frames)

            if current_len > last_frame_idx + 1:
                for fi in range(last_frame_idx + 1, current_len):
                    await websocket.send_json({
                        'type': 'frame',
                        'frame_idx': fi,
                        'data': frames[fi],
                        'status': sim['status'],
                        'current_step': sim.get('current_step', 0),
                        'total_steps': sim.get('total_steps', 0),
                    })
                last_frame_idx = current_len - 1

            if sim['status'] in ('completed', 'error', 'stopped'):
                await websocket.send_json({
                    'type': 'done',
                    'status': sim['status'],
                    'error': sim.get('error', None),
                })
                break

            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
