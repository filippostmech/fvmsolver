# FDM Extrusion CFD Solver

A high-fidelity 2D axisymmetric Computational Fluid Dynamics (CFD) solver for simulating Fused Deposition Modeling (FDM) extrusion. Built with the Finite Volume Method (FVM) in axisymmetric r-z coordinates, the solver models TPU polymer melt extruding through a heated nozzle, forming a free-surface jet, and depositing onto a print bed.

Includes a real-time web-based visualization interface for interactive parameter exploration.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![NumPy](https://img.shields.io/badge/NumPy-2.x-orange)
![Numba](https://img.shields.io/badge/Numba-JIT-red)

## Features

- **Full Navier-Stokes solver** in axisymmetric (r-z) coordinates with incompressibility constraint
- **Two-phase VOF** (Volume of Fluid) tracking of polymer/air interface with surface tension
- **Non-Newtonian viscosity**: Carreau-Yasuda model with Arrhenius temperature dependence
- **Heat transfer**: Energy equation with convective cooling at the polymer-air interface
- **Non-uniform grid stretching**: Power-law clustering near the axis and nozzle exit
- **Adaptive timestepping**: CFL, thermal diffusion, and per-cell viscous stability constraints
- **Real-time web UI**: Canvas-based visualization with interactive parameter controls
- **Numba JIT acceleration**: All inner loops compiled to machine code for performance

## Physics & Governing Equations

### Incompressible Navier-Stokes (Axisymmetric)

**Continuity:**

$$\frac{1}{r}\frac{\partial(r u_r)}{\partial r} + \frac{\partial u_z}{\partial z} = 0$$

**Radial momentum:**

$$\rho\left(\frac{\partial u_r}{\partial t} + \mathbf{u}\cdot\nabla u_r\right) = -\frac{\partial p}{\partial r} + \nabla\cdot(2\eta\mathbf{D}) - \frac{\eta u_r}{r^2} + F_{st,r}$$

**Axial momentum:**

$$\rho\left(\frac{\partial u_z}{\partial t} + \mathbf{u}\cdot\nabla u_z\right) = -\frac{\partial p}{\partial z} + \nabla\cdot(2\eta\mathbf{D}) + \rho g + F_{st,z}$$

**Energy:**

$$\rho c_p\left(\frac{\partial T}{\partial t} + \mathbf{u}\cdot\nabla T\right) = \nabla\cdot(k\nabla T)$$

### Constitutive Models

**Carreau-Yasuda viscosity:**

$$\eta(\dot{\gamma}, T) = a_T\left[\eta_\infty + (\eta_0 - \eta_\infty)\left(1 + (\lambda\dot{\gamma}_{eff})^a\right)^{(n-1)/a}\right]$$

where the Arrhenius temperature shift factor is:

$$a_T = \exp\left[\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right]$$

**VOF two-phase tracking:**

$$\frac{\partial\alpha}{\partial t} + \nabla\cdot(\alpha\mathbf{u}) = 0$$

with CSF (Continuum Surface Force) surface tension:

$$\mathbf{F}_{st} = \sigma\kappa\nabla\alpha$$

### Default Material: TPU (Thermoplastic Polyurethane)

| Property | Symbol | Default Value | Unit |
|----------|--------|--------------|------|
| Polymer density | rho_polymer | 1150 | kg/m^3 |
| Zero-shear viscosity | eta_0 | 3000 | Pa-s |
| Infinite-shear viscosity | eta_inf | 50 | Pa-s |
| Relaxation time | lambda | 0.1 | s |
| Yasuda parameter | a | 2.0 | - |
| Power-law index | n | 0.35 | - |
| Activation energy | E_a | 40000 | J/mol |
| Thermal conductivity (polymer) | k_polymer | 0.22 | W/(m-K) |
| Specific heat (polymer) | c_p_polymer | 1800 | J/(kg-K) |
| Surface tension | sigma | 0.035 | N/m |
| Air density | rho_air | 100 | kg/m^3 |
| Air thermal conductivity | k_air | 0.026 | W/(m-K) |
| Air specific heat | c_p_air | 1005 | J/(kg-K) |
| Air dynamic viscosity | mu_air | 1.8e-5 | Pa-s |

> **Note:** Air density is set to 100 kg/m^3 (elevated from true 1.2 kg/m^3) for numerical stability with the explicit scheme. This is a common practice in VOF simulations to reduce the density ratio and improve convergence.

## Numerical Methods

| Component | Method |
|-----------|--------|
| Spatial discretization | Cell-centered FVM on structured axisymmetric grid |
| Advection | First-order upwind |
| Pressure-velocity coupling | Projection method with SOR Poisson solver |
| VOF advection | Upwind with compression term for interface sharpening |
| Time integration | Explicit forward Euler with adaptive sub-stepping |
| Grid geometry | Conservative: 2*pi*r face areas, pi*(r_e^2 - r_w^2) annular volumes |

### Stability Constraints

The adaptive timestep enforces multiple stability criteria:

- **CFL (advective):** dt < CFL_max * min(dr, dz) / max(|u|)
- **Thermal diffusion:** dt < 0.25 * rho*c_p * dx_min^2 / k_max
- **Viscous diffusion:** Per-cell local sub-stepping: dt_local = rho * dx^2 / (4*eta)
- **Temperature safety clamp:** T_ambient - 5K <= T <= T_nozzle + 20K

### Grid Stretching

Non-uniform grids use power-law stretching to cluster cells where resolution matters most:

- **Radial:** Cells cluster near the symmetry axis (r=0) where velocity gradients are steepest
- **Axial:** Two-sided clustering near the nozzle exit (z=0) where die swell and free-surface deformation occur

## Project Structure

```
fvmsolver/
├── main.py                 # Entry point (uvicorn server on port 5000)
├── solver/
│   ├── __init__.py
│   ├── fvm_solver.py       # Core FVM solver engine (~1170 lines)
│   │                       #   - Momentum, energy, VOF, pressure equations
│   │                       #   - Numba-accelerated inner loops
│   │                       #   - Adaptive timestepping
│   │                       #   - Diagnostics (swell ratio, pressure drop)
│   ├── grid.py             # Axisymmetric structured grid
│   │                       #   - Uniform and power-law stretched grids
│   │                       #   - Cell volumes, face areas (2*pi*r geometric factors)
│   └── materials.py        # TPU material properties
│                           #   - Carreau-Yasuda + Arrhenius viscosity
│                           #   - Air/polymer property blending
├── backend/
│   ├── __init__.py
│   └── server.py           # FastAPI REST API + WebSocket server
│                           #   - POST /api/simulate - Launch simulation
│                           #   - GET /api/latest/{id} - Poll latest frame + status
│                           #   - GET /api/diagnostics/{id} - History data
│                           #   - POST /api/pause/{id}, /api/stop/{id}
│                           #   - WebSocket /ws/{id} - Real-time streaming
├── frontend/
│   ├── index.html          # Web UI with parameter input panels
│   ├── style.css           # Dark theme styling
│   └── app.js              # Canvas-based visualization (~900 lines)
│                           #   - Field rendering (VOF, velocity, temperature, pressure, viscosity)
│                           #   - Free-surface contour with Catmull-Rom spline smoothing
│                           #   - Diagnostic plots (pressure drop, swell ratio, temperature)
├── pyproject.toml          # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.11 or later
- A C compiler (for Numba LLVM JIT) - usually pre-installed on Linux/macOS

### Setup

```bash
git clone https://github.com/<your-username>/fvmsolver.git
cd fvmsolver

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
# Or using pyproject.toml:
pip install .
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 2.4.2 | Array operations, linear algebra |
| numba | >= 0.64.0 | JIT compilation of solver loops |
| scipy | >= 1.17.0 | Scientific computing utilities |
| fastapi | >= 0.129.0 | REST API server |
| uvicorn | >= 0.41.0 | ASGI server |
| websockets | >= 16.0 | Real-time data streaming |

Install via pip:

```bash
pip install -r requirements.txt
```

Or install directly from `pyproject.toml`:
```bash
pip install .
```

## Usage

### Running the Solver

```bash
python main.py
```

Open your browser at `http://localhost:5000`. The web interface allows you to:

1. Set nozzle geometry (diameter, length, bed gap)
2. Configure flow conditions (flow rate, temperatures)
3. Adjust material properties (viscosity model parameters)
4. Choose grid resolution and type (uniform/stretched)
5. Run, pause, and stop simulations
6. Visualize fields in real time (VOF, velocity, temperature, pressure, viscosity)
7. Monitor diagnostics (pressure drop, die swell ratio, centerline temperature)

### Using the Solver Programmatically

```python
from solver.fvm_solver import CFDSolver

config = {
    # Nozzle geometry
    'nozzle_diameter': 0.4e-3,     # 0.4 mm nozzle
    'nozzle_length': 2.0e-3,       # 2 mm nozzle length
    'domain_z_ext': 3.0e-3,        # Nozzle-to-bed gap (m)

    # Grid
    'nr': 30,                      # Radial grid cells
    'nz': 60,                      # Axial grid cells
    'dt': 1e-5,                    # Initial timestep (s)
    'stretch_type': 'stretched',   # 'uniform' or 'stretched'
    'stretch_ratio': 1.5,          # Grid stretch ratio (1.0 = uniform)

    # Flow conditions
    'flow_rate': 5e-9,             # Volumetric flow rate (m^3/s)
    'T_nozzle': 493.15,            # Nozzle temperature (K) = 220 C
    'T_ambient': 298.15,           # Ambient temperature (K) = 25 C
    'h_conv': 10.0,                # Convection coefficient (W/m^2-K)

    # Material (TPU) - passed as nested dict
    'material': {
        'eta_0': 3000.0,           # Zero-shear viscosity (Pa-s)
        'eta_inf': 50.0,           # Infinite-shear viscosity (Pa-s)
        'lambda_cy': 0.1,          # Relaxation time (s)
        'a_cy': 2.0,               # Yasuda parameter
        'n_cy': 0.35,              # Power-law index
        'E_a': 40000.0,            # Activation energy (J/mol)
        'sigma': 0.035,            # Surface tension (N/m)
    },
}

solver = CFDSolver(config)

for step in range(1000):
    diagnostics = solver.step()
    if step % 100 == 0:
        print(f"Step {diagnostics['step']}: "
              f"t={diagnostics['time']:.3e} s, "
              f"CFL={diagnostics['cfl']:.4f}, "
              f"T=[{diagnostics['T_min']:.1f}, {diagnostics['T_max']:.1f}] K, "
              f"dP={diagnostics['pressure_drop']:.0f} Pa")

# Access field data
frame = solver.get_frame_data()
# frame['alpha']  - VOF field (nr x nz)
# frame['u_mag']  - Velocity magnitude (masked in air)
# frame['T']      - Temperature field
# frame['p']      - Pressure field
# frame['eta']    - Viscosity field
# frame['contour_r'], frame['contour_z'] - Free surface contour points
```

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/simulate` | POST | Start a simulation with given config, returns `run_id` |
| `/api/status/{run_id}` | GET | Get simulation status (step count, paused state) |
| `/api/latest/{run_id}` | GET | Get the most recent frame + diagnostics |
| `/api/frame/{run_id}/{idx}` | GET | Get a specific saved frame by index |
| `/api/diagnostics/{run_id}` | GET | Get last 100 diagnostic snapshots |
| `/api/pause/{run_id}` | POST | Toggle pause/resume |
| `/api/stop/{run_id}` | POST | Stop and clean up a simulation |
| `/ws/{run_id}` | WebSocket | Real-time frame streaming |

## Visualization

The web UI provides five field visualizations:

| Field | Colormap | Description |
|-------|----------|-------------|
| VOF (alpha) | Red/Blue binary | Polymer (red, alpha >= 0.5) vs Air (blue, alpha < 0.5) |
| Velocity magnitude | Viridis | Flow speed in m/s (masked to zero in air region) |
| Temperature | Inferno | Temperature field in Kelvin |
| Pressure | Viridis | Pressure field in Pascals |
| Viscosity | Plasma | Effective viscosity in Pa-s |

Additional overlays:
- **Free surface contour**: Catmull-Rom spline-smoothed alpha=0.5 isoline
- **Nozzle geometry**: Rendered with walls, exit lip, and flow direction arrow
- **Print bed**: Shown at domain bottom with gradient shading
- **Scale bar**: Physical scale reference in mm

Diagnostic plots (updated each frame):
- Pressure drop vs time
- Die swell ratio vs axial position
- Centerline temperature vs axial position

## Configuration Guide

### Grid Resolution

For quick exploration, use the defaults (15 x 30). For publication-quality results:

| Use Case | Nr x Nz | Grid Type | Notes |
|----------|---------|-----------|-------|
| Quick test | 15 x 30 | Uniform | ~35 steps/sec |
| Standard | 30 x 60 | Uniform | Good balance |
| High fidelity | 50 x 100 | Stretched (1.5) | Resolve die swell |
| Convergence study | 80 x 160 | Stretched (2.0) | Research quality |

### Timestep Selection

The solver automatically reduces dt below your input value when stability requires it. As a starting point:

- **dt = 1e-5 s**: Good for most cases with default material
- **dt = 1e-6 s**: More conservative, use for high flow rates or fine grids
- The CFL number in the status panel should stay below ~0.5

### Typical TPU Parameters

| TPU Grade | eta_0 (Pa-s) | eta_inf (Pa-s) | n | lambda (s) |
|-----------|-------------|----------------|---|-----------|
| Soft (80A) | 1500 | 30 | 0.35 | 0.08 |
| Medium (90A) | 3000 | 50 | 0.35 | 0.10 |
| Hard (55D) | 6000 | 80 | 0.30 | 0.15 |

## Known Limitations

- **2D axisymmetric only**: Cannot capture 3D effects like layer-to-layer deposition or toolpath geometry
- **Explicit time integration**: The forward Euler scheme limits timestep size, especially for high-viscosity materials on fine grids
- **No viscoelasticity**: The Carreau-Yasuda model captures shear-thinning but not elastic effects (normal stress differences, extrudate memory)
- **Simplified free surface**: VOF with CSF captures the interface but does not model contact angle dynamics at the bed
- **No solidification model**: The polymer does not solidify upon cooling; viscosity increases via the Arrhenius shift but the material remains fluid
- **Single nozzle geometry**: Only straight cylindrical nozzles are supported (no conical or compound geometries)

## Contributing

Contributions are welcome. Some areas where help would be valuable:

- **Implicit viscous diffusion**: Would allow larger timesteps for high-viscosity materials
- **Higher-order advection**: QUICK or TVD schemes for momentum and energy
- **Viscoelastic constitutive models**: Oldroyd-B, PTT, or Giesekus models for elastic die swell prediction
- **3D extension**: Layer-by-layer deposition with toolpath input
- **Solidification/crystallization**: Phase change modeling for semi-crystalline polymers
- **Adaptive mesh refinement**: Concentrate resolution at the free surface dynamically
- **GPU acceleration**: Port Numba kernels to CUDA for larger grids

## References

1. Brackbill, J. U., Kothe, D. B., & Zemach, C. (1992). A continuum method for modeling surface tension. *Journal of Computational Physics*, 100(2), 335-354.
2. Carreau, P. J. (1972). Rheological equations from molecular network theories. *Transactions of the Society of Rheology*, 16(1), 99-127.
3. Yasuda, K. (1979). Investigation of the analogies between viscometric and linear viscoelastic properties of polystyrene fluids. *PhD thesis, MIT*.
4. Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing.
5. Hirt, C. W., & Nichols, B. D. (1981). Volume of fluid (VOF) method for the dynamics of free boundaries. *Journal of Computational Physics*, 39(1), 201-225.
6. Comminal, R., Serdeczny, M. P., Pedersen, D. B., & Spangenberg, J. (2018). Numerical modeling of the strand deposition flow in extrusion-based additive manufacturing. *Additive Manufacturing*, 20, 68-76.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
