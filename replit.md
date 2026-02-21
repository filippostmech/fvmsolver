# FDM Extrusion CFD Solver

## Overview
High-fidelity 2D axisymmetric CFD solver for FDM (Fused Deposition Modeling) extrusion simulation.
Simulates TPU melt extruding through a nozzle and forming a free-surface jet/extrudate, using the Finite Volume Method (FVM) in axisymmetric r-z coordinates.

## Current State
- Fully operational web-based CFD solver
- All 5 milestones implemented (M1-M5)
- Real-time visualization via polling API
- Conservative FVM with proper axisymmetric geometric terms (2pi*r face areas, annular volumes)

## Architecture
```
/solver       - Core FVM solver (Python + Numba acceleration)
  grid.py     - Axisymmetric structured grid with 2pi*r geometric factors
  materials.py - TPU material properties and Carreau-Yasuda viscosity model
  fvm_solver.py - Full FVM solver engine with projection method
/backend
  server.py   - FastAPI server with REST API + WebSocket streaming
/frontend
  index.html  - Web UI with parameter controls
  style.css   - Dark theme styling
  app.js      - Canvas visualization + diagnostic plots
main.py       - Entry point (uvicorn on port 5000)
```

## Physics & Equations
- **Governing equations**: Incompressible Navier-Stokes (axisymmetric r-z form)
  - Continuity: 1/r d(r*ur)/dr + duz/dz = 0
  - r-momentum: rho*(dur/dt + u.grad(ur)) = -dp/dr + div(2*eta*D) - eta*ur/r^2 + F_st_r
  - z-momentum: rho*(duz/dt + u.grad(uz)) = -dp/dz + div(2*eta*D) + rho*g + F_st_z
  - Energy: rho*cp*(dT/dt + u.grad(T)) = div(k*grad(T))
- **Two-phase flow**: VOF method (alpha=1 polymer, alpha=0 air) with property blending
- **Viscosity**: Carreau-Yasuda model eta(gamma_dot) with Arrhenius temperature shift a(T)
- **Surface tension**: CSF (Continuum Surface Force) with curvature from alpha gradients
- **Boundary conditions**: Parabolic inlet velocity profile, no-slip walls, pressure outlet (p=0)

## Numerical Methods
- Cell-centered FVM on structured axisymmetric grid
- Conservative flux formulation using 2*pi*r face areas and pi*(r_e^2 - r_w^2) annular areas
- Upwind advection for momentum, energy, and VOF
- Projection method for pressure-velocity coupling with Jacobi pressure Poisson solver
- VOF compression term for interface sharpening
- Adaptive CFL-based timestep
- Hoop stress term (eta*ur/r^2) in radial momentum
- NaN/Inf clamping as safety net
- Empty-domain initialization: polymer enters from inlet boundary only (no pre-filling)

## Stability Constraints
- CFL condition: dt < CFL_max * min(dr,dz) / max(|u|)
- Capillary constraint disabled for web demo performance
- Explicit diffusion: limited by grid size and max viscosity (practical for small grids)

## User Preferences
- None recorded yet

## Recent Changes
- Fixed VOF initialization: domain starts empty, polymer enters from inlet boundary
- Removed capillary timestep constraint for practical simulation speed
- Updated default dt to 1e-5 and steps to 500 for visible polymer advancement
- Major UI overhaul: nozzle/print bed overlays, larger fonts, engineering notation, info bar
- Rewrote FVM solver with fully conservative axisymmetric discretization
- Replaced BiCGSTAB pressure solver with robust Jacobi iteration
- Added velocity and pressure clamping for stability
