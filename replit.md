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
- Cell-centered FVM on structured axisymmetric grid (uniform or stretched)
- Non-uniform grid support: per-cell dr_arr[i], dz_arr[j] spacing arrays
- Grid stretching: power-law radial (clusters near axis), two-sided power-law axial (clusters near nozzle exit z=0)
- Conservative flux formulation using 2*pi*r face areas and pi*(r_e^2 - r_w^2) annular areas
- All gradient/diffusion calculations use local cell-center distances for non-uniform grids
- Upwind advection for momentum, energy, and VOF
- Projection method for pressure-velocity coupling with SOR pressure Poisson solver
- VOF compression term for interface sharpening
- Adaptive CFL-based timestep (uses minimum cell spacing for CFL)
- Hoop stress term (eta*ur/r^2) in radial momentum
- NaN/Inf clamping as safety net
- Empty-domain initialization: polymer enters from inlet boundary only (no pre-filling)

## Stability Constraints
- CFL condition: dt < CFL_max * min(dr,dz) / max(|u|)
- Capillary constraint disabled for web demo performance
- Explicit diffusion: limited by grid size and max viscosity (practical for small grids)

## User Preferences
- Two-color VOF display: red for polymer, blue for air (not rainbow)
- Smooth contour lines preferred over jagged marching-squares output

## Recent Changes
- Non-uniform grid stretching: grid.py supports power-law stretching in r (cluster near axis) and z (cluster near nozzle exit)
- All Numba-accelerated FVM functions updated from scalar dr/dz to per-cell dr_arr[i]/dz_arr[j] spacing arrays
- Gradient calculations use cell-center distances: r_centers[i+1]-r_centers[i] for r, z_centers-based for z
- Diffusion coefficients use local face distances instead of uniform spacing
- Pressure Poisson solver uses non-uniform Laplacian stencil with dr_arr and dz_arr
- UI: Grid Type dropdown (Uniform/Stretched) and Stretch Ratio slider (1.0-3.0)
- Frontend renders non-uniform cells using r_faces/z_faces arrays for correct pixel mapping
- Frame data now includes r_faces and z_faces arrays for visualization
- Contour interpolation uses cell-center distances instead of uniform dr/dz
- Binary VOF colormap: solid red (polymer, alpha>=0.5) / solid blue (air, alpha<0.5) with sharp cutoff
- Smooth free surface contour using Catmull-Rom spline (cubic Bezier) interpolation
- Simulation runs stably with both uniform and stretched grids
- Velocity display masked in air region (alpha < 0.5) to show zero velocity outside polymer
- Colorbar title text wraps automatically to prevent truncation on narrow canvas
- Nozzle Exit / Extrudate labels auto-separate to prevent overlap in small-gap configurations
- Diagnostic plots filter NaN/Inf values before rendering
- Swell ratio calculation uses fallback (previous value or 1.0) when no polymer found at z-level
- Pressure drop computed from nozzle-radius-aligned cells at both inlet and outlet (polymer region only)
