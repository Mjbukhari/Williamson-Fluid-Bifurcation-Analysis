# Williamson Fluid Bifurcation Analysis

Numerical solver for the full governing equations of Williamson fluid in peristaltic flows, without lubrication approximation.

## Governing Equation

The solver implements the full stream function formulation:

F(Ψ) = δRe[∂C₁/∂y + δ²∂C₂/∂x] - (1+We)[δ⁴Ψ_xxxx + 2δ²Ψ_xxyy + Ψ_yyyy] = 0

where:
- C₁ = Ψ_y·Ψ_xy - Ψ_x·Ψ_yy
- C₂ = Ψ_y·Ψ_xx - Ψ_x·Ψ_xy
- Re = Reynolds number
- We = Weissenberg number (elasticity parameter)
- δ = Aspect ratio

## Numerical Method
- Newton's method with finite difference Jacobian
- Finite Difference Method (FDM) on transformed coordinates
- Parameter continuation for bifurcation tracking

## Files
- `solver.py` - DirectNavierStokesSolver class implementing the numerical scheme
- `solution_store.py` - SolutionStore class for managing results

## Requirements
numpy, scipy, matplotlib