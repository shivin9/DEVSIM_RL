"""
Modular Drift-Diffusion Semiconductor Device Simulator

A refactored implementation of the drift-diffusion solver with modular components
for better maintainability and extensibility.

Modules:
- physics: Physical models and constants
- numerics: Numerical stability and transformations  
- solvers: PDE solvers for Poisson and continuity equations
- geometry: Mesh generation and boundary conditions
- analysis: Current calculation and post-processing
"""

__version__ = "1.0.0"
__author__ = "DEVSIM Research Project"

# Main solver interface will be defined in main_solver.py