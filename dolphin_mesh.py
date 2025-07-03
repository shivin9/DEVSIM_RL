# Import necessary libraries
# FEniCSx is the core library for FEM simulations.
# UFL (Unified Form Language) is used to define the mathematical equations.
# mpi4py is required for parallel processing, a core feature of FEniCSx.
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import Function
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
import ufl
from mpi4py import MPI

# ====================================================================
# 1. Define Physical and Geometric Parameters
# ====================================================================
# We define all constants and parameters here for clarity and ease of modification.
# All units are converted to the base SI system (meters, Volts, Coulombs, etc.)
# to ensure consistency in the PDE calculations.

# --- Geometric Parameters ---
width_um = 2.0  # Width of the simulation domain in micrometers
height_um = 1.0 # Height of the simulation domain in micrometers
width_m = width_um * 1e-6  # Convert width to meters
height_m = height_um * 1e-6 # Convert height to meters

# --- Material and Physical Constants ---
q = 1.60217663e-19  # Elementary charge (Coulomb)
epsilon_0 = 8.85418782e-12 # Vacuum permittivity (F/m)
epsilon_si = 11.7         # Relative permittivity of Silicon
epsilon = epsilon_si * epsilon_0 # Permittivity of Silicon (F/m)
k_B = 1.380649e-23        # Boltzmann constant (J/K)
T = 300                   # Temperature (Kelvin)
V_T = k_B * T / q         # Thermal voltage (Volts), ~0.0259 V at 300K

# --- Doping Parameters ---
# Doping concentrations are defined in cm^-3 and converted to m^-3
NA_max_cm3 = 1e18  # Maximum acceptor concentration (p-type)
ND_max_cm3 = 1e18  # Maximum donor concentration (n-type)
NA_max_m3 = NA_max_cm3 * 1e6 # Convert to m^-3
ND_max_m3 = ND_max_cm3 * 1e6 # Convert to m^-3
n_i_cm3 = 1.0e10          # Intrinsic carrier concentration for Si at 300K
n_i_m3 = n_i_cm3 * 1e6    # Convert to m^-3

# --- Mesh Parameters ---
# Number of elements in each direction
nx = 80
ny = 40

# ====================================================================
# 2. Create Mesh and Define Boundaries
# ====================================================================

# The MPI communicator allows the code to run in parallel, though for this
# example, we'll run on a single processor.
comm = MPI.COMM_WORLD

# Create a 2D rectangular mesh
domain = mesh.create_rectangle(
    comm,
    points=[(0, 0), (width_m, height_m)],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

# CRITICAL: Compute mesh connectivity before defining boundary conditions
domain.topology.create_connectivity(1, 2)

# --- Define and Mark Boundaries ---
# We need to identify the different parts of the boundary to apply
# different physical conditions (e.g., voltage at contacts).

# Create functions to locate facets (edges in 2D) on each boundary
def top_boundary(x):
    return np.isclose(x[1], height_m)

def bottom_boundary(x):
    return np.isclose(x[1], 0)

def side_walls(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], width_m))

# Locate the facets on the mesh corresponding to our functions
top_facets = mesh.locate_entities_boundary(domain, 1, top_boundary)
bottom_facets = mesh.locate_entities_boundary(domain, 1, bottom_boundary)
side_facets = mesh.locate_entities_boundary(domain, 1, side_walls)

# Assign integer markers to each boundary part. This helps in applying
# boundary conditions later.
# 1: Top boundary (Anode)
# 2: Bottom boundary (Cathode)
# 3: Side walls (Insulating)
marked_facets = np.hstack([top_facets, bottom_facets, side_facets])
markers = np.hstack([
    np.full_like(top_facets, 1),
    np.full_like(bottom_facets, 2),
    np.full_like(side_facets, 3)
])
# Create a MeshTag object to store the boundary markers
facet_tag = mesh.meshtags(domain, 1, marked_facets, markers)


# ====================================================================
# 3. Set Up Individual Function Spaces
# ====================================================================
# Instead of a mixed element, we use three separate function spaces
# for each variable: psi, n, and p.

# Use the lowercase 'functionspace' factory
V_psi = fem.functionspace(domain, ("CG", 1)) # For potential
V_n = fem.functionspace(domain, ("CG", 1))   # For electron concentration
V_p = fem.functionspace(domain, ("CG", 1))   # For hole concentration


# ====================================================================
# 4. Define Initial Doping Profile (Pre-Optimization)
# ====================================================================
# Before we start optimizing, we'll create a simple, non-optimized
# p-n junction. The top half will be p-type, and the bottom half n-type.
# This will be replaced by the topology optimization variable later.

# Use a Discontinuous Galerkin space of order 0 for material properties.
Q = fem.functionspace(domain, ("DG", 0))

# Use the uppercase 'Function' constructor
N_A = Function(Q) # Acceptor concentration
N_D = Function(Q) # Donor concentration

# Find the cells in the top and bottom halves of the domain
top_cells = mesh.locate_entities(domain, 2, lambda x: x[1] > height_m / 2)
bottom_cells = mesh.locate_entities(domain, 2, lambda x: x[1] <= height_m / 2)

# Assign doping values to the cells
# Top half (p-type)
N_A.x.array[top_cells] = NA_max_m3
N_D.x.array[top_cells] = n_i_m3**2 / NA_max_m3 # Minority carriers
# Bottom half (n-type)
N_D.x.array[bottom_cells] = ND_max_m3
N_A.x.array[bottom_cells] = n_i_m3**2 / ND_max_m3 # Minority carriers

# The net doping C = N_D - N_A
C = N_D - N_A


# ====================================================================
# 5. Define Boundary Conditions
# ====================================================================
# These are the Dirichlet (fixed value) boundary conditions.
# We define them on the individual function spaces.

# --- Terminal Voltages ---
V_forward = 0.7  # Forward bias voltage
V_reverse = -5.0 # Reverse bias voltage

# --- Carrier concentrations at contacts (Ohmic, charge neutral) ---
# At the p-contact (Anode, top)
n_p_contact = n_i_m3**2 / NA_max_m3
p_p_contact = NA_max_m3
# At the n-contact (Cathode, bottom)
n_n_contact = ND_max_m3
p_n_contact = n_i_m3**2 / ND_max_m3

# --- Create the DirichletBC objects for FORWARD bias ---
# Potential at Anode (top, marker 1)
anode_dofs_psi = fem.locate_dofs_topological(V_psi, 1, top_facets)
bc_anode_psi_fwd = fem.dirichletbc(dolfinx.default_scalar_type(V_forward), anode_dofs_psi, V_psi)
# Potential at Cathode (bottom, marker 2)
cathode_dofs_psi = fem.locate_dofs_topological(V_psi, 1, bottom_facets)
bc_cathode_psi_fwd = fem.dirichletbc(dolfinx.default_scalar_type(0.0), cathode_dofs_psi, V_psi)

# Carriers at Anode (top, marker 1)
anode_dofs_n = fem.locate_dofs_topological(V_n, 1, top_facets)
bc_anode_n_fwd = fem.dirichletbc(dolfinx.default_scalar_type(n_p_contact), anode_dofs_n, V_n)
anode_dofs_p = fem.locate_dofs_topological(V_p, 1, top_facets)
bc_anode_p_fwd = fem.dirichletbc(dolfinx.default_scalar_type(p_p_contact), anode_dofs_p, V_p)

# Carriers at Cathode (bottom, marker 2)
cathode_dofs_n = fem.locate_dofs_topological(V_n, 1, bottom_facets)
bc_cathode_n_fwd = fem.dirichletbc(dolfinx.default_scalar_type(n_n_contact), cathode_dofs_n, V_n)
cathode_dofs_p = fem.locate_dofs_topological(V_p, 1, bottom_facets)
bc_cathode_p_fwd = fem.dirichletbc(dolfinx.default_scalar_type(p_n_contact), cathode_dofs_p, V_p)

bcs_forward = [bc_anode_psi_fwd, bc_cathode_psi_fwd, bc_anode_n_fwd, bc_anode_p_fwd, bc_cathode_n_fwd, bc_cathode_p_fwd]

# --- Create the DirichletBC objects for REVERSE bias ---
# Potential at Anode (top, marker 1)
bc_anode_psi_rev = fem.dirichletbc(dolfinx.default_scalar_type(V_reverse), anode_dofs_psi, V_psi)
# Cathode potential and carrier concentration BCs are the same, so we can reuse them.

bcs_reverse = [bc_anode_psi_rev, bc_cathode_psi_fwd, bc_anode_n_fwd, bc_anode_p_fwd, bc_cathode_n_fwd, bc_cathode_p_fwd]

print("--- Simulation Environment Setup Complete (API Corrected) ---")
print(f"Domain size: {width_um} um x {height_um} um")
print(f"Mesh: {nx}x{ny} elements")
print("Individual function spaces for (psi, n, p) created.")
print("Initial p-n junction doping profile defined.")
print("Boundary conditions for forward and reverse bias have been prepared.")
print("\nNext step: Define the weak form of the drift-diffusion equations.")

# ====================================================================
# 6. Define the Weak Form of the Drift-Diffusion Equations
# ====================================================================
# This is the core physics definition. We translate the PDEs into
# their integral (weak) form.

# --- Define solution and test functions ---
# Use the Function (uppercase) constructor for the variables that will hold the solution.
psi = Function(V_psi)
n = Function(V_n)
p = Function(V_p)

# The test functions are from the same spaces.
v_psi = ufl.TestFunction(V_psi)
v_n = ufl.TestFunction(V_n)
v_p = ufl.TestFunction(V_p)

# --- Define physical models ---
# For simplicity, we assume constant mobility and lifetime for now.
# In a more advanced model, these could depend on doping, temperature, etc.
mu_n = fem.Constant(domain, dolfinx.default_scalar_type(0.14))  # Electron mobility (m^2/V/s)
mu_p = fem.Constant(domain, dolfinx.default_scalar_type(0.045)) # Hole mobility (m^2/V/s)
tau_n = fem.Constant(domain, dolfinx.default_scalar_type(1e-7))   # Electron lifetime (s)
tau_p = fem.Constant(domain, dolfinx.default_scalar_type(1e-7))   # Hole lifetime (s)

# Einstein relation: D = V_T * mu
D_n = V_T * mu_n
D_p = V_T * mu_p

# Shockley-Read-Hall (SRH) Recombination Model R = (p*n - n_i^2) / (tau_n*(p+n_i) + tau_p*(n+n_i))
# We add a small epsilon to the denominator to prevent division by zero if n or p are zero.
srh_denom = tau_p * (n + n_i_m3) + tau_n * (p + n_i_m3)
R_srh = (p * n - n_i_m3**2) / (srh_denom + 1e-10)

# --- Define current densities ---
# Jn = n * mu_n * grad(psi) + D_n * grad(n)
# Jp = p * mu_p * grad(psi) - D_p * grad(p)
# We use ufl.grad for the gradient operator.
J_n = q * n * mu_n * ufl.grad(psi) + q * D_n * ufl.grad(n)
J_p = q * p * mu_p * ufl.grad(psi) - q * D_p * ufl.grad(p)

# --- Define the residual equations (Weak Forms) ---
# Each equation is written in the form F(u, v) = 0.
# We use ufl.inner for dot products and ufl.dx for the integration measure over the domain.

# F_psi: Weak form of Poisson's equation
# ∫ (ε * ∇ψ ⋅ ∇v_ψ) dx - ∫ q * (p - n + C) * v_ψ dx = 0
F_psi = ufl.inner(epsilon * ufl.grad(psi), ufl.grad(v_psi)) * ufl.dx \
      - q * (p - n + C) * v_psi * ufl.dx

# F_n: Weak form of the electron continuity equation
# ∫ (∇ ⋅ Jn) * v_n dx - ∫ q * R * v_n dx = 0
# After integration by parts: -∫ Jn ⋅ ∇v_n dx + ∫ (boundary_flux) ds - ∫ q * R * v_n dx = 0
# The boundary flux term is handled by the boundary conditions.
F_n = -ufl.inner(J_n, ufl.grad(v_n)) * ufl.dx \
    - q * R_srh * v_n * ufl.dx

# F_p: Weak form of the hole continuity equation
# ∫ (∇ ⋅ Jp) * v_p dx + ∫ q * R * v_p dx = 0
# After integration by parts: -∫ Jp ⋅ ∇v_p dx + ∫ (boundary_flux) ds + ∫ q * R * v_p dx = 0
F_p = -ufl.inner(J_p, ufl.grad(v_p)) * ufl.dx \
    + q * R_srh * v_p * ufl.dx

# --- Combine the residuals ---
# We are solving a coupled system, so we simply add the forms together.
F = F_psi + F_n + F_p

print("--- Weak Form Definitions Complete ---")
print("SRH recombination model defined.")
print("Electron and hole current densities defined.")
print("Residuals for Poisson, electron continuity, and hole continuity equations created.")
print("\nNext step: Set up the nonlinear solver and run the simulation.")

## ====================================================================
# 7. Set Up and Run the Simulation
# ====================================================================
# We implement the Gummel iterative method to solve the coupled system.
# This approach avoids using Mixed Elements, respecting the API constraints.

def solve_diode_system(applied_voltage, boundary_conditions, initial_guess=None):
    """
    Solves the drift-diffusion system for a given applied voltage using
    the Gummel iteration method.

    Args:
        applied_voltage (float): The voltage to apply at the anode.
        boundary_conditions (list): The list of dolfinx.fem.DirichletBC objects.
        initial_guess (tuple, optional): (psi, n, p) functions from a previous run.
                                         Defaults to None (calculates equilibrium guess).

    Returns:
        A tuple containing the solved functions: (psi, n, p).
    """
    print(f"\n--- Starting Gummel Iteration for V = {applied_voltage} V ---")

    # --- Create functions to hold the solution for this solve ---
    # Use the uppercase 'Function' constructor
    psi_sol = Function(V_psi, name="Potential")
    n_sol = Function(V_n, name="Electrons")
    p_sol = Function(V_p, name="Holes")

    # --- Set initial guess for the iteration ---
    if initial_guess is None:
        print("Calculating initial guess from thermal equilibrium...")
        # For the first solve, we use a thermal equilibrium approximation
        # Built-in potential
        V_bi = V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)
        
        # Interpolate initial potential from 0 to V_bi + V_applied
        # Note: Using a lambda function is a more direct way to define the expression
        psi_sol.interpolate(lambda x: V_bi * (1 - x[1]/height_m) + applied_voltage)

        # Use equilibrium relations for n and p, based on the interpolated potential
        n_sol.interpolate(lambda x: n_i_m3 * np.exp( (V_bi * (1 - x[1]/height_m) + applied_voltage) / V_T))
        p_sol.interpolate(lambda x: n_i_m3 * np.exp(-(V_bi * (1 - x[1]/height_m) + applied_voltage) / V_T))
    else:
        print("Using provided initial guess...")
        psi_sol.x.array[:] = initial_guess[0].x.array
        n_sol.x.array[:] = initial_guess[1].x.array
        p_sol.x.array[:] = initial_guess[2].x.array


    # --- Gummel Iteration Loop ---
    max_iterations = 100
    tolerance = 1e-5
    for i in range(max_iterations):
        print(f"Gummel Iteration: {i+1}/{max_iterations}")

        # Store solution from previous iteration to check for convergence
        psi_old = Function(V_psi)
        psi_old.x.array[:] = psi_sol.x.array
        
        # 1. Solve Poisson's equation (linear problem)
        # We use n and p from the previous step (n_sol, p_sol)
        psi_trial = ufl.TrialFunction(V_psi)
        a_psi = ufl.inner(epsilon * ufl.grad(psi_trial), ufl.grad(v_psi)) * ufl.dx
        L_psi = q * (p_sol - n_sol + C) * v_psi * ufl.dx
        
        # Define and solve the linear problem for psi
        problem_psi = fem.petsc.LinearProblem(a_psi, L_psi, bcs=boundary_conditions, u=psi_sol)
        problem_psi.solve()

        # 2. Solve continuity equations (nonlinear problems)
        # Now use the newly computed psi_sol
        
        # Electron continuity equation - must be defined with n_sol as the unknown
        J_n_gummel = q * n_sol * mu_n * ufl.grad(psi_sol) + q * D_n * ufl.grad(n_sol)
        R_srh_gummel_n = (p_sol * n_sol - n_i_m3**2) / (tau_p * (n_sol + n_i_m3) + tau_n * (p_sol + n_i_m3) + 1e-10)
        F_n_gummel = -ufl.inner(J_n_gummel, ufl.grad(v_n)) * ufl.dx - q * R_srh_gummel_n * v_n * ufl.dx
        problem_n = NonlinearProblem(F_n_gummel, n_sol, bcs=boundary_conditions)
        solver_n = NewtonSolver(comm, problem_n)
        solver_n.solve(n_sol)
        
        # Hole continuity equation - must be defined with p_sol as the unknown
        J_p_gummel = q * p_sol * mu_p * ufl.grad(psi_sol) - q * D_p * ufl.grad(p_sol)
        R_srh_gummel_p = (p_sol * n_sol - n_i_m3**2) / (tau_p * (n_sol + n_i_m3) + tau_n * (p_sol + n_i_m3) + 1e-10)
        F_p_gummel = -ufl.inner(J_p_gummel, ufl.grad(v_p)) * ufl.dx + q * R_srh_gummel_p * v_p * ufl.dx
        problem_p = NonlinearProblem(F_p_gummel, p_sol, bcs=boundary_conditions)
        solver_p = NewtonSolver(comm, problem_p)
        solver_p.solve(p_sol)

        # --- Convergence Check ---
        psi_update_norm = np.linalg.norm(psi_sol.x.array - psi_old.x.array)
        psi_norm = np.linalg.norm(psi_sol.x.array)
        relative_error = psi_update_norm / psi_norm
        print(f"  Relative error in potential: {relative_error:.4e}")
        if relative_error < tolerance:
            print(f"Gummel loop converged after {i+1} iterations.")
            break
    else: # This else belongs to the for loop, runs if loop finishes without break
        print("Warning: Gummel loop did not converge within max iterations.")
        
    return psi_sol, n_sol, p_sol


# ====================================================================
# 8. Run Simulations and Compute Currents
# ====================================================================

# --- Run Forward Bias Simulation ---
psi_fwd, n_fwd, p_fwd = solve_diode_system(V_forward, bcs_forward)

# --- Run Reverse Bias Simulation ---
# Use the forward bias solution as a good initial guess
psi_rev, n_rev, p_rev = solve_diode_system(V_reverse, bcs_reverse, initial_guess=(psi_fwd, n_fwd, p_fwd))

# --- Calculate Terminal Currents ---
# We integrate the total current density J_n + J_p over a contact.
# Let's use the bottom contact (cathode, marker 2).

# Re-define the current densities with the solved functions
J_n_fwd_sol = q * n_fwd * mu_n * ufl.grad(psi_fwd) + q * D_n * ufl.grad(n_fwd)
J_p_fwd_sol = q * p_fwd * mu_p * ufl.grad(psi_fwd) - q * D_p * ufl.grad(p_fwd)
J_total_fwd = J_n_fwd_sol + J_p_fwd_sol

J_n_rev_sol = q * n_rev * mu_n * ufl.grad(psi_rev) + q * D_n * ufl.grad(n_rev)
J_p_rev_sol = q * p_rev * mu_p * ufl.grad(psi_rev) - q * D_p * ufl.grad(p_rev)
J_total_rev = J_n_rev_sol + J_p_rev_sol

# Define the surface integral measure 'ds' over the marked boundaries
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
# The normal vector 'n_hat' to the boundary
n_hat = ufl.FacetNormal(domain)

# Integrate the normal component of the total current density
# We integrate over the cathode (marker 2)
# Current is often measured per unit length in 2D, so units are A/m
current_fwd_form = fem.form(ufl.dot(J_total_fwd, n_hat) * ds(2))
current_rev_form = fem.form(ufl.dot(J_total_rev, n_hat) * ds(2))

# Assemble the integrals
current_fwd = fem.assemble_scalar(current_fwd_form)
current_rev = fem.assemble_scalar(current_rev_form)

# Since current flowing out is negative, we take the absolute value.
current_fwd_abs = abs(current_fwd)
current_rev_abs = abs(current_rev)

print("\n--- Simulation Results ---")
print(f"Forward Current (at {V_forward} V): {current_fwd_abs:.4e} A/m")
print(f"Reverse Current (at {V_reverse} V): {current_rev_abs:.4e} A/m")
print(f"Rectification Ratio: {current_fwd_abs / current_rev_abs:.4e}")

print("\nNext step: Visualize the results and set up the topology optimization loop.")
