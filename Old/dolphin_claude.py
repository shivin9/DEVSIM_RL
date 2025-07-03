# Import necessary libraries
import dolfinx
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import Function, Constant
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt
from petsc4py import PETSc

# ====================================================================
# 1. Define Physical and Geometric Parameters
# ====================================================================
# Physical constants
q = 1.60217663e-19  # Elementary charge (C)
epsilon_0 = 8.85418782e-12 # Vacuum permittivity (F/m)
epsilon_si = 11.7         # Relative permittivity of Silicon
epsilon = epsilon_si * epsilon_0 # Permittivity of Silicon (F/m)
k_B = 1.380649e-23        # Boltzmann constant (J/K)
T = 300                   # Temperature (K)
V_T = k_B * T / q         # Thermal voltage (V), ~0.0259 V at 300K

# Geometric parameters
width_um = 2.0
height_um = 1.0
width_m = width_um * 1e-6
height_m = height_um * 1e-6

# Doping parameters
NA_max_cm3 = 5e16
ND_max_cm3 = 5e16
NA_max_m3 = NA_max_cm3 * 1e6
ND_max_m3 = ND_max_cm3 * 1e6
n_i_cm3 = 1.0e10
n_i_m3 = n_i_cm3 * 1e6

# Mobility parameters (m²/V/s)
mu_n0 = 0.14  # Electron mobility
mu_p0 = 0.045  # Hole mobility

# SRH Recombination parameters
tau_n0 = 1e-6  # Electron lifetime (s)
tau_p0 = 1e-6  # Hole lifetime (s)
E_trap = 0.0   # Trap energy level relative to intrinsic (eV)

# Mesh parameters
nx = 40
ny = 20

print(f"Physical parameters:")
print(f"V_T = {V_T:.4f} V")
print(f"Built-in potential estimate: {V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2):.4f} V")

# ====================================================================
# 2. Create Mesh and Define Boundaries
# ====================================================================
comm = MPI.COMM_WORLD
domain = mesh.create_rectangle(
    comm,
    points=[(0, 0), (width_m, height_m)],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

# Create connectivity for all dimensions
domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim-1)
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)

# Define boundaries
def top_boundary(x):
    return np.isclose(x[1], height_m)

def bottom_boundary(x):
    return np.isclose(x[1], 0)

def side_walls(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], width_m))

# Locate boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

top_facets = mesh.locate_entities_boundary(domain, fdim, top_boundary)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_boundary)
side_facets = mesh.locate_entities_boundary(domain, fdim, side_walls)

# Create facet tags
marked_facets = np.hstack([top_facets, bottom_facets, side_facets])
markers = np.hstack([
    np.full_like(top_facets, 1),
    np.full_like(bottom_facets, 2),
    np.full_like(side_facets, 3)
])

# Sort the arrays to ensure proper indexing
sorted_indices = np.argsort(marked_facets)
marked_facets = marked_facets[sorted_indices]
markers = markers[sorted_indices]

facet_tag = mesh.meshtags(domain, fdim, marked_facets, markers)

# Create measure for boundary integrals
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# ====================================================================
# 3. Set Up Function Spaces and Doping Profile
# ====================================================================
# Create function spaces for each variable
V = fem.functionspace(domain, ("CG", 1))  # For potential, n, and p

# Doping profile
Q = fem.functionspace(domain, ("DG", 0))
N_A = Function(Q)
N_D = Function(Q)

# Simple doping profile - top half p-type, bottom half n-type
def doping_na(x):
    return np.where(x[1] > height_m / 2, NA_max_m3, n_i_m3**2 / ND_max_m3)

def doping_nd(x):
    return np.where(x[1] > height_m / 2, n_i_m3**2 / NA_max_m3, ND_max_m3)

N_A.interpolate(doping_na)
N_D.interpolate(doping_nd)

# Create net doping as a Function
C = Function(Q)
C.x.array[:] = N_D.x.array[:] - N_A.x.array[:]

print("Doping profile created successfully")

# ====================================================================
# 4. Helper Functions for Stability
# ====================================================================
def bernoulli(x):
    """Bernoulli function B(x) = x/(exp(x)-1) for Scharfetter-Gummel"""
    # Use conditional to avoid numerical issues
    small_x = 1e-10
    return ufl.conditional(ufl.gt(abs(x), small_x), 
                          x / (ufl.exp(x) - 1), 
                          1 - x/2 + x**2/12 - x**4/720)

def avg_bernoulli(x):
    """Average Bernoulli function for flux calculation"""
    return 0.5 * (bernoulli(x) + bernoulli(-x))

# ====================================================================
# 5. SRH Recombination Model
# ====================================================================
# def srh_recombination(n, p, n_i):
#     """Shockley-Read-Hall recombination rate"""
#     # Trap concentration factors
#     n1 = n_i * np.exp(E_trap / V_T)
#     p1 = n_i * np.exp(-E_trap / V_T)
    
#     # SRH recombination rate with stabilization
#     denominator = tau_p0 * (n + n1) + tau_n0 * (p + p1)
#     R_srh = (n * p - n_i**2) / (denominator + 1e-30)  # Add small value for stability
#     return R_srh

def srh_recombination(n, p):
    """SRH recombination rate"""
    denominator = tau_p0 * (n + n_i_m3) + tau_n0 * (p + n_i_m3)
    R_srh = (n * p - n_i_m3**2) / (denominator + 1e-30)
    return R_srh

# ====================================================================
# 6. Improved Gummel Iteration Solver
# ====================================================================
# def gummel_solver(applied_voltage, psi_init=None, n_init=None, p_init=None, max_iterations=50, tolerance=1e-6):
    """
    Solve drift-diffusion equations using Gummel iteration with improved stability
    """
    print(f"\n--- Gummel Solver: V_app = {applied_voltage:.3f} V ---")
    
    # Initialize solution variables
    psi = Function(V, name="Potential")
    n = Function(V, name="Electron_Density")
    p = Function(V, name="Hole_Density")
    
    # Use previous solution or create initial guess
    if psi_init is not None:
        psi.x.array[:] = psi_init.x.array[:]
    else:
        # Initial guess based on equilibrium
        V_bi = V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)
        def init_potential(x):
            # Linear potential drop with built-in voltage
            return V_bi * (1 - x[1] / height_m) + applied_voltage * x[1] / height_m
        psi.interpolate(init_potential)
    
    # Initialize carrier concentrations
    if n_init is not None and p_init is not None:
        n.x.array[:] = n_init.x.array[:]
        p.x.array[:] = p_init.x.array[:]
    else:
        def n_init_func(x):
            y = x[1]
            n_eq = np.where(y > height_m / 2, 
                            n_i_m3**2 / NA_max_m3,  # p-type region
                            ND_max_m3)              # n-type region
            return n_eq
        
        def p_init_func(x):
            y = x[1]
            p_eq = np.where(y > height_m / 2,
                            NA_max_m3,              # p-type region
                            n_i_m3**2 / ND_max_m3)  # n-type region
            return p_eq
        
        n.interpolate(n_init_func)
        p.interpolate(p_init_func)
    
    # Ensure positive carrier concentrations
    n.x.array[:] = np.maximum(n.x.array[:], 1e10)  # Minimum carrier density
    p.x.array[:] = np.maximum(p.x.array[:], 1e10)
    
    # Boundary conditions
    anode_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
    cathode_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
    
    # Potential BCs
    bc_psi_anode = fem.dirichletbc(default_scalar_type(applied_voltage), anode_dofs, V)
    bc_psi_cathode = fem.dirichletbc(default_scalar_type(0.0), cathode_dofs, V)
    bcs_psi = [bc_psi_anode, bc_psi_cathode]
    
    # Carrier concentration BCs (equilibrium values with bias)
    # At anode (p-type contact)
    n_anode_val = n_i_m3**2 / NA_max_m3 * np.exp(applied_voltage / V_T)
    p_anode_val = NA_max_m3
    
    # At cathode (n-type contact)
    n_cathode_val = ND_max_m3
    p_cathode_val = n_i_m3**2 / ND_max_m3
    
    bc_n_anode = fem.dirichletbc(default_scalar_type(n_anode_val), anode_dofs, V)
    bc_n_cathode = fem.dirichletbc(default_scalar_type(n_cathode_val), cathode_dofs, V)
    bcs_n = [bc_n_anode, bc_n_cathode]
    
    bc_p_anode = fem.dirichletbc(default_scalar_type(p_anode_val), anode_dofs, V)
    bc_p_cathode = fem.dirichletbc(default_scalar_type(p_cathode_val), cathode_dofs, V)
    bcs_p = [bc_p_anode, bc_p_cathode]
    
    # Define variational problems
    v = ufl.TestFunction(V)
    
    # Gummel iteration
    converged = False
    for iteration in range(max_iterations):
        # Store old values for convergence check
        psi_old = Function(V)
        psi_old.x.array[:] = psi.x.array[:]
        n_old = Function(V)
        n_old.x.array[:] = n.x.array[:]
        p_old = Function(V)
        p_old.x.array[:] = p.x.array[:]
        
        # Step 1: Solve Poisson equation
        psi_trial = ufl.TrialFunction(V)
        a_psi = ufl.inner(epsilon * ufl.grad(psi_trial), ufl.grad(v)) * ufl.dx
        L_psi = q * (p - n + C) * v * ufl.dx
        
        try:
            problem_psi = LinearProblem(a_psi, L_psi, bcs=bcs_psi)
            psi_new = problem_psi.solve()
            psi.x.array[:] = psi_new.x.array[:]
        except Exception as e:
            print(f"  Warning: Poisson solve failed: {e}")
            print(f"  Trying to continue with current solution")
        
        # Step 2: Solve electron continuity equation with improved formulation
        n_trial = ufl.TrialFunction(V)
        
        # Diffusion coefficient
        D_n = mu_n0 * V_T
        
        # Standard drift-diffusion formulation
        a_n = (D_n * ufl.inner(ufl.grad(n_trial), ufl.grad(v)) * ufl.dx
               + mu_n0 * ufl.inner(n_trial * ufl.grad(psi), ufl.grad(v)) * ufl.dx
               + srh_recombination(n_trial, p, n_i_m3) * v * ufl.dx)
        
        L_n = Constant(domain, default_scalar_type(0.0)) * v * ufl.dx
        
        try:
            # problem_n = LinearProblem(a_n, L_n, bcs=bcs_n)
            # n_new = problem_n.solve()

            # F_u is the nonlinear weak form for the electron continuity equation
            problem_u = NonlinearProblem(F_u, u, bcs=bc_u)
            solver_u = NewtonSolver(comm, problem_u)
            solver_u.solve(u) # Correctly solves the nonlinear system

            n.x.array[:] = n_new.x.array[:]
            # Ensure positivity
            n.x.array[:] = np.maximum(n.x.array[:], 1e10)
        except Exception as e:
            print(f"  Warning: Electron continuity solve failed: {e}")
            print(f"  Trying to continue with current solution")
        
        # Step 3: Solve hole continuity equation
        p_trial = ufl.TrialFunction(V)
        
        # Diffusion coefficient
        D_p = mu_p0 * V_T
        
        # Standard drift-diffusion formulation (note sign difference for holes)
        a_p = (D_p * ufl.inner(ufl.grad(p_trial), ufl.grad(v)) * ufl.dx
               - mu_p0 * ufl.inner(p_trial * ufl.grad(psi), ufl.grad(v)) * ufl.dx
               + srh_recombination(n, p_trial, n_i_m3) * v * ufl.dx)
        
        L_p = Constant(domain, default_scalar_type(0.0)) * v * ufl.dx
        
        try:
            problem_p = LinearProblem(a_p, L_p, bcs=bcs_p)
            p_new = problem_p.solve()
            p.x.array[:] = p_new.x.array[:]
            # Ensure positivity
            p.x.array[:] = np.maximum(p.x.array[:], 1e10)
        except Exception as e:
            print(f"  Warning: Hole continuity solve failed: {e}")
            print(f"  Trying to continue with current solution")

        
        # Check convergence
        psi_norm = np.linalg.norm(psi.x.array)
        n_norm = np.linalg.norm(n.x.array)
        p_norm = np.linalg.norm(p.x.array)
        
        psi_change = np.linalg.norm(psi.x.array - psi_old.x.array) / (psi_norm + 1e-30)
        n_change = np.linalg.norm(n.x.array - n_old.x.array) / (n_norm + 1e-30)
        p_change = np.linalg.norm(p.x.array - p_old.x.array) / (p_norm + 1e-30)
        
        max_change = max(psi_change, n_change, p_change)
        
        if iteration % 5 == 0 or max_change < tolerance:
            print(f"  Iteration {iteration+1}: ψ_change={psi_change:.2e}, "
                  f"n_change={n_change:.2e}, p_change={p_change:.2e}")
        
        if max_change < tolerance:
            print(f"  ✓ Converged after {iteration+1} iterations")
            converged = True
            break
    
    if not converged:
        print(f"  ⚠ Did not converge after {max_iterations} iterations")
    
    return psi, n, p, converged

def gummel_solver(applied_voltage, bcs_psi, bcs_u, bcs_v,
                    psi_init=None, u_init=None, v_init=None,
                    max_iterations=50, tolerance=1e-6):
    """
    Solves the coupled drift-diffusion system using the robust Slotboom 
    variable formulation and a Gummel iterative map. This function is
    a numerically stable replacement for the original solver.

    This method is equivalent to the Scharfetter-Gummel scheme.

    Args:
        applied_voltage (float): The voltage applied at the anode.
        bcs_psi (list): Boundary conditions for the potential (psi).
        bcs_u (list): Boundary conditions for the electron Slotboom variable (u).
        bcs_v (list): Boundary conditions for the hole Slotboom variable (v).
        psi_init (Function, optional): Initial guess for psi.
        u_init (Function, optional): Initial guess for u.
        v_init (Function, optional): Initial guess for v.
        max_iterations (int): Maximum number of Gummel iterations.
        tolerance (float): Convergence tolerance.

    Returns:
        Tuple[Function, Function, Function, bool]: The solved potential, u, v,
                                                   and a boolean indicating convergence.
    """
    print(f"\n--- Slotboom Solver: V_app = {applied_voltage:.3f} V ---")

    # Define function spaces and test functions inside the solver for clarity
    V = fem.functionspace(domain, ("CG", 1))
    v_psi, v_u, v_v = ufl.TestFunction(V), ufl.TestFunction(V), ufl.TestFunction(V)
    
    # Initialize solution variables
    psi = Function(V, name="Potential")
    u = Function(V, name="Slotboom_u")
    v = Function(V, name="Slotboom_v")

    # Use previous solution or create an initial guess from equilibrium
    if psi_init is not None and u_init is not None and v_init is not None:
        psi.x.array[:], u.x.array[:], v.x.array[:] = psi_init.x.array, u_init.x.array, v_init.x.array
    else:
        # Equilibrium initial guess
        V_bi = V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)
        psi.interpolate(lambda x: V_bi * (1 - x[1]/height_m) + applied_voltage * x[1]/height_m)
        u.interpolate(lambda x: 1.0 + 0*x[0])  # u = 1 at equilibrium
        v.interpolate(lambda x: 1.0 + 0*x[0])  # v = 1 at equilibrium
    
    # Diffusion coefficients
    D_n = mu_n0 * V_T
    D_p = mu_p0 * V_T

    # --- Main Gummel Iteration Loop ---
    for k in range(max_iterations):
        psi_old = Function(V); psi_old.x.array[:] = psi.x.array

        # --- Step 1: Solve Poisson Equation (Linear) ---
        # Define n and p based on the last iteration's solution
        n_k = n_i_m3 * u * ufl.exp(psi / V_T)
        p_k = n_i_m3 * v * ufl.exp(-psi / V_T)
        
        psi_trial = ufl.TrialFunction(V)
        a_psi = ufl.inner(epsilon * ufl.grad(psi_trial), ufl.grad(v_psi)) * ufl.dx
        L_psi = q * (p_k - n_k + C) * v_psi * ufl.dx
        
        problem_psi = LinearProblem(a_psi, L_psi, bcs=bcs_psi, u=psi)
        problem_psi.solve()
        
        # --- Step 2: Solve Electron Continuity Equation (Nonlinear) ---
        # This equation solves for the updated 'u'
        exp_psi = ufl.exp(psi / V_T)
        n_new = n_i_m3 * u * exp_psi
        p_k = n_i_m3 * v * ufl.exp(-psi / V_T) # Use previous v
        
        R = srh_recombination(n_new, p_k)
        
        # Current density for electrons in terms of u
        J_n = q * D_n * n_i_m3 * exp_psi * ufl.grad(u)
        # Weak form: Integral(Jn . grad(v_u) - q*R*v_u)dx = 0
        F_u = ufl.inner(J_n, ufl.grad(v_u)) * ufl.dx - q * R * v_u * ufl.dx
        
        problem_u = NonlinearProblem(F_u, u, bcs=bcs_u)
        solver_u = NewtonSolver(comm, problem_u)
        solver_u.relaxation_parameter = 0.7 # Add relaxation for stability
        solver_u.solve(u)
        
        # --- Step 3: Solve Hole Continuity Equation (Nonlinear) ---
        # This equation solves for the updated 'v'
        exp_neg_psi = ufl.exp(-psi / V_T)
        n_k = n_i_m3 * u * ufl.exp(psi / V_T) # Use newly updated u
        p_new = n_i_m3 * v * exp_neg_psi
        
        R = srh_recombination(n_k, p_new)

        # Current density for holes in terms of v (note the negative sign)
        J_p = -q * D_p * n_i_m3 * exp_neg_psi * ufl.grad(v)
        # Weak form: Integral(Jp . grad(v_v) + q*R*v_v)dx = 0
        F_v = ufl.inner(J_p, ufl.grad(v_v)) * ufl.dx + q * R * v_v * ufl.dx
        
        problem_v = NonlinearProblem(F_v, v, bcs=bcs_v)
        solver_v = NewtonSolver(comm, problem_v)
        solver_v.relaxation_parameter = 0.7 # Add relaxation for stability
        solver_v.solve(v)

        # --- Convergence Check ---
        psi_change = np.linalg.norm(psi.x.array - psi_old.x.array) / (np.linalg.norm(psi.x.array) + 1e-30)
        print(f"  Iteration {k+1}: Relative change in psi = {psi_change:.2e}")
        if psi_change < tolerance:
            print(f"  ✓ Converged after {k+1} iterations")
            return psi, u, v, True
            
    print(f"  ⚠ Did not converge after {max_iterations} iterations")
    return psi, u, v, False


# ====================================================================
# 7. Current Calculation from Continuity Equations
# ====================================================================
def calculate_current_density(psi, n, p):
    """
    Calculate current density using drift-diffusion model
    """
    # Calculate electric field
    E = -ufl.grad(psi)
    
    # Electron current density: J_n = q * mu_n * n * E + q * D_n * grad(n)
    D_n = mu_n0 * V_T
    J_n = q * (mu_n0 * n * E + D_n * ufl.grad(n))
    
    # Hole current density: J_p = q * mu_p * p * E - q * D_p * grad(p)
    D_p = mu_p0 * V_T
    J_p = q * (mu_p0 * p * E - D_p * ufl.grad(p))
    
    # Total current density
    J_total = J_n + J_p
    
    return J_n, J_p, J_total

def calculate_terminal_current(psi, n, p, facet_tag, boundary_id):
    """
    Calculate current through a boundary (terminal)
    """
    # Get current densities
    J_n, J_p, J_total = calculate_current_density(psi, n, p)
    
    # Normal vector (outward pointing)
    n_vec = ufl.FacetNormal(domain)
    
    # Create measure for specific boundary
    ds_boundary = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    
    # Current = integral of J·n over boundary
    # Note: Current is positive when flowing out of domain
    I_n = fem.assemble_scalar(fem.form(ufl.dot(J_n, n_vec) * ds_boundary(boundary_id)))
    I_p = fem.assemble_scalar(fem.form(ufl.dot(J_p, n_vec) * ds_boundary(boundary_id)))
    I_total = fem.assemble_scalar(fem.form(ufl.dot(J_total, n_vec) * ds_boundary(boundary_id)))
    
    return float(I_n), float(I_p), float(I_total)

# ====================================================================
# 8. Enhanced Voltage Stepping with Full Physics
# ====================================================================
def voltage_stepping_drift_diffusion(voltage_list, plot_results=True):
    """
    Voltage stepping with full drift-diffusion model
    """
    print("\n" + "="*60)
    print("DRIFT-DIFFUSION VOLTAGE STEPPING SIMULATION")
    print("="*60)
    
    # Storage for results
    voltages = []
    currents = []
    electron_currents = []
    hole_currents = []
    
    # Initialize with equilibrium solution
    print("\nSolving equilibrium (V=0)...")
    psi_prev, n_prev, p_prev, converged = gummel_solver(0.0)
    
    if not converged:
        print("Failed to solve equilibrium!")
        return voltages, currents, electron_currents, hole_currents
    
    for i, voltage in enumerate(voltage_list):
        print(f"\n--- Step {i+1}/{len(voltage_list)}: V = {voltage:.3f} V ---")
        
        try:
            # Solve drift-diffusion equations using previous solution
            psi, n, p, converged = gummel_solver(voltage, psi_prev, n_prev, p_prev)
            
            if not converged:
                print(f"  ⚠ Solution did not converge fully")
            
            # Calculate current at cathode (bottom boundary, id=2)
            I_n, I_p, I_total = calculate_terminal_current(psi, n, p, facet_tag, 2)
            
            # For a diode, current convention: positive in forward bias
            # Since we're measuring at cathode, flip sign
            I_total = -I_total
            I_n = -I_n
            I_p = -I_p
            
            # Store results (multiply by width for 2D->1D)
            voltages.append(voltage)
            currents.append(I_total * width_m)
            electron_currents.append(I_n * width_m)
            hole_currents.append(I_p * width_m)
            
            print(f"✓ Total current: {I_total*width_m:.2e} A")
            print(f"  Electron current: {I_n*width_m:.2e} A")
            print(f"  Hole current: {I_p*width_m:.2e} A")
            
            # Save fields for next iteration
            psi_prev, n_prev, p_prev = psi, n, p
            
            # Optional: Print field statistics
            if i == 0 or i == len(voltage_list)//2 or i == len(voltage_list)-1:
                print(f"  Max n: {np.max(n.x.array):.2e} m⁻³")
                print(f"  Max p: {np.max(p.x.array):.2e} m⁻³")
                print(f"  Potential range: [{np.min(psi.x.array):.3f}, {np.max(psi.x.array):.3f}] V")
            
        except Exception as e:
            print(f"✗ Failed at voltage {voltage:.3f}V: {e}")
            import traceback
            traceback.print_exc()
            # Try to continue with smaller step
            if i > 0:
                psi_prev, n_prev, p_prev = psi, n, p
            continue
    
    print(f"\n✓ Simulation completed! Solved {len(currents)}/{len(voltage_list)} points.")
    
    if plot_results and len(voltages) > 1:
        plot_drift_diffusion_results(voltages, currents, electron_currents, hole_currents)
    
    return voltages, currents, electron_currents, hole_currents

def plot_drift_diffusion_results(voltages, currents, electron_currents, hole_currents):
    """
    Plot I-V characteristics with component currents
    """
    plt.figure(figsize=(12, 10))
    
    # Convert to arrays for easier manipulation
    voltages = np.array(voltages)
    currents = np.array(currents)
    electron_currents = np.array(electron_currents)
    hole_currents = np.array(hole_currents)
    
    # Total current (log scale)
    plt.subplot(2, 2, 1)
    # Plot absolute value for log scale
    plt.semilogy(voltages, np.abs(currents), 'ko-', linewidth=2, markersize=6, label='Total')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('|Current| (A)')
    plt.title('I-V Characteristics (Log Scale)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Component currents
    plt.subplot(2, 2, 2)
    plt.semilogy(voltages, np.abs(electron_currents), 'bo-', label='Electron Current', alpha=0.7)
    plt.semilogy(voltages, np.abs(hole_currents), 'ro-', label='Hole Current', alpha=0.7)
    plt.semilogy(voltages, np.abs(currents), 'k--', label='Total', alpha=0.5)
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('|Current| (A)')
    plt.title('Current Components')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Linear scale
    plt.subplot(2, 2, 3)
    plt.plot(voltages, currents, 'ko-', linewidth=2, markersize=6)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('I-V Characteristics (Linear Scale)')
    plt.grid(True, alpha=0.3)
    
    # Rectification analysis
    plt.subplot(2, 2, 4)
    forward_mask = voltages > 0.1
    reverse_mask = voltages < -0.1
    
    if np.any(forward_mask) and np.any(reverse_mask):
        forward_i_max = np.max(np.abs(currents[forward_mask]))
        reverse_i_max = np.max(np.abs(currents[reverse_mask]))
        
        if reverse_i_max > 0:
            rectification = forward_i_max / reverse_i_max
        else:
            rectification = np.inf
        
        # Try to calculate ideality factor
        forward_v = voltages[forward_mask]
        forward_i = currents[forward_mask]
        
        if len(forward_v) > 2 and np.all(forward_i > 0):
            ideality = calculate_ideality_factor(forward_v, forward_i)
        else:
            ideality = None
        
        plt.text(0.1, 0.8, f'Forward Current (max): {forward_i_max:.2e} A', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.7, f'Reverse Current (max): {reverse_i_max:.2e} A', 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.6, f'Rectification Ratio: {rectification:.1e}', 
                transform=plt.gca().transAxes, fontsize=10)
        if ideality:
            plt.text(0.1, 0.5, f'Ideality Factor: {ideality:.2f}', 
                    transform=plt.gca().transAxes, fontsize=10)
    
    plt.semilogy(voltages, np.abs(currents), 'ko-', linewidth=2, markersize=6, label='Simulation')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('|Current| (A)')
    plt.title('Diode Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_ideality_factor(voltages, currents):
    """
    Calculate ideality factor from I-V data
    """
    # Filter for reasonable voltage range and positive currents
    mask = (voltages > 0.2) & (voltages < 0.5) & (currents > 1e-15)
    
    if np.sum(mask) < 2:
        return None
    
    v_fit = voltages[mask]
    i_fit = currents[mask]
    
    try:
        # Calculate ideality from slope of log(I) vs V
        log_i = np.log(i_fit)
        # Linear fit
        coeffs = np.polyfit(v_fit, log_i, 1)
        slope = coeffs[0]
        ideality = q / (k_B * T * slope)
        
        # Reasonable bounds
        if 1.0 < ideality < 3.0:
            return ideality
        else:
            return None
    except:
        return None

# ====================================================================
# 9. Main Simulation
# ====================================================================
if __name__ == "__main__":
    print("Starting Full Drift-Diffusion Simulation...")
    print("Features:")
    print("  ✓ Continuity equations for electrons and holes")
    print("  ✓ Drift-diffusion transport")
    print("  ✓ SRH recombination model")
    print("  ✓ Gummel iteration solver")
    print("  ✓ Accurate current calculation from fluxes")
    
    # Define voltage sweep
    voltage_list = []
    
    # Start with small forward bias steps
    voltage_list.extend(np.linspace(0.1, 0.3, 5))
    voltage_list.extend(np.linspace(0.35, 0.6, 6))
    voltage_list.extend([0.65, 0.7])
    
    # Small reverse bias steps
    voltage_list.extend(np.linspace(-0.1, -0.5, 5))
    voltage_list.extend([-0.7, -1.0])
    
    # Sort for better convergence (start from 0 and go outward)
    voltage_list = sorted(voltage_list, key=abs)
    
    print(f"\nVoltage sweep: {len(voltage_list)} points")
    print(f"Range: {min(voltage_list):.2f}V to {max(voltage_list):.2f}V")
    
    # Run simulation
    try:
        voltages, currents, e_currents, h_currents = voltage_stepping_drift_diffusion(
            voltage_list, plot_results=True)
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        print(f"Successfully simulated {len(voltages)} voltage points")
        
        if len(currents) > 0:
            print(f"Current range: {min(np.abs(currents)):.2e} to {max(np.abs(currents)):.2e} A")
            
            # Save results to file
            results = np.column_stack((voltages, currents, e_currents, h_currents))
            np.savetxt('iv_characteristics.txt', results, 
                      header='Voltage(V) Current(A) ElectronCurrent(A) HoleCurrent(A)',
                      delimiter='\t', fmt='%.6e')
            print("Results saved to 'iv_characteristics.txt'")
        
        print("\n✓ Full drift-diffusion simulation completed!")
        print("✓ Results include electron and hole current components")
        print("✓ Physics includes SRH recombination and full transport")
        
    except Exception as e:
        print(f"\nSimulation failed: {e}")
        import traceback
        traceback.print_exc()