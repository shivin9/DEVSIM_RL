# Import necessary libraries
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import Function
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import numpy as np
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt
import os

# Import gmsh utilities
from dolfinx.io import gmshio

# ====================================================================
# 1. Define Physical and Geometric Parameters
# ====================================================================
q = 1.60217663e-19
epsilon_0 = 8.85418782e-12
epsilon_si = 11.7
epsilon = epsilon_si * epsilon_0
k_B = 1.380649e-23
T = 300
V_T = k_B * T / q

total_length_m = 10.0e-6
junction_center = 5.0e-6
thickness_m = 1.0e-6

NA_max_m3 = 1e17 * 1e6
ND_max_m3 = 1e17 * 1e6
n_i_m3 = 1.0e10 * 1e6

print("Physical and geometric parameters loaded.")

# ====================================================================
# 2. Load Mesh from Gmsh
# ====================================================================
comm = MPI.COMM_WORLD
msh_file = "diode.msh"

if comm.rank == 0:
    if not os.path.exists(msh_file):
        print(f"Mesh file {msh_file} not found. Generating it with Gmsh...")
        os.system(f"gmsh diode.geo -2 -o {msh_file}")
comm.barrier()

domain, _, facet_tag = gmshio.read_from_msh(msh_file, comm, rank=0, gdim=2)

print(f"Mesh loaded from {msh_file}.")
P_CONTACT_ID, N_CONTACT_ID = 1, 2

# ====================================================================
# 3. Set Up Function Spaces and Doping Profile
# ====================================================================
V = fem.functionspace(domain, ("CG", 1))
Q = fem.functionspace(domain, ("DG", 0))

C = Function(Q)
C.interpolate(lambda x: np.where(x[0] <= junction_center, -NA_max_m3, ND_max_m3))

print("Doping profile and function spaces created.")

# ====================================================================
# 4. Core Solver Implementation
# ====================================================================

def srh_recombination(n, p):
    tau_n, tau_p = 1e-7, 1e-7
    # Add a small regularization to prevent division by zero or very large values
    epsilon_reg = 1e-30 
    return (n * p - n_i_m3**2) / (tau_p * (n + n_i_m3) + tau_n * (p + n_i_m3) + epsilon_reg)

def solve_equilibrium():
    print("\n--- Solving Equilibrium (0V bias) ---")
    psi_eq = Function(V, name="Equilibrium_Potential")
    n_eq = Function(V, name="Equilibrium_Electrons")
    p_eq = Function(V, name="Equilibrium_Holes")

    # Initial guess for potential: linear drop from V_bi
    V_bi = V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)
    psi_eq.interpolate(lambda x: V_bi * (1 - x[0] / total_length_m))

    # Initial guess for carriers based on local charge neutrality (more robust)
    n_eq.interpolate(lambda x: np.where(x[0] <= junction_center, n_i_m3**2/NA_max_m3, ND_max_m3))
    p_eq.interpolate(lambda x: np.where(x[0] <= junction_center, NA_max_m3, n_i_m3**2/ND_max_m3))
    
    # Ensure initial n and p are strictly positive and within reasonable bounds
    n_eq.x.array[n_eq.x.array < 1e-30] = 1e-30
    p_eq.x.array[p_eq.x.array < 1e-30] = 1e-30
    n_eq.x.array[n_eq.x.array > 1e30] = 1e30
    p_eq.x.array[p_eq.x.array > 1e30] = 1e30

    anode_facets = facet_tag.find(P_CONTACT_ID)
    anode_dofs = fem.locate_dofs_topological(V, 1, anode_facets)
    bc_anode_eq = fem.dirichletbc(dolfinx.default_scalar_type(0.0), anode_dofs, V)
    
    # Iterative solution for equilibrium (Gummel-like loop)
    for i in range(100): # Increased iterations for robustness
        psi_old = Function(V)
        psi_old.x.array[:] = psi_eq.x.array

        # Solve Poisson equation
        v_psi = ufl.TestFunction(V)
        F_psi = ufl.inner(epsilon * ufl.grad(psi_eq), ufl.grad(v_psi)) * ufl.dx - q * (p_eq - n_eq + C) * v_psi * ufl.dx
        problem_psi = NonlinearProblem(F_psi, psi_eq, bcs=[bc_anode_eq])
        solver_psi = NewtonSolver(comm, problem_psi)
        solver_psi.convergence_criterion = "incremental"
        solver_psi.rtol = 1e-7 # Tighter tolerance for equilibrium
        solver_psi.max_it = 20 # Max 20 Newton iterations per Gummel step
        
        # Store current psi_eq before solving for relaxation
        psi_eq_before_solve = Function(V)
        psi_eq_before_solve.x.array[:] = psi_eq.x.array

        n_iter, converged = solver_psi.solve(psi_eq)

        # Apply relaxation to psi_eq
        relaxation_factor_psi = 0.5 # New relaxation factor for psi_eq
        psi_eq.x.array[:] = (1 - relaxation_factor_psi) * psi_eq_before_solve.x.array + relaxation_factor_psi * psi_eq.x.array

        # Update n and p based on new psi (thermal equilibrium) with relaxation
        n_new_expr = fem.Expression(n_i_m3 * ufl.exp(psi_eq / V_T), V.element.interpolation_points())
        p_new_expr = fem.Expression(n_i_m3 * ufl.exp(-psi_eq / V_T), V.element.interpolation_points())
        
        n_new = Function(V)
        p_new = Function(V)
        n_new.interpolate(n_new_expr)
        p_new.interpolate(p_new_expr)

        relaxation_factor = 0.5 # Relaxation for stability
        n_eq.x.array[:] = (1 - relaxation_factor) * n_eq.x.array + relaxation_factor * n_new.x.array
        p_eq.x.array[:] = (1 - relaxation_factor) * p_eq.x.array + relaxation_factor * p_new.x.array

        change = np.linalg.norm(psi_eq.x.array - psi_old.x.array) / np.linalg.norm(psi_eq.x.array)
        print(f"  Equilibrium Gummel iter {i+1}: Potential change = {change:.2e}")
        if change < 1e-6 and i > 1:
            print("  Equilibrium converged.")
            break

    return psi_eq, n_eq, p_eq

def solve_bias(voltage, psi_in, n_in, p_in):
    psi = Function(V)
    n = Function(V)
    p = Function(V)
    psi.x.array[:], n.x.array[:], p.x.array[:] = psi_in.x.array, n_in.x.array, p_in.x.array

    # Define boundary conditions
    anode_facets = facet_tag.find(P_CONTACT_ID)
    cathode_facets = facet_tag.find(N_CONTACT_ID)
    anode_dofs = fem.locate_dofs_topological(V, 1, anode_facets)
    cathode_dofs = fem.locate_dofs_topological(V, 1, cathode_facets)

    V_bi = V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)
    psi_anode = V_bi - voltage
    psi_cathode = 0.0

    n_anode = n_i_m3**2 / NA_max_m3
    p_anode = NA_max_m3
    n_cathode = ND_max_m3
    p_cathode = n_i_m3**2 / ND_max_m3

    bcs_psi = [fem.dirichletbc(dolfinx.default_scalar_type(psi_anode), anode_dofs, V),
               fem.dirichletbc(dolfinx.default_scalar_type(psi_cathode), cathode_dofs, V)]
    bcs_n = [fem.dirichletbc(dolfinx.default_scalar_type(n_anode), anode_dofs, V),
             fem.dirichletbc(dolfinx.default_scalar_type(n_cathode), cathode_dofs, V)]
    bcs_p = [fem.dirichletbc(dolfinx.default_scalar_type(p_anode), anode_dofs, V),
             fem.dirichletbc(dolfinx.default_scalar_type(p_cathode), cathode_dofs, V)]

    # Gummel loop
    for i in range(50): # Increased iterations for robustness
        psi_old = Function(V)
        psi_old.x.array[:] = psi.x.array

        # Solve Poisson
        v_psi = ufl.TestFunction(V)
        F_psi = ufl.inner(epsilon * ufl.grad(psi), ufl.grad(v_psi)) * ufl.dx - q * (p - n + C) * v_psi * ufl.dx
        problem_psi = NonlinearProblem(F_psi, psi, bcs=bcs_psi)
        solver_psi = NewtonSolver(comm, problem_psi)
        solver_psi.convergence_criterion = "incremental"
        solver_psi.rtol = 1e-6
        solver_psi.max_it = 10
        solver_psi.solve(psi)

        # Solve Electron Continuity
        n_test = ufl.TestFunction(V)
        mu_n, D_n = 0.14, 0.14 * V_T
        J_n = -q * mu_n * n * ufl.grad(psi) - q * D_n * ufl.grad(n)
        F_n = ufl.inner(J_n, ufl.grad(n_test)) * ufl.dx - q * srh_recombination(n, p) * n_test * ufl.dx
        problem_n = NonlinearProblem(F_n, n, bcs=bcs_n)
        solver_n = NewtonSolver(comm, problem_n)
        solver_n.convergence_criterion = "incremental"
        solver_n.rtol = 1e-6
        solver_n.max_it = 10
        solver_n.solve(n)

        # Solve Hole Continuity
        p_test = ufl.TestFunction(V)
        mu_p, D_p = 0.045, 0.045 * V_T
        J_p = -q * mu_p * p * ufl.grad(psi) + q * D_p * ufl.grad(p)
        F_p = ufl.inner(J_p, ufl.grad(p_test)) * ufl.dx + q * srh_recombination(n, p) * p_test * ufl.dx
        problem_p = NonlinearProblem(F_p, p, bcs=bcs_p)
        solver_p = NewtonSolver(comm, problem_p)
        solver_p.convergence_criterion = "incremental"
        solver_p.rtol = 1e-6
        solver_p.max_it = 10
        solver_p.solve(p)

        change = np.linalg.norm(psi.x.array - psi_old.x.array) / np.linalg.norm(psi.x.array)
        print(f"  Bias Gummel iter {i+1}: Potential change = {change:.2e}")
        if change < 1e-5 and i > 1:
            print("  Converged.")
            break
    return psi, n, p

# ====================================================================
# 5. Current Calculation
# ====================================================================
def calculate_current(psi, n, p):
    mu_n, D_n = 0.14, 0.14 * V_T
    mu_p, D_p = 0.045, 0.045 * V_T
    E = -ufl.grad(psi)
    J_n = -q * mu_n * n * E - q * D_n * ufl.grad(n)
    J_p = -q * mu_p * p * E + q * D_p * ufl.grad(p)
    J_total = J_n + J_p

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
    anode_current = fem.assemble_scalar(fem.form(ufl.dot(J_total, ufl.FacetNormal(domain)) * ds(P_CONTACT_ID)))
    cathode_current = fem.assemble_scalar(fem.form(ufl.dot(J_total, ufl.FacetNormal(domain)) * ds(N_CONTACT_ID)))
    
    print(f"    Current check: Anode={anode_current:.2e} A/m, Cathode={cathode_current:.2e} A/m")
    return ((anode_current - cathode_current) / 2.0) * thickness_m

# ====================================================================
# 6. Main Simulation Loop
# ====================================================================
if __name__ == "__main__":
    print("\nStarting Final Diode Simulation...")
    
    psi_eq, n_eq, p_eq = solve_equilibrium()

    voltages = [0.0]
    currents = [calculate_current(psi_eq, n_eq, p_eq)]
    
    voltage_sweep = np.concatenate([
        np.linspace(0, -0.5, 11, endpoint=False),
        np.linspace(0, 0.7, 14)
    ])

    for voltage in voltage_sweep:
        print(f"\n--- Voltage Step: {voltage:.3f} V ---")
        try:
            psi_sol, n_sol, p_sol = solve_bias(voltage, psi_eq, n_eq, p_eq)
            current = calculate_current(psi_sol, n_sol, p_sol)
            voltages.append(voltage)
            currents.append(current)
            psi_eq, n_eq, p_eq = psi_sol, n_sol, p_sol
        except Exception as e:
            print(f"✗ Failed at voltage {voltage:.3f}V: {e}")
            break

    print("\n" + "="*60)
    print("SIMULATION RESULTS SUMMARY")
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.semilogy(voltages, np.abs(currents), 'bo-', label='Total Current')
    plt.xlabel('Applied Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('Final I-V Characteristics')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("iv_curve_final.png")
    print("\nI-V curve saved to iv_curve_final.png")
    plt.show()