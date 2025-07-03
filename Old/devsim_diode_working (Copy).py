# Import necessary libraries
import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import Function
from dolfinx.fem.petsc import LinearProblem
import numpy as np
import ufl
from mpi4py import MPI
import matplotlib.pyplot as plt

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

# Mesh parameters
nx = 40  # Slightly increased for better accuracy
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

domain.topology.create_connectivity(1, 2)

# Define boundaries
def top_boundary(x):
    return np.isclose(x[1], height_m)

def bottom_boundary(x):
    return np.isclose(x[1], 0)

def side_walls(x):
    return np.logical_or(np.isclose(x[0], 0), np.isclose(x[0], width_m))

top_facets = mesh.locate_entities_boundary(domain, 1, top_boundary)
bottom_facets = mesh.locate_entities_boundary(domain, 1, bottom_boundary)
side_facets = mesh.locate_entities_boundary(domain, 1, side_walls)

marked_facets = np.hstack([top_facets, bottom_facets, side_facets])
markers = np.hstack([
    np.full_like(top_facets, 1),
    np.full_like(bottom_facets, 2),
    np.full_like(side_facets, 3)
])
facet_tag = mesh.meshtags(domain, 1, marked_facets, markers)

# ====================================================================
# 3. Set Up Function Spaces and Doping Profile
# ====================================================================
V_psi = fem.functionspace(domain, ("CG", 1))

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
# 4. Enhanced Equilibrium Solution
# ====================================================================
def solve_equilibrium():
    """
    Solve for equilibrium potential (0V bias)
    """
    print("\n--- Solving Equilibrium (0V bias) ---")
    
    psi_eq = Function(V_psi, name="Equilibrium_Potential")
    
    # Better initial guess using built-in potential
    V_bi = V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)
    
    def init_potential_eq(x):
        return V_bi * (1 - x[1] / height_m)
    
    psi_eq.interpolate(init_potential_eq)
    
    # Boundary conditions for equilibrium (both contacts at 0V)
    anode_dofs = fem.locate_dofs_topological(V_psi, 1, top_facets)
    cathode_dofs = fem.locate_dofs_topological(V_psi, 1, bottom_facets)
    
    bc_anode_eq = fem.dirichletbc(dolfinx.default_scalar_type(0.0), anode_dofs, V_psi)
    bc_cathode_eq = fem.dirichletbc(dolfinx.default_scalar_type(0.0), cathode_dofs, V_psi)
    bcs_eq = [bc_anode_eq, bc_cathode_eq]
    
    # Iterative solution for equilibrium
    v_psi = ufl.TestFunction(V_psi)
    max_iter_eq = 15
    
    for i in range(max_iter_eq):
        # Calculate carrier concentrations using charge neutrality
        n_eq = Function(fem.functionspace(domain, ("DG", 0)))
        p_eq = Function(fem.functionspace(domain, ("DG", 0)))
        
        for cell_idx in range(len(n_eq.x.array)):
            C_val = C.x.array[cell_idx]
            discriminant = C_val**2 + 4 * n_i_m3**2
            sqrt_disc = np.sqrt(discriminant)
            
            n_eq.x.array[cell_idx] = 0.5 * (sqrt_disc + C_val)
            p_eq.x.array[cell_idx] = 0.5 * (sqrt_disc - C_val)
        
        # Solve Poisson equation
        psi_trial = ufl.TrialFunction(V_psi)
        a_psi = ufl.inner(epsilon * ufl.grad(psi_trial), ufl.grad(v_psi)) * ufl.dx
        L_psi = q * (p_eq - n_eq + C) * v_psi * ufl.dx
        
        problem_eq = LinearProblem(a_psi, L_psi, bcs=bcs_eq, u=psi_eq)
        problem_eq.solve()
        
        if i % 5 == 0:
            psi_max = np.max(psi_eq.x.array)
            psi_min = np.min(psi_eq.x.array)
            print(f"  Iteration {i+1}: Potential range [{psi_min:.4f}, {psi_max:.4f}] V")
    
    return psi_eq, n_eq, p_eq

# ====================================================================
# 5. Enhanced Bias Solution with Better Physics
# ====================================================================
def solve_with_bias(applied_voltage, psi_initial, n_initial, p_initial, max_iterations=20):
    """
    Solve with applied bias using improved carrier transport
    """
    print(f"\n--- Solving with applied voltage: {applied_voltage:.3f} V ---")
    
    # Start from previous solution
    psi_sol = Function(V_psi)
    psi_sol.x.array[:] = psi_initial.x.array[:]
    
    # Boundary conditions with applied bias
    anode_dofs = fem.locate_dofs_topological(V_psi, 1, top_facets)
    cathode_dofs = fem.locate_dofs_topological(V_psi, 1, bottom_facets)
    
    bc_anode = fem.dirichletbc(dolfinx.default_scalar_type(applied_voltage), anode_dofs, V_psi)
    bc_cathode = fem.dirichletbc(dolfinx.default_scalar_type(0.0), cathode_dofs, V_psi)
    bcs_bias = [bc_anode, bc_cathode]
    
    # Enhanced carrier calculation with spatial dependence
    v_psi = ufl.TestFunction(V_psi)
    
    for iteration in range(max_iterations):
        # More sophisticated carrier concentration calculation
        n_bias = Function(fem.functionspace(domain, ("DG", 0)))
        p_bias = Function(fem.functionspace(domain, ("DG", 0)))
        
        # Get mesh coordinates for better spatial interpolation
        cells = domain.topology.connectivity(2, 0)  # cells to vertices
        geom = domain.geometry.x  # vertex coordinates
        
        for cell_idx in range(len(n_bias.x.array)):
            C_val = C.x.array[cell_idx]
            
            # Estimate potential at cell center (improved method)
            cell_vertices = cells.links(cell_idx)
            cell_coords = geom[cell_vertices]
            cell_center_y = np.mean(cell_coords[:, 1])
            
            # Linear interpolation of potential based on y-coordinate
            y_frac = cell_center_y / height_m
            psi_est = applied_voltage * y_frac
            
            # Enhanced carrier calculation with quasi-Fermi level effects
            if abs(applied_voltage) < 0.1:  # Small bias: use charge neutrality + small perturbation
                # Base equilibrium values
                discriminant = C_val**2 + 4 * n_i_m3**2
                sqrt_disc = np.sqrt(discriminant)
                n_eq_local = 0.5 * (sqrt_disc + C_val)
                p_eq_local = 0.5 * (sqrt_disc - C_val)
                
                # Small bias perturbation
                bias_factor = np.exp(np.clip(psi_est / V_T, -5, 5))
                n_bias.x.array[cell_idx] = n_eq_local * bias_factor
                p_bias.x.array[cell_idx] = p_eq_local / bias_factor
                
            else:  # Larger bias: use modified Boltzmann with limits
                exp_arg_n = np.clip(psi_est / V_T, -20, 20)
                exp_arg_p = np.clip(-psi_est / V_T, -20, 20)
                
                if C_val > 0:  # n-type region
                    n_base = np.sqrt(C_val * n_i_m3)
                    p_base = n_i_m3**2 / n_base
                else:  # p-type region
                    p_base = np.sqrt(abs(C_val) * n_i_m3)
                    n_base = n_i_m3**2 / p_base
                
                n_bias.x.array[cell_idx] = n_base * np.exp(exp_arg_n)
                p_bias.x.array[cell_idx] = p_base * np.exp(exp_arg_p)
        
        # Store old potential for convergence check
        psi_old = Function(V_psi)
        psi_old.x.array[:] = psi_sol.x.array[:]
        
        # Solve Poisson equation
        psi_trial = ufl.TrialFunction(V_psi)
        a_psi = ufl.inner(epsilon * ufl.grad(psi_trial), ufl.grad(v_psi)) * ufl.dx
        L_psi = q * (p_bias - n_bias + C) * v_psi * ufl.dx
        
        problem_bias = LinearProblem(a_psi, L_psi, bcs=bcs_bias, u=psi_sol)
        problem_bias.solve()
        
        # Check convergence
        psi_change = np.linalg.norm(psi_sol.x.array - psi_old.x.array)
        psi_norm = np.linalg.norm(psi_sol.x.array)
        relative_change = psi_change / psi_norm if psi_norm > 0 else 0
        
        if iteration % 5 == 0 or relative_change < 1e-6:
            psi_max = np.max(psi_sol.x.array)
            psi_min = np.min(psi_sol.x.array)
            print(f"  Iteration {iteration+1}: Potential range [{psi_min:.4f}, {psi_max:.4f}] V, "
                  f"Change: {relative_change:.2e}")
        
        if relative_change < 1e-6:
            print(f"  Converged after {iteration+1} iterations")
            break
    
    return psi_sol, n_bias, p_bias

# ====================================================================
# 6. Helper Functions for Numerical Stability
# ====================================================================
def bernoulli_function(x):
    """
    Bernoulli function B(x) = x/(exp(x)-1) for Scharfetter-Gummel discretization
    Uses Taylor expansion for small |x| to avoid numerical issues
    """
    if isinstance(x, (int, float)):
        if abs(x) < 1e-10:
            return 1.0 - x/2.0 + x**2/12.0 - x**4/720.0
        else:
            return x / (np.exp(x) - 1.0)
    else:
        # For arrays
        result = np.zeros_like(x)
        small_mask = np.abs(x) < 1e-10
        large_mask = ~small_mask
        
        # Taylor expansion for small values
        result[small_mask] = 1.0 - x[small_mask]/2.0 + x[small_mask]**2/12.0 - x[small_mask]**4/720.0
        
        # Exact formula for large values
        result[large_mask] = x[large_mask] / (np.exp(x[large_mask]) - 1.0)
        
        return result

# ====================================================================
# 7. Current Calculation Enhancement
# ====================================================================
def calculate_current(psi, n, p, applied_voltage):
    """
    Enhanced current calculation using better physics
    """
    # Physical parameters
    mu_n = 0.14  # m²/V/s
    mu_p = 0.045  # m²/V/s
    
    # Calculate average quantities
    avg_n = np.mean(n.x.array)
    avg_p = np.mean(p.x.array)
    
    # Enhanced conductivity calculation
    avg_conductivity = q * (avg_n * mu_n + avg_p * mu_p)
    
    # Potential drop calculation
    V_drop = np.max(psi.x.array) - np.min(psi.x.array)
    
    # Current estimate with better physics
    # Include both drift and diffusion components (simplified)
    drift_current = avg_conductivity * (V_drop / height_m) * width_m
    
    # Add diffusion contribution (simplified)
    # ∇n and ∇p contribute to current even without electric field
    n_gradient = (np.max(n.x.array) - np.min(n.x.array)) / height_m
    p_gradient = (np.max(p.x.array) - np.min(p.x.array)) / height_m
    
    diffusion_current = q * width_m * V_T * (mu_n * n_gradient - mu_p * p_gradient)
    
    total_current = drift_current + diffusion_current
    
    return total_current, drift_current, diffusion_current, avg_conductivity

# ====================================================================
# 8. Voltage Stepping Algorithm
# ====================================================================
def voltage_stepping_simulation(voltage_list, plot_results=True):
    """
    Perform voltage stepping simulation
    """
    print("\n" + "="*60)
    print("VOLTAGE STEPPING SIMULATION")
    print("="*60)
    
    # Start with equilibrium
    psi_prev, n_prev, p_prev = solve_equilibrium()
    
    # Storage for results
    voltages = [0.0]  # Start with equilibrium
    currents = [0.0]  # No current at equilibrium
    drift_currents = [0.0]
    diffusion_currents = [0.0]
    conductivities = []
    
    # Calculate equilibrium conductivity
    _, _, _, eq_conductivity = calculate_current(psi_prev, n_prev, p_prev, 0.0)
    conductivities.append(eq_conductivity)
    
    print(f"\n✓ Equilibrium solution complete")
    
    # Step through voltages
    for i, voltage in enumerate(voltage_list):
        print(f"\n--- Voltage Step {i+1}/{len(voltage_list)}: {voltage:.3f} V ---")
        
        try:
            # Solve with current voltage using previous solution as initial guess
            psi_new, n_new, p_new = solve_with_bias(voltage, psi_prev, n_prev, p_prev)
            
            # Calculate current
            total_current, drift_current, diffusion_current, conductivity = calculate_current(
                psi_new, n_new, p_new, voltage)
            
            # Store results
            voltages.append(voltage)
            currents.append(abs(total_current))  # Take absolute value
            drift_currents.append(abs(drift_current))
            diffusion_currents.append(abs(diffusion_current))
            conductivities.append(conductivity)
            
            # Update for next iteration
            psi_prev, n_prev, p_prev = psi_new, n_new, p_new
            
            print(f"✓ Voltage {voltage:.3f}V: Current = {abs(total_current):.2e} A/m")
            
        except Exception as e:
            print(f"✗ Failed at voltage {voltage:.3f}V: {e}")
            break
    
    print(f"\n✓ Voltage stepping completed! Solved {len(currents)} points.")
    
    # Plot results if requested
    if plot_results and len(voltages) > 1:
        plot_iv_characteristics(voltages, currents, drift_currents, diffusion_currents)
    
    return voltages, currents, drift_currents, diffusion_currents, conductivities

def plot_iv_characteristics(voltages, currents, drift_currents, diffusion_currents):
    """
    Plot I-V characteristics
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Main I-V plot
        plt.subplot(2, 2, 1)
        plt.semilogy(voltages, currents, 'bo-', linewidth=2, markersize=6, label='Total Current')
        plt.xlabel('Applied Voltage (V)')
        plt.ylabel('Current (A/m)')
        plt.title('I-V Characteristics')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Current components
        plt.subplot(2, 2, 2)
        plt.semilogy(voltages, drift_currents, 'ro-', label='Drift Current', alpha=0.7)
        plt.semilogy(voltages, diffusion_currents, 'go-', label='Diffusion Current', alpha=0.7)
        plt.xlabel('Applied Voltage (V)')
        plt.ylabel('Current (A/m)')
        plt.title('Current Components')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Linear scale plot
        plt.subplot(2, 2, 3)
        plt.plot(voltages, currents, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Applied Voltage (V)')
        plt.ylabel('Current (A/m)')
        plt.title('I-V Characteristics (Linear Scale)')
        plt.grid(True, alpha=0.3)
        
        # Current vs Voltage (highlighting diode behavior)
        plt.subplot(2, 2, 4)
        if len(voltages) > 1:
            rectification_ratio = max(currents) / min(currents[1:]) if min(currents[1:]) > 0 else float('inf')
            plt.text(0.1, 0.8, f'Max Current: {max(currents):.2e} A/m', transform=plt.gca().transAxes)
            plt.text(0.1, 0.7, f'Min Current: {min(currents[1:]):.2e} A/m', transform=plt.gca().transAxes)
            plt.text(0.1, 0.6, f'Ratio: {rectification_ratio:.1e}', transform=plt.gca().transAxes)
        
        plt.semilogy(voltages, currents, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Applied Voltage (V)')
        plt.ylabel('Current (A/m)')
        plt.title('Diode Characteristics')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for plotting. Results saved to variables.")

# ====================================================================
# 8. Run Voltage Stepping Simulation
# ====================================================================
if __name__ == "__main__":
    print("Starting Enhanced Voltage Stepping Simulation...")
    
    # Define voltage sweep
    # Start with small steps, then larger steps
    voltage_list = []
    
    # Small forward bias steps
    voltage_list.extend(np.linspace(0.05, 0.3, 6))  # 0.05 to 0.3V
    voltage_list.extend(np.linspace(0.35, 0.7, 8))  # 0.35 to 0.7V
    
    # Small reverse bias steps  
    voltage_list.extend(np.linspace(-0.1, -1.0, 5))  # Small reverse bias
    
    print(f"Voltage sweep: {len(voltage_list)} points from {min(voltage_list):.2f}V to {max(voltage_list):.2f}V")
    
    # Run simulation
    try:
        voltages, currents, drift_currents, diffusion_currents, conductivities = voltage_stepping_simulation(
            voltage_list, plot_results=True)
        
        print("\n" + "="*60)
        print("SIMULATION RESULTS SUMMARY")
        print("="*60)
        print(f"Successfully simulated {len(voltages)} voltage points")
        print(f"Voltage range: {min(voltages):.3f}V to {max(voltages):.3f}V")
        print(f"Current range: {min(currents):.2e} to {max(currents):.2e} A/m")
        
        if len(currents) > 1:
            forward_currents = [c for v, c in zip(voltages, currents) if v > 0]
            reverse_currents = [c for v, c in zip(voltages, currents) if v < 0]
            
            if forward_currents and reverse_currents:
                rectification = max(forward_currents) / max(reverse_currents)
                print(f"Rectification ratio: {rectification:.1e}")
        
        print("\n✓ Enhanced drift-diffusion simulation completed successfully!")
        print("Next steps: Add full current continuity equations or mesh refinement")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        print("Try reducing voltage steps or mesh density")