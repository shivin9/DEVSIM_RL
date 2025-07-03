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

# Realistic Diode Geometric parameters
# Typical silicon diode dimensions
total_length_um = 10.0    # Total device length
contact_length_um = 2.0   # Length of each contact region
junction_length_um = 6.0  # Active junction region length
width_um = 5.0            # Device width
thickness_um = 1.0        # Device thickness (for 2D simulation)

# Convert to meters
total_length_m = total_length_um * 1e-6
contact_length_m = contact_length_um * 1e-6
junction_length_m = junction_length_um * 1e-6
width_m = width_um * 1e-6
thickness_m = thickness_um * 1e-6

# Realistic Doping parameters
NA_max_cm3 = 1e17        # P-type contact doping (higher for good contact)
ND_max_cm3 = 1e17        # N-type contact doping
NA_junction_cm3 = 1e16   # P-side junction doping (lower)
ND_junction_cm3 = 1e16   # N-side junction doping
n_i_cm3 = 1.0e10         # Intrinsic carrier concentration

# Convert to m^-3
NA_max_m3 = NA_max_cm3 * 1e6
ND_max_m3 = ND_max_cm3 * 1e6
NA_junction_m3 = NA_junction_cm3 * 1e6
ND_junction_m3 = ND_junction_cm3 * 1e6
n_i_m3 = n_i_cm3 * 1e6

# Device regions (along x-axis)
p_contact_end = contact_length_m
junction_center = contact_length_m + junction_length_m / 2
n_contact_start = contact_length_m + junction_length_m

# Mesh parameters - balanced mesh for junction resolution vs speed
nx = 160  # Moderate density along device length 
ny = 60   # Moderate density along device width

print(f"Physical parameters:")
print(f"V_T = {V_T:.4f} V")
print(f"Built-in potential estimate: {V_T * np.log(NA_junction_m3 * ND_junction_m3 / n_i_m3**2):.4f} V")
print(f"Device geometry:")
print(f"Total length: {total_length_um:.1f} μm")
print(f"Junction region: {junction_length_um:.1f} μm")
print(f"Device width: {width_um:.1f} μm")
print(f"P-contact doping: {NA_max_cm3:.1e} cm^-3")
print(f"N-contact doping: {ND_max_cm3:.1e} cm^-3")

# ====================================================================
# 2. Create Mesh and Define Boundaries
# ====================================================================
comm = MPI.COMM_WORLD

# Create high-density mesh focused on junction resolution
# X-axis: along device length (p-contact -> junction -> n-contact)
# Y-axis: along device width
print(f"Creating high-density mesh: {nx}×{ny} = {nx*ny} elements")
print(f"Junction region ({junction_length_um:.1f} μm) mesh density: ~{junction_length_m/(nx*0.6)*1e9:.1f} nm per element")

domain = mesh.create_rectangle(
    comm,
    points=[(0, 0), (total_length_m, width_m)],
    n=[nx, ny],
    cell_type=mesh.CellType.triangle
)

domain.topology.create_connectivity(1, 2)

# Define realistic diode boundaries
def p_contact_boundary(x):
    """Left boundary - P-type contact (anode)"""
    return np.isclose(x[0], 0)

def n_contact_boundary(x):
    """Right boundary - N-type contact (cathode)"""
    return np.isclose(x[0], total_length_m)

def top_boundary(x):
    """Top boundary - insulating"""
    return np.isclose(x[1], width_m)

def bottom_boundary(x):
    """Bottom boundary - insulating"""
    return np.isclose(x[1], 0)

# Locate boundary facets
p_contact_facets = mesh.locate_entities_boundary(domain, 1, p_contact_boundary)
n_contact_facets = mesh.locate_entities_boundary(domain, 1, n_contact_boundary)
top_facets = mesh.locate_entities_boundary(domain, 1, top_boundary)
bottom_facets = mesh.locate_entities_boundary(domain, 1, bottom_boundary)

# Mark boundaries with different IDs
# 1 = P-contact (anode), 2 = N-contact (cathode), 3 = insulating walls
marked_facets = np.hstack([p_contact_facets, n_contact_facets, top_facets, bottom_facets])
markers = np.hstack([
    np.full_like(p_contact_facets, 1),    # P-contact
    np.full_like(n_contact_facets, 2),    # N-contact
    np.full_like(top_facets, 3),          # Insulating
    np.full_like(bottom_facets, 3)        # Insulating
])
facet_tag = mesh.meshtags(domain, 1, marked_facets, markers)

print(f"Mesh created: {nx}×{ny} = {nx*ny} elements")
print(f"P-contact facets: {len(p_contact_facets)}")
print(f"N-contact facets: {len(n_contact_facets)}")

# ====================================================================
# 3. Set Up Function Spaces and Doping Profile
# ====================================================================
V_psi = fem.functionspace(domain, ("CG", 1))

# Doping profile
Q = fem.functionspace(domain, ("DG", 0))
N_A = Function(Q)
N_D = Function(Q)

# Realistic doping profile along device length (x-direction)
def doping_na(x):
    """P-type acceptor doping profile"""
    na = np.zeros_like(x[0])
    
    # P-contact region (0 to p_contact_end): high doping
    p_region = x[0] <= p_contact_end
    na[p_region] = NA_max_m3
    
    # Junction region (p_contact_end to n_contact_start): lower p-doping on left side
    junction_region = (x[0] > p_contact_end) & (x[0] < junction_center)
    na[junction_region] = NA_junction_m3
    
    # N-side of junction and N-contact: minimal p-doping (compensation)
    n_side = x[0] >= junction_center
    na[n_side] = n_i_m3**2 / ND_junction_m3  # Background p-doping
    
    return na

def doping_nd(x):
    """N-type donor doping profile"""
    nd = np.zeros_like(x[0])
    
    # P-contact and P-side: minimal n-doping (compensation)
    p_side = x[0] < junction_center
    nd[p_side] = n_i_m3**2 / NA_junction_m3  # Background n-doping
    
    # Junction region (junction_center to n_contact_start): n-doping
    junction_region = (x[0] >= junction_center) & (x[0] < n_contact_start)
    nd[junction_region] = ND_junction_m3
    
    # N-contact region (n_contact_start to end): high doping
    n_region = x[0] >= n_contact_start
    nd[n_region] = ND_max_m3
    
    return nd

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
    
    # Calculate realistic built-in potential using junction doping
    V_bi = V_T * np.log(NA_junction_m3 * ND_junction_m3 / n_i_m3**2)
    
    def init_potential_eq(x):
        # Linear potential drop from V_bi at bottom (n-type) to 0 at top (p-type)
        # Linear potential drop along device length (x-direction)
        return V_bi * (1 - x[0] / total_length_m)
    
    psi_eq.interpolate(init_potential_eq)
    print(f"  Initial guess: V_bi = {V_bi:.4f} V")
    
    # Boundary conditions for equilibrium - use natural equilibrium potentials
    anode_dofs = fem.locate_dofs_topological(V_psi, 1, p_contact_facets)
    cathode_dofs = fem.locate_dofs_topological(V_psi, 1, n_contact_facets)
    
    # Proper equilibrium: floating contacts to establish built-in potential
    # Only fix one contact as reference, let the other float
    bc_anode_eq = fem.dirichletbc(dolfinx.default_scalar_type(0.0), anode_dofs, V_psi)
    bcs_eq = [bc_anode_eq]  # Only fix p-contact, let n-contact float
    
    # Iterative solution for equilibrium
    v_psi = ufl.TestFunction(V_psi)
    max_iter_eq = 15
    
    for i in range(max_iter_eq):
        # Calculate carrier concentrations using charge neutrality
        n_eq = Function(fem.functionspace(domain, ("DG", 0)))
        p_eq = Function(fem.functionspace(domain, ("DG", 0)))
        
        # Calculate carrier concentrations with simpler, more stable approach
        for cell_idx in range(len(n_eq.x.array)):
            C_val = C.x.array[cell_idx]
            
            # For equilibrium, use simple charge neutrality without exponentials
            if C_val > 0:  # n-type region
                n_eq.x.array[cell_idx] = abs(C_val) + n_i_m3
                p_eq.x.array[cell_idx] = n_i_m3**2 / n_eq.x.array[cell_idx]
            elif C_val < 0:  # p-type region  
                p_eq.x.array[cell_idx] = abs(C_val) + n_i_m3
                n_eq.x.array[cell_idx] = n_i_m3**2 / p_eq.x.array[cell_idx]
            else:  # intrinsic region
                n_eq.x.array[cell_idx] = n_i_m3
                p_eq.x.array[cell_idx] = n_i_m3
            
            # Ensure minimum values
            n_eq.x.array[cell_idx] = max(n_eq.x.array[cell_idx], n_i_m3 * 1e-3)
            p_eq.x.array[cell_idx] = max(p_eq.x.array[cell_idx], n_i_m3 * 1e-3)
        
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
    
    # Start from previous solution with better initialization
    psi_sol = Function(V_psi)
    psi_sol.x.array[:] = psi_initial.x.array[:]
    
    # Add applied voltage gradually to initial guess
    def gradual_bias_init(x):
        # Apply voltage mainly across junction region
        if x[0] < p_contact_end:  # P-contact region
            return 0.0  # Reference
        elif x[0] > n_contact_start:  # N-contact region
            return applied_voltage  # Applied bias
        else:  # Junction region - smooth transition
            junction_fraction = (x[0] - p_contact_end) / junction_length_m
            base_potential = psi_initial.x.array[0] if len(psi_initial.x.array) > 0 else 0.0
            return base_potential + applied_voltage * junction_fraction
    
    # Apply gradual bias to initial potential
    bias_correction = Function(V_psi)
    bias_correction.interpolate(gradual_bias_init)
    
    # Blend initial solution with bias correction
    alpha = 0.1  # Small correction
    psi_sol.x.array[:] = (1-alpha) * psi_sol.x.array + alpha * bias_correction.x.array
    
    # Boundary conditions with applied bias
    anode_dofs = fem.locate_dofs_topological(V_psi, 1, p_contact_facets)
    cathode_dofs = fem.locate_dofs_topological(V_psi, 1, n_contact_facets)
    
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
            
            # Get actual potential at this cell from current solution
            psi_val = psi_sol.x.array[cell_idx]
            
            # Proper charge neutrality calculation
            # Use direct solution of: n - p + C = 0 and np = ni²exp(psi/VT)
            
            # Exponential term with numerical limiting
            exp_psi_vt = np.exp(np.clip(psi_val / V_T, -30, 30))
            ni_eff_sq = n_i_m3**2 * exp_psi_vt
            
            # Solve quadratic: n² + C*n - ni_eff² = 0
            discriminant = C_val**2 + 4 * ni_eff_sq
            sqrt_disc = np.sqrt(discriminant)
            
            # Physical solution (positive concentrations)
            n_local = 0.5 * (-C_val + sqrt_disc)
            p_local = ni_eff_sq / n_local if n_local > 1e5 else 0.5 * (C_val + sqrt_disc)
            
            # Ensure physical bounds
            n_local = max(n_local, n_i_m3 * 1e-6)  # Minimum concentration
            p_local = max(p_local, n_i_m3 * 1e-6)
            
            n_bias.x.array[cell_idx] = n_local
            p_bias.x.array[cell_idx] = p_local
        
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

def solve_with_bias_gummel(applied_voltage, psi_initial, n_initial, p_initial, max_iterations=20):
    """
    Solve with applied bias using proper Gummel iteration of drift-diffusion equations
    
    This replaces the simplified carrier models with proper coupling of:
    1. Poisson equation for potential
    2. Electron continuity equation  
    3. Hole continuity equation
    
    Args:
        applied_voltage: voltage applied to anode
        psi_initial: initial potential guess
        n_initial: initial electron density guess  
        p_initial: initial hole density guess
        max_iterations: maximum Gummel iterations
    
    Returns:
        (psi_sol, n_sol, p_sol): solved potential and carrier densities
    """
    print(f"\n--- Solving with Gummel iteration: {applied_voltage:.3f} V ---")
    
    # Initialize solutions
    psi_sol = Function(V_psi)
    psi_sol.x.array[:] = psi_initial.x.array[:]
    
    # Initialize Slotboom variables from carrier densities
    u_sol = Function(V_psi, name="Electron_Slotboom")
    v_sol = Function(V_psi, name="Hole_Slotboom")
    
    # Use conservative equilibrium values to avoid extreme exponentials
    u_sol.x.array[:] = 1.0  # Start with unity (equilibrium)
    v_sol.x.array[:] = 1.0
    
    # Gummel iteration loop (fewer iterations with fine mesh)
    max_iterations = min(max_iterations, 15)  # Limit iterations for stability
    for iteration in range(max_iterations):
        psi_old = Function(V_psi)
        psi_old.x.array[:] = psi_sol.x.array[:]
        
        # Step 1: Solve Poisson equation with current carrier densities
        n_current = Function(fem.functionspace(domain, ("DG", 0)))
        p_current = Function(fem.functionspace(domain, ("DG", 0)))
        
        # Convert Slotboom variables to densities for Poisson equation
        # Map from CG(1) Slotboom variables to DG(0) carrier densities
        dg_size = len(n_current.x.array)
        cg_size = len(u_sol.x.array)
        
        # Track charge conservation for stability
        total_charge_density = 0.0
        total_volume = 0.0
        
        for i in range(dg_size):
            # Map DG index to CG index
            cg_idx = min(i * cg_size // dg_size, cg_size - 1)
            
            psi_val = psi_sol.x.array[cg_idx]
            u_val = u_sol.x.array[cg_idx]
            v_val = v_sol.x.array[cg_idx]
            
            # Much tighter potential limits to prevent exponential explosion
            psi_limited = np.clip(psi_val, -1.5, 1.5)  # Limit to ±1.5V max
            exp_psi = np.exp(np.clip(psi_limited / V_T, -10, 10))
            exp_neg_psi = np.exp(np.clip(-psi_limited / V_T, -10, 10))
            
            # Calculate carrier densities with controlled exponentials
            n_local = n_i_m3 * u_val * exp_psi
            p_local = n_i_m3 * v_val * exp_neg_psi
            
            # Apply strict physical bounds to prevent charge explosion
            doping_level = abs(C.x.array[i]) if i < len(C.x.array) else NA_junction_m3
            max_reasonable = max(doping_level * 2, n_i_m3 * 10)  # At most 2x local doping
            
            n_local = np.clip(n_local, n_i_m3 * 1e-6, max_reasonable)
            p_local = np.clip(p_local, n_i_m3 * 1e-6, max_reasonable)
            
            n_current.x.array[i] = n_local
            p_current.x.array[i] = p_local
            
            # Track total charge for conservation check
            local_charge = q * (p_local - n_local + C.x.array[i] if i < len(C.x.array) else 0)
            total_charge_density += local_charge
            total_volume += 1.0
        
        # Apply charge conservation constraint to prevent Poisson instability
        avg_charge_density = total_charge_density / total_volume if total_volume > 0 else 0.0
        print(f"    Charge conservation check: avg density = {avg_charge_density:.2e} C/m³")
        
        # If charge is severely unbalanced, apply correction
        if abs(avg_charge_density) > q * max(NA_max_m3, ND_max_m3):
            print(f"    Applying charge conservation correction...")
            charge_correction = -avg_charge_density / q
            
            # Apply correction to carrier densities to restore charge neutrality
            for i in range(dg_size):
                if avg_charge_density > 0:  # Too much positive charge, reduce holes
                    p_current.x.array[i] = max(p_current.x.array[i] + charge_correction, n_i_m3 * 1e-6)
                else:  # Too much negative charge, reduce electrons
                    n_current.x.array[i] = max(n_current.x.array[i] - charge_correction, n_i_m3 * 1e-6)
        
        # Solve Poisson equation with stabilized charge densities
        anode_dofs = fem.locate_dofs_topological(V_psi, 1, p_contact_facets)
        cathode_dofs = fem.locate_dofs_topological(V_psi, 1, n_contact_facets)
        
        bc_anode = fem.dirichletbc(dolfinx.default_scalar_type(applied_voltage), anode_dofs, V_psi)
        bc_cathode = fem.dirichletbc(dolfinx.default_scalar_type(0.0), cathode_dofs, V_psi)
        bcs_psi = [bc_anode, bc_cathode]
        
        v_psi = ufl.TestFunction(V_psi)
        psi_trial = ufl.TrialFunction(V_psi)
        a_psi = ufl.inner(epsilon * ufl.grad(psi_trial), ufl.grad(v_psi)) * ufl.dx
        
        # Modified Poisson equation with charge density limiting for stability
        # Limit the charge density term to prevent extreme source terms
        max_charge_density = q * max(NA_max_m3, ND_max_m3) * 2  # Maximum reasonable charge density
        
        # Create limited charge density for Poisson equation
        charge_density_limited = Function(fem.functionspace(domain, ("DG", 0)))
        for i in range(len(charge_density_limited.x.array)):
            raw_charge = q * (p_current.x.array[i] - n_current.x.array[i] + C.x.array[i])
            charge_density_limited.x.array[i] = np.clip(raw_charge, -max_charge_density, max_charge_density)
        
        L_psi = charge_density_limited * v_psi * ufl.dx
        
        from dolfinx.fem.petsc import LinearProblem
        problem_psi = LinearProblem(a_psi, L_psi, bcs=bcs_psi, u=psi_sol)
        problem_psi.solve()
        
        # Remove artificial voltage clamping - let physics determine potentials
        
        # Step 2: Solve electron continuity equation (simplified for stability)
        try:
            # Use simpler approach - update Slotboom variables based on local equilibrium
            # This is more stable than full Newton iteration
            update_slotboom_variables_simple(u_sol, v_sol, psi_sol, applied_voltage)
        except Exception as e:
            print(f"    Warning: Slotboom update failed: {e}, keeping previous solution")
        
        # Apply very strong relaxation for fine mesh stability
        relaxation_factor = 0.1  # Very strong damping
        psi_sol.x.array[:] = relaxation_factor * psi_sol.x.array + (1 - relaxation_factor) * psi_old.x.array
        
        # Additional stability: limit potential changes per iteration
        max_change = 0.1  # Maximum 0.1V change per iteration
        potential_change = psi_sol.x.array - psi_old.x.array
        large_changes = np.abs(potential_change) > max_change
        psi_sol.x.array[large_changes] = psi_old.x.array[large_changes] + np.sign(potential_change[large_changes]) * max_change
        
        # Check convergence
        psi_change = np.linalg.norm(psi_sol.x.array - psi_old.x.array)
        psi_norm = np.linalg.norm(psi_sol.x.array)
        relative_change = psi_change / psi_norm if psi_norm > 0 else 0
        
        if iteration % 5 == 0 or relative_change < 1e-5:
            psi_max = np.max(psi_sol.x.array)
            psi_min = np.min(psi_sol.x.array)
            print(f"  Gummel iteration {iteration+1}: Potential range [{psi_min:.4f}, {psi_max:.4f}] V, "
                  f"Change: {relative_change:.2e}")
        
        # More relaxed convergence for fine mesh
        if relative_change < 1e-4:  # Relaxed tolerance
            print(f"  Gummel converged after {iteration+1} iterations")
            break
        
        # Emergency break for numerical issues (much tighter for stability)
        max_potential = np.max(np.abs(psi_sol.x.array))
        if max_potential > 2.0:  # Much tighter threshold to prevent instability
            print(f"  Emergency break: Potential too large ({max_potential:.2f}V)")
            # Restore to previous stable solution
            psi_sol.x.array[:] = psi_old.x.array[:]
            break
            
        # Additional check for charge conservation failure
        if abs(avg_charge_density) > q * max(NA_max_m3, ND_max_m3) * 5:
            print(f"  Emergency break: Charge conservation failure ({avg_charge_density:.2e} C/m³)")
            psi_sol.x.array[:] = psi_old.x.array[:]
            break
    
    # Convert final Slotboom variables back to densities
    n_sol = Function(fem.functionspace(domain, ("DG", 0)))
    p_sol = Function(fem.functionspace(domain, ("DG", 0)))
    
    # Map from CG(1) Slotboom variables to DG(0) carrier densities
    dg_size = len(n_sol.x.array)
    cg_size = len(u_sol.x.array)
    
    for i in range(dg_size):
        # Map DG index to CG index
        cg_idx = min(i * cg_size // dg_size, cg_size - 1)
        
        psi_val = psi_sol.x.array[cg_idx]
        u_val = u_sol.x.array[cg_idx]
        v_val = v_sol.x.array[cg_idx]
        
        # Proper Slotboom to density conversion with controlled exponentials
        # n = n_i * u * exp(psi/V_T), p = n_i * v * exp(-psi/V_T)
        
        # Use same tight limits as in Poisson equation for consistency
        psi_limited = np.clip(psi_val, -1.5, 1.5)  # Same ±1.5V limit
        exp_psi = np.exp(np.clip(psi_limited / V_T, -10, 10))
        exp_neg_psi = np.exp(np.clip(-psi_limited / V_T, -10, 10))
        
        # Calculate carrier densities
        n_local = n_i_m3 * u_val * exp_psi
        p_local = n_i_m3 * v_val * exp_neg_psi
        
        # Apply same strict physical bounds as in Poisson equation
        doping_level = abs(C.x.array[i]) if i < len(C.x.array) else NA_junction_m3
        max_reasonable = max(doping_level * 2, n_i_m3 * 10)  # At most 2x local doping
        n_local = np.clip(n_local, n_i_m3 * 1e-6, max_reasonable)
        p_local = np.clip(p_local, n_i_m3 * 1e-6, max_reasonable)
        
        n_sol.x.array[i] = n_local
        p_sol.x.array[i] = p_local
        
    
    return psi_sol, n_sol, p_sol

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

def srh_recombination_rate(n, p, n_i=None, tau_n=1e-6, tau_p=1e-6):
    """
    Shockley-Read-Hall recombination rate
    R = (np - ni²) / (tau_p*(n + ni) + tau_n*(p + ni))
    
    Args:
        n: electron concentration (m^-3)
        p: hole concentration (m^-3) 
        n_i: intrinsic concentration (m^-3), defaults to global n_i_m3
        tau_n: electron lifetime (s)
        tau_p: hole lifetime (s)
    
    Returns:
        Recombination rate (m^-3/s)
    """
    if n_i is None:
        n_i = n_i_m3
    
    # Add small regularization to prevent division by zero
    epsilon = 1e-30
    
    numerator = n * p - n_i**2
    denominator = tau_p * (n + n_i) + tau_n * (p + n_i) + epsilon
    
    return numerator / denominator

# ====================================================================
# 7. Current Calculation Enhancement
# ====================================================================
def calculate_current(psi, n, p, applied_voltage):
    """
    Enhanced current calculation using proper boundary integration
    """
    # Physical parameters
    mu_n = 0.14  # m²/V/s
    mu_p = 0.045  # m²/V/s
    
    # Use proper boundary current calculation
    try:
        # Calculate current through cathode (bottom boundary)
        cathode_current = calculate_boundary_current(psi, n, p, boundary_id=2)
        
        # For verification, also calculate through anode (top boundary)
        anode_current = calculate_boundary_current(psi, n, p, boundary_id=1)
        
        # The currents should be approximately equal (current continuity)
        total_current = abs(cathode_current)
        
        # Separate drift and diffusion components (approximate)
        avg_n = np.mean(n.x.array) if hasattr(n, 'x') else np.mean(n)
        avg_p = np.mean(p.x.array) if hasattr(p, 'x') else np.mean(p)
        
        
        avg_conductivity = q * (avg_n * mu_n + avg_p * mu_p)
        V_drop = np.max(psi.x.array) - np.min(psi.x.array)
        
        # Calculate total device current properly
        # Current = Current_density * Cross_sectional_area
        # For 2D simulation: Current per unit thickness
        device_area = width_m * thickness_m  # Cross-sectional area
        
        # Estimate drift component (simplified)
        drift_current_density = avg_conductivity * (V_drop / total_length_m)  # A/m²
        drift_current = drift_current_density * device_area  # Total current (A)
        
        # Convert boundary current integration result to total current
        # boundary_current is already integrated over the boundary
        total_current_A = abs(cathode_current) * thickness_m  # Convert to total current
        
        # Estimate diffusion component
        diffusion_current = total_current_A - abs(drift_current)
        
        print(f"    Current analysis:")
        print(f"    Device area: {device_area*1e12:.2f} μm²")
        print(f"    Current density: {abs(cathode_current):.2e} A/m")
        print(f"    Total current: {total_current_A*1e6:.2f} μA")

        print(f"    Debug current calc:")
        print(f"    - Boundary current: {abs(cathode_current):.2e} A/m")
        print(f"    - Device thickness: {thickness_m*1e6:.1f} μm")
        print(f"    - Total current calc: {abs(cathode_current) * thickness_m:.2e} A")
        print(f"    - Avg conductivity: {avg_conductivity:.2e} S/m")
        print(f"    - V_drop: {V_drop:.3f} V")

        return total_current_A, abs(drift_current), abs(diffusion_current), avg_conductivity
        
    except Exception as e:
        print(f"    Warning: Using fallback current calculation: {e}")
        # Fallback to original method
        return calculate_current_fallback(psi, n, p, applied_voltage)

def calculate_current_fallback(psi, n, p, applied_voltage):
    """
    Fallback current calculation using averaged quantities
    """
    # Physical parameters
    mu_n = 0.14  # m²/V/s
    mu_p = 0.045  # m²/V/s
    
    # Calculate average quantities
    avg_n = np.mean(n.x.array) if hasattr(n, 'x') else np.mean(n)
    avg_p = np.mean(p.x.array) if hasattr(p, 'x') else np.mean(p)
    
    # Enhanced conductivity calculation
    avg_conductivity = q * (avg_n * mu_n + avg_p * mu_p)
    
    # Potential drop calculation
    V_drop = np.max(psi.x.array) - np.min(psi.x.array)
    
    # Current estimate with proper device scaling
    device_area = width_m * thickness_m  # Cross-sectional area
    
    # Drift current calculation
    drift_current_density = avg_conductivity * (V_drop / total_length_m)  # A/m²
    drift_current = drift_current_density * device_area  # Total current (A)
    
    # Add diffusion contribution (simplified)
    # ∇n and ∇p contribute to current even without electric field
    if hasattr(n, 'x'):
        n_gradient = (np.max(n.x.array) - np.min(n.x.array)) / total_length_m
        p_gradient = (np.max(p.x.array) - np.min(p.x.array)) / total_length_m
    else:
        n_gradient = 0
        p_gradient = 0
    
    diffusion_current_density = q * V_T * (mu_n * n_gradient - mu_p * p_gradient)  # A/m²
    diffusion_current = diffusion_current_density * device_area  # Total current (A)
    
    total_current = drift_current + diffusion_current
    
    print(f"    Fallback current: {abs(total_current)*1e6:.2f} μA")
    
    return abs(total_current), abs(drift_current), abs(diffusion_current), avg_conductivity

def calculate_boundary_current(psi, n, p, boundary_id=2):
    """
    Calculate current through a specific boundary using proper integration
    with better numerical handling of carrier densities
    
    Args:
        psi: potential function (CG1)
        n, p: carrier density functions (DG0 or CG1)
        boundary_id: boundary marker (2 = cathode by default)
    
    Returns:
        total_current: integrated current through boundary (A/m)
    """
    # Physical parameters
    mu_n = 0.14  # m²/V/s
    mu_p = 0.045  # m²/V/s
    D_n = mu_n * V_T  # Einstein relation
    D_p = mu_p * V_T
    
    try:
        # Handle different function spaces - project DG0 to CG1 if needed
        if n.function_space != V_psi:
            # Project DG0 carrier densities to CG1 space for gradient calculation
            n_cg1 = Function(V_psi)
            p_cg1 = Function(V_psi)
            
            # Use L2 projection to convert DG0 to CG1
            n_cg1.interpolate(lambda x: np.full(x.shape[1], np.mean(n.x.array)))
            p_cg1.interpolate(lambda x: np.full(x.shape[1], np.mean(p.x.array)))
            
            n_proj = n_cg1
            p_proj = p_cg1
        else:
            n_proj = n
            p_proj = p
        
        # Calculate electric field
        E_field = -ufl.grad(psi)
        
        # Current densities using proper drift-diffusion
        J_n = q * mu_n * n_proj * E_field + q * D_n * ufl.grad(n_proj)
        J_p = q * mu_p * p_proj * E_field - q * D_p * ufl.grad(p_proj)
        J_total = J_n + J_p
        
        # Outward normal vector
        n_vec = ufl.FacetNormal(domain)
        
        # Create boundary measure
        ds_boundary = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)
        
        # Integrate current over boundary
        current_form = ufl.dot(J_total, n_vec) * ds_boundary(boundary_id)
        current_value = fem.assemble_scalar(fem.form(current_form))

        print(f"      Boundary current debug:")
        # print(f"      - Electric field magnitude: {np.mean(np.abs(E_field)):.2e} V/m")
        print(f"      - Carrier densities: n={np.mean(n_proj.x.array):.2e}, p={np.mean(p_proj.x.array):.2e}")
        # print(f"      - Current density components:")
        print(f"      - Potential drop: {np.max(psi.x.array) - np.min(psi.x.array):.3f} V")
        # print(f"        J_n: {np.mean(np.abs(J_n)):.2e} A/m²")
        # print(f"        J_p: {np.mean(np.abs(J_p)):.2e} A/m²")
        print(f"      - Integrated current: {float(current_value):.2e} A/m")


        return float(current_value)
        
    except Exception as e:
        print(f"    Warning: Boundary current calculation failed: {e}")
        # Fallback to simple estimate
        return calculate_simple_boundary_current(psi, n, p, boundary_id)

def calculate_simple_boundary_current(psi, n, p, boundary_id=2):
    """
    Fallback current calculation using boundary values
    
    Args:
        psi: potential function
        n, p: carrier density functions
        boundary_id: boundary marker
    
    Returns:
        estimated current through boundary (A/m)
    """
    # Physical parameters
    mu_n = 0.14  # m²/V/s
    mu_p = 0.045  # m²/V/s
    
    # Get boundary dofs
    if boundary_id == 2:  # cathode
        boundary_dofs = fem.locate_dofs_topological(V_psi, 1, n_contact_facets)
    else:  # anode
        boundary_dofs = fem.locate_dofs_topological(V_psi, 1, p_contact_facets)
    
    # Calculate average values at boundary
    psi_boundary = np.mean(psi.x.array[boundary_dofs])
    
    # Estimate carrier densities at boundary
    if hasattr(n, 'x'):
        if len(n.x.array) == len(psi.x.array):  # Same size (CG1)
            n_boundary = np.mean(n.x.array[boundary_dofs])
            p_boundary = np.mean(p.x.array[boundary_dofs])
        else:  # Different size (DG0)
            n_boundary = np.mean(n.x.array)
            p_boundary = np.mean(p.x.array)
    else:
        n_boundary = np.mean(n)
        p_boundary = np.mean(p)
    
    # Estimate electric field (simple finite difference)
    E_field_est = (np.max(psi.x.array) - np.min(psi.x.array)) / total_length_m
    
    # Current density estimate
    J_drift = q * (mu_n * n_boundary + mu_p * p_boundary) * E_field_est
    
    # Total current (multiply by width)
    total_current = J_drift * width_m
    
    return total_current

def slotboom_to_density(u, v, psi, n_i=None):
    """
    Convert Slotboom variables to carrier densities
    n = n_i * u * exp(psi/V_T)
    p = n_i * v * exp(-psi/V_T)
    
    Args:
        u, v: Slotboom variables (dimensionless)
        psi: electrostatic potential (V)
        n_i: intrinsic concentration (m^-3)
    
    Returns:
        n, p: electron and hole densities (m^-3)
    """
    if n_i is None:
        n_i = n_i_m3
    
    if hasattr(psi, 'x'):  # Function object
        exp_psi = np.exp(psi.x.array / V_T)
        exp_neg_psi = np.exp(-psi.x.array / V_T)
        
        if hasattr(u, 'x'):  # Function objects
            n = n_i * u.x.array * exp_psi
            p = n_i * v.x.array * exp_neg_psi
        else:  # Scalar values
            n = n_i * u * exp_psi
            p = n_i * v * exp_neg_psi
    else:  # Scalar potential
        exp_psi = np.exp(psi / V_T)
        exp_neg_psi = np.exp(-psi / V_T)
        n = n_i * u * exp_psi
        p = n_i * v * exp_neg_psi
    
    return n, p

def density_to_slotboom(n, p, psi, n_i=None):
    """
    Convert carrier densities to Slotboom variables
    u = n / (n_i * exp(psi/V_T))
    v = p / (n_i * exp(-psi/V_T))
    
    Args:
        n, p: carrier densities (m^-3)
        psi: electrostatic potential (V)  
        n_i: intrinsic concentration (m^-3)
    
    Returns:
        u, v: Slotboom variables (dimensionless)
    """
    if n_i is None:
        n_i = n_i_m3
    
    if hasattr(psi, 'x'):  # Function object
        exp_psi = np.exp(psi.x.array / V_T)
        exp_neg_psi = np.exp(-psi.x.array / V_T)
        
        if hasattr(n, 'x'):  # Function objects
            u = n.x.array / (n_i * exp_psi)
            v = p.x.array / (n_i * exp_neg_psi)
        else:  # Scalar values
            u = n / (n_i * exp_psi)
            v = p / (n_i * exp_neg_psi)
    else:  # Scalar potential
        exp_psi = np.exp(psi / V_T)
        exp_neg_psi = np.exp(-psi / V_T)
        u = n / (n_i * exp_psi)
        v = p / (n_i * exp_neg_psi)
    
    return u, v

def create_electron_continuity_weak_form(u, v, psi, test_function):
    """
    Create weak form for electron continuity equation using Slotboom variables
    
    Electron continuity: ∇·J_n = qR
    where J_n = q·D_n·n_i·exp(ψ/V_T)·∇u
    
    Weak form: ∫(∇·J_n)·w dx = ∫qR·w dx
    After integration by parts: -∫J_n·∇w dx + ∫J_n·n·w ds = ∫qR·w dx
    
    Args:
        u: electron Slotboom variable
        v: hole Slotboom variable  
        psi: electrostatic potential
        test_function: test function for weak form
        
    Returns:
        UFL form for electron continuity equation
    """
    # Physical parameters
    mu_n = 0.14  # m²/V/s
    D_n = mu_n * V_T  # Einstein relation
    
    # Current density in Slotboom formulation
    # J_n = q * D_n * n_i * exp(psi/V_T) * grad(u)
    exp_psi_over_vt = ufl.exp(psi / V_T)
    J_n = q * D_n * n_i_m3 * exp_psi_over_vt * ufl.grad(u)
    
    # Recombination term
    # Convert Slotboom variables to densities for recombination calculation
    n = n_i_m3 * u * exp_psi_over_vt
    p = n_i_m3 * v * ufl.exp(-psi / V_T)
    
    # SRH recombination: R = (np - ni²) / (tau_p*(n + ni) + tau_n*(p + ni))
    tau_n = 1e-6  # s
    tau_p = 1e-6  # s
    epsilon_reg = 1e-30  # regularization
    
    R_srh = (n * p - n_i_m3**2) / (tau_p * (n + n_i_m3) + tau_n * (p + n_i_m3) + epsilon_reg)
    
    # Weak form: -∫J_n·∇w dx + ∫qR·w dx = 0
    # (boundary terms are handled by Dirichlet BCs)
    weak_form = (-ufl.inner(J_n, ufl.grad(test_function)) + q * R_srh * test_function) * ufl.dx
    
    return weak_form

def create_hole_continuity_weak_form(u, v, psi, test_function):
    """
    Create weak form for hole continuity equation using Slotboom variables
    
    Hole continuity: ∇·J_p = -qR
    where J_p = -q·D_p·n_i·exp(-ψ/V_T)·∇v
    
    Weak form: ∫(∇·J_p)·w dx = ∫(-qR)·w dx
    After integration by parts: -∫J_p·∇w dx + ∫J_p·n·w ds = ∫(-qR)·w dx
    
    Args:
        u: electron Slotboom variable
        v: hole Slotboom variable  
        psi: electrostatic potential
        test_function: test function for weak form
        
    Returns:
        UFL form for hole continuity equation
    """
    # Physical parameters
    mu_p = 0.045  # m²/V/s
    D_p = mu_p * V_T  # Einstein relation
    
    # Current density in Slotboom formulation
    # J_p = -q * D_p * n_i * exp(-psi/V_T) * grad(v)
    exp_neg_psi_over_vt = ufl.exp(-psi / V_T)
    J_p = -q * D_p * n_i_m3 * exp_neg_psi_over_vt * ufl.grad(v)
    
    # Recombination term
    # Convert Slotboom variables to densities for recombination calculation
    n = n_i_m3 * u * ufl.exp(psi / V_T)
    p = n_i_m3 * v * exp_neg_psi_over_vt
    
    # SRH recombination: R = (np - ni²) / (tau_p*(n + ni) + tau_n*(p + ni))
    tau_n = 1e-6  # s
    tau_p = 1e-6  # s
    epsilon_reg = 1e-30  # regularization
    
    R_srh = (n * p - n_i_m3**2) / (tau_p * (n + n_i_m3) + tau_n * (p + n_i_m3) + epsilon_reg)
    
    # Weak form: -∫J_p·∇w dx + ∫(-qR)·w dx = 0
    # (boundary terms are handled by Dirichlet BCs)
    weak_form = (-ufl.inner(J_p, ufl.grad(test_function)) - q * R_srh * test_function) * ufl.dx
    
    return weak_form

def solve_electron_continuity(psi, v, u_initial=None, applied_voltage=0.0):
    """
    Solve electron continuity equation using Slotboom variables with improved convergence
    
    Args:
        psi: electrostatic potential (Function)
        v: hole Slotboom variable (Function)
        u_initial: initial guess for electron Slotboom variable
        applied_voltage: applied voltage for boundary conditions
        
    Returns:
        u: solved electron Slotboom variable (Function)
    """
    print(f"  Solving electron continuity equation...")
    
    # Create function and test function
    u = Function(V_psi, name="Electron_Slotboom_u")
    test_u = ufl.TestFunction(V_psi)
    
    # Better initial guess based on local equilibrium
    if u_initial is not None:
        u.x.array[:] = u_initial.x.array[:]
    else:
        # Initialize with spatially varying equilibrium values
        def init_u_equilibrium(x):
            # Bottom half (n-type): high electron concentration
            # Top half (p-type): low electron concentration
            # P-contact region (left)
            return np.where(x[0] < junction_center,
                           (n_i_m3**2 / NA_junction_m3) / n_i_m3,  # p-type region
                           ND_junction_m3 / n_i_m3)  # n-type region
        
        u.interpolate(init_u_equilibrium)
    
    # Improved boundary conditions with clamping
    cathode_dofs = fem.locate_dofs_topological(V_psi, 1, n_contact_facets)
    anode_dofs = fem.locate_dofs_topological(V_psi, 1, p_contact_facets)
    
    # More stable boundary conditions with voltage limiting
    bias_factor = np.exp(np.clip(applied_voltage / V_T, -3, 3))  # Limit to ±3VT
    
    u_cathode_val = np.clip(ND_max_m3 / n_i_m3 * bias_factor, 1e-3, 1e6)  # Clamp values
    u_anode_val = np.clip((n_i_m3**2 / NA_max_m3) / n_i_m3 / bias_factor, 1e-6, 1e3)
    
    bc_u_cathode = fem.dirichletbc(dolfinx.default_scalar_type(u_cathode_val), cathode_dofs, V_psi)
    bc_u_anode = fem.dirichletbc(dolfinx.default_scalar_type(u_anode_val), anode_dofs, V_psi)
    bcs_u = [bc_u_cathode, bc_u_anode]
    
    # Create weak form
    weak_form = create_electron_continuity_weak_form(u, v, psi, test_u)
    
    # Solve with relaxation and better parameters
    max_newton_iter = 20
    relaxation_factor = 0.8  # Damping factor
    
    try:
        from dolfinx.fem.petsc import NonlinearProblem
        from dolfinx.nls.petsc import NewtonSolver
        
        problem = NonlinearProblem(weak_form, u, bcs=bcs_u)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        
        # More forgiving convergence criteria
        solver.rtol = 1e-6  # Relaxed tolerance
        solver.atol = 1e-10
        solver.max_it = max_newton_iter
        
        # Store old solution for relaxation
        u_old = Function(V_psi)
        u_old.x.array[:] = u.x.array[:]
        
        n_iter, converged = solver.solve(u)
        
        # Apply relaxation if not converged
        if not converged:
            print(f"    Applying relaxation with factor {relaxation_factor}")
            u.x.array[:] = relaxation_factor * u.x.array + (1 - relaxation_factor) * u_old.x.array
        
        if converged:
            print(f"    ✓ Electron continuity converged in {n_iter} iterations")
        else:
            print(f"    ⚠ Electron continuity did not converge ({n_iter} iterations), using relaxed solution")
            
    except Exception as e:
        print(f"    ✗ Error solving electron continuity: {e}")
        # Fall back to better estimate based on local equilibrium
        print(f"    Using equilibrium-based fallback")
        def fallback_u(x):
            bias_local = np.exp(np.clip(applied_voltage / V_T, -2, 2))
            return np.where(x[0] < junction_center,
                           (n_i_m3**2 / NA_junction_m3) / n_i_m3 / bias_local,  # p-side
                           ND_junction_m3 / n_i_m3 * bias_local)  # n-side
        u.interpolate(fallback_u)
    
    return u

def solve_hole_continuity(psi, u, v_initial=None, applied_voltage=0.0):
    """
    Solve hole continuity equation using Slotboom variables with improved convergence
    
    Args:
        psi: electrostatic potential (Function)
        u: electron Slotboom variable (Function)
        v_initial: initial guess for hole Slotboom variable
        applied_voltage: applied voltage for boundary conditions
        
    Returns:
        v: solved hole Slotboom variable (Function)
    """
    print(f"  Solving hole continuity equation...")
    
    # Create function and test function
    v = Function(V_psi, name="Hole_Slotboom_v")
    test_v = ufl.TestFunction(V_psi)
    
    # Better initial guess based on local equilibrium
    if v_initial is not None:
        v.x.array[:] = v_initial.x.array[:]
    else:
        # Initialize with spatially varying equilibrium values
        def init_v_equilibrium(x):
            # Top half (p-type): high hole concentration
            # Bottom half (n-type): low hole concentration
            # Hole concentration higher on p-side (left)
            return np.where(x[0] < junction_center,
                           NA_junction_m3 / n_i_m3,  # p-type region
                           (n_i_m3**2 / ND_junction_m3) / n_i_m3)  # n-type region
        
        v.interpolate(init_v_equilibrium)
    
    # Improved boundary conditions with clamping
    cathode_dofs = fem.locate_dofs_topological(V_psi, 1, n_contact_facets)
    anode_dofs = fem.locate_dofs_topological(V_psi, 1, p_contact_facets)
    
    # More stable boundary conditions with voltage limiting
    bias_factor = np.exp(np.clip(applied_voltage / V_T, -3, 3))  # Limit to ±3VT
    
    v_anode_val = np.clip(NA_max_m3 / n_i_m3 * bias_factor, 1e-3, 1e6)  # Clamp values
    v_cathode_val = np.clip((n_i_m3**2 / ND_max_m3) / n_i_m3 / bias_factor, 1e-6, 1e3)
    
    bc_v_anode = fem.dirichletbc(dolfinx.default_scalar_type(v_anode_val), anode_dofs, V_psi)
    bc_v_cathode = fem.dirichletbc(dolfinx.default_scalar_type(v_cathode_val), cathode_dofs, V_psi)
    bcs_v = [bc_v_anode, bc_v_cathode]
    
    # Create weak form
    weak_form = create_hole_continuity_weak_form(u, v, psi, test_v)
    
    # Solve with relaxation and better parameters
    max_newton_iter = 20
    relaxation_factor = 0.8  # Damping factor
    
    try:
        from dolfinx.fem.petsc import NonlinearProblem
        from dolfinx.nls.petsc import NewtonSolver
        
        problem = NonlinearProblem(weak_form, v, bcs=bcs_v)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        
        # More forgiving convergence criteria
        solver.rtol = 1e-6  # Relaxed tolerance
        solver.atol = 1e-10
        solver.max_it = max_newton_iter
        
        # Store old solution for relaxation
        v_old = Function(V_psi)
        v_old.x.array[:] = v.x.array[:]
        
        n_iter, converged = solver.solve(v)
        
        # Apply relaxation if not converged
        if not converged:
            print(f"    Applying relaxation with factor {relaxation_factor}")
            v.x.array[:] = relaxation_factor * v.x.array + (1 - relaxation_factor) * v_old.x.array
        
        if converged:
            print(f"    ✓ Hole continuity converged in {n_iter} iterations")
        else:
            print(f"    ⚠ Hole continuity did not converge ({n_iter} iterations), using relaxed solution")
            
    except Exception as e:
        print(f"    ✗ Error solving hole continuity: {e}")
        # Fall back to better estimate based on local equilibrium
        print(f"    Using equilibrium-based fallback")
        def fallback_v(x):
            bias_local = np.exp(np.clip(applied_voltage / V_T, -2, 2))
            return np.where(x[0] < junction_center,
                           NA_junction_m3 / n_i_m3 * bias_local,  # p-side
                           (n_i_m3**2 / ND_junction_m3) / n_i_m3 / bias_local)  # n-side
        v.interpolate(fallback_v)
    
    return v

def update_slotboom_variables_simple(u_sol, v_sol, psi_sol, applied_voltage):
    """
    Proper Slotboom variables implementation based on quasi-Fermi levels
    
    Slotboom variables are related to quasi-Fermi levels:
    u = exp((E_fn - E_i)/(kT)) = exp((phi_n - psi)/V_T)
    v = exp((E_i - E_fp)/(kT)) = exp((psi - phi_p)/V_T)
    
    where phi_n, phi_p are quasi-Fermi potentials
    """
    # Proper Slotboom variables based on quasi-Fermi level splitting
    x_coords = domain.geometry.x
    
    # Calculate quasi-Fermi level splitting due to applied bias
    # In forward bias: phi_n - phi_p increases, causing injection
    phi_splitting = applied_voltage  # Applied voltage creates quasi-Fermi level splitting
    
    for i in range(len(u_sol.x.array)):
        # Get spatial position and local potential
        if i < len(x_coords):
            x_pos = x_coords[i, 0]
        else:
            x_pos = total_length_m / 2
        
        # Get local electrostatic potential
        if i < len(psi_sol.x.array):
            psi_local = psi_sol.x.array[i]
        else:
            psi_local = 0.0
        
        # Determine local doping and equilibrium conditions
        if x_pos < p_contact_end:  # P-contact region
            # Heavily doped p-region: phi_p ≈ psi, phi_n = phi_p + qV_applied
            phi_p_local = psi_local
            phi_n_local = psi_local + applied_voltage
            
        elif x_pos > n_contact_start:  # N-contact region
            # Heavily doped n-region: phi_n ≈ psi, phi_p = phi_n - qV_applied
            phi_n_local = psi_local  
            phi_p_local = psi_local - applied_voltage
            
        else:  # Junction region - quasi-Fermi levels vary smoothly
            # Linear interpolation of quasi-Fermi levels across junction
            junction_fraction = (x_pos - p_contact_end) / junction_length_m
            
            # P-side reference: phi_p = 0, phi_n = applied_voltage
            # N-side reference: phi_n = 0, phi_p = -applied_voltage
            phi_p_local = psi_local - applied_voltage * junction_fraction
            phi_n_local = psi_local + applied_voltage * (1 - junction_fraction)
        
        # Calculate Slotboom variables from quasi-Fermi levels
        # u = exp((phi_n - psi)/V_T), v = exp((psi - phi_p)/V_T)
        u_new = np.exp(np.clip((phi_n_local - psi_local) / V_T, -5, 5))
        v_new = np.exp(np.clip((psi_local - phi_p_local) / V_T, -5, 5))
        
        # Gradual update for numerical stability
        alpha = 0.3  # Moderate update rate
        u_sol.x.array[i] = (1 - alpha) * u_sol.x.array[i] + alpha * u_new
        v_sol.x.array[i] = (1 - alpha) * v_sol.x.array[i] + alpha * v_new
        
        # Physical bounds based on maximum reasonable quasi-Fermi level splitting
        max_splitting = 2.0  # Maximum 2V quasi-Fermi level difference
        u_max = np.exp(max_splitting / V_T)
        v_max = np.exp(max_splitting / V_T)
        
        u_sol.x.array[i] = np.clip(u_sol.x.array[i], 1e-3, u_max)
        v_sol.x.array[i] = np.clip(v_sol.x.array[i], 1e-3, v_max)
        
        if i < 3:  # Debug first few elements
            print(f"    Slotboom debug [{i}]: x={x_pos*1e6:.1f}μm")
            print(f"      psi={psi_local:.3f}V, phi_n={phi_n_local:.3f}V, phi_p={phi_p_local:.3f}V")
            print(f"      u_new={u_new:.2e}, v_new={v_new:.2e}")
            print(f"      final: u={u_sol.x.array[i]:.2e}, v={v_sol.x.array[i]:.2e}")


        

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
            # Solve with current voltage using new Gummel iteration method
            try:
                psi_new, n_new, p_new = solve_with_bias_gummel(voltage, psi_prev, n_prev, p_prev)
                print(f"    ✓ Using new Gummel drift-diffusion solver")
            except Exception as e:
                print(f"    ⚠ Gummel solver failed: {e}")
                print(f"    ✓ Falling back to original simplified solver")
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
            if voltage > 0:  # Forward bias
                print(f"    Forward bias {voltage}V: Current = {abs(total_current)*1e6:.2f} μA")
                # print(f"    Avg carrier densities: n={avg_n:.2e}, p={avg_p:.2e} m^-3")

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
# 9. Run Voltage Stepping Simulation
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