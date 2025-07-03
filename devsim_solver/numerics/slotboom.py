"""
Slotboom variable transformations for semiconductor device simulation

Slotboom variables provide numerical stability for drift-diffusion equations
by transforming exponentially varying carrier densities into smoother variables.

The transformation is:
u = n / (n_i * exp(ψ/V_T))  (electron Slotboom variable)
v = p / (n_i * exp(-ψ/V_T)) (hole Slotboom variable)

This eliminates exponential terms from the transport equations.
"""

import numpy as np
import ufl
from ..physics.constants import n_i_m3, V_T

def density_to_slotboom(n, p, psi, n_i=None):
    """
    Convert carrier densities to Slotboom variables
    
    u = n / (n_i * exp(ψ/V_T))
    v = p / (n_i * exp(-ψ/V_T))
    
    Args:
        n, p: carrier densities (m^-3)
        psi: electrostatic potential (V)  
        n_i: intrinsic concentration (m^-3), defaults to global value
    
    Returns:
        (u, v): Slotboom variables (dimensionless)
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

def slotboom_to_density(u, v, psi, n_i=None):
    """
    Convert Slotboom variables to carrier densities
    
    n = n_i * u * exp(ψ/V_T)
    p = n_i * v * exp(-ψ/V_T)
    
    Args:
        u, v: Slotboom variables (dimensionless)
        psi: electrostatic potential (V)
        n_i: intrinsic concentration (m^-3), defaults to global value
    
    Returns:
        (n, p): electron and hole densities (m^-3)
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

def slotboom_to_density_ufl(u, v, psi, n_i=None):
    """
    UFL version for use in variational forms
    
    Args:
        u, v: Slotboom variables (UFL expressions)
        psi: electrostatic potential (UFL expression)
        n_i: intrinsic concentration (scalar), defaults to global value
    
    Returns:
        (n, p): UFL expressions for carrier densities
    """
    if n_i is None:
        n_i = n_i_m3
    
    exp_psi_over_vt = ufl.exp(psi / V_T)
    exp_neg_psi_over_vt = ufl.exp(-psi / V_T)
    
    n = n_i * u * exp_psi_over_vt
    p = n_i * v * exp_neg_psi_over_vt
    
    return n, p

def equilibrium_slotboom_values(doping_concentration, doping_type='n', n_i=None):
    """
    Calculate equilibrium Slotboom variable values for given doping
    
    For n-type: n ≈ N_D, p ≈ n_i²/N_D
    For p-type: p ≈ N_A, n ≈ n_i²/N_A
    At equilibrium with ψ = 0: u = n/n_i, v = p/n_i
    
    Args:
        doping_concentration: doping level (m^-3)
        doping_type: 'n' for n-type, 'p' for p-type
        n_i: intrinsic concentration (m^-3)
    
    Returns:
        (u_eq, v_eq): equilibrium Slotboom variables
    """
    if n_i is None:
        n_i = n_i_m3
    
    if doping_type.lower() == 'n':
        # n-type material
        n_eq = doping_concentration
        p_eq = n_i**2 / doping_concentration
    elif doping_type.lower() == 'p':
        # p-type material
        p_eq = doping_concentration
        n_eq = n_i**2 / doping_concentration
    else:
        raise ValueError("doping_type must be 'n' or 'p'")
    
    # At equilibrium (ψ = 0)
    u_eq = n_eq / n_i
    v_eq = p_eq / n_i
    
    return u_eq, v_eq

def slotboom_current_densities_ufl(u, v, psi, grad_u, grad_v, n_i=None):
    """
    Calculate current densities in Slotboom formulation for UFL
    
    J_n = q * D_n * n_i * exp(ψ/V_T) * ∇u
    J_p = -q * D_p * n_i * exp(-ψ/V_T) * ∇v
    
    Args:
        u, v: Slotboom variables (UFL expressions)
        psi: electrostatic potential (UFL expression)
        grad_u, grad_v: gradients of Slotboom variables (UFL expressions)
        n_i: intrinsic concentration (scalar)
    
    Returns:
        (J_n, J_p): UFL expressions for current densities
    """
    from ..physics.constants import q, D_n, D_p
    
    if n_i is None:
        n_i = n_i_m3
    
    exp_psi_over_vt = ufl.exp(psi / V_T)
    exp_neg_psi_over_vt = ufl.exp(-psi / V_T)
    
    J_n = q * D_n * n_i * exp_psi_over_vt * grad_u
    J_p = -q * D_p * n_i * exp_neg_psi_over_vt * grad_v
    
    return J_n, J_p

def boundary_slotboom_values(applied_voltage, contact_type='n', n_i=None):
    """
    Calculate Slotboom variable boundary values for biased contacts
    
    Args:
        applied_voltage: voltage applied to contact (V)
        contact_type: 'n' for n-type contact, 'p' for p-type contact
        n_i: intrinsic concentration (m^-3)
    
    Returns:
        (u_boundary, v_boundary): boundary values for Slotboom variables
    """
    from ..physics.constants import NA_max_m3, ND_max_m3
    
    if n_i is None:
        n_i = n_i_m3
    
    # Bias factor
    bias_factor = np.exp(np.clip(applied_voltage / V_T, -10, 10))
    
    if contact_type.lower() == 'n':
        # n-type contact
        u_boundary = ND_max_m3 / n_i * bias_factor
        v_boundary = n_i / ND_max_m3 / bias_factor
    elif contact_type.lower() == 'p':
        # p-type contact  
        u_boundary = n_i / NA_max_m3 / bias_factor
        v_boundary = NA_max_m3 / n_i * bias_factor
    else:
        raise ValueError("contact_type must be 'n' or 'p'")
    
    return u_boundary, v_boundary

def validate_slotboom_transformation(n_test, p_test, psi_test, tolerance=1e-12):
    """
    Validate round-trip accuracy of Slotboom transformations
    
    Args:
        n_test, p_test: test carrier densities
        psi_test: test potential
        tolerance: acceptable relative error
    
    Returns:
        True if transformation is accurate within tolerance
    """
    # Forward transformation
    u, v = density_to_slotboom(n_test, p_test, psi_test)
    
    # Backward transformation
    n_back, p_back = slotboom_to_density(u, v, psi_test)
    
    # Calculate errors
    n_error = abs(n_back - n_test) / n_test if n_test > 0 else abs(n_back)
    p_error = abs(p_back - p_test) / p_test if p_test > 0 else abs(p_back)
    
    return n_error < tolerance and p_error < tolerance

def slotboom_test():
    """
    Test Slotboom variable transformations
    """
    print("Testing Slotboom variable transformations:")
    
    # Test round-trip conversion
    test_cases = [
        (1e16, 1e14, 0.0),    # Equilibrium
        (1e17, 1e13, 0.5),    # Forward bias
        (1e15, 1e15, -0.3),   # Reverse bias
        (1e20, 1e10, 1.0),    # High injection
    ]
    
    all_passed = True
    for i, (n, p, psi) in enumerate(test_cases):
        valid = validate_slotboom_transformation(n, p, psi)
        status = "✓ PASS" if valid else "✗ FAIL"
        print(f"  Test {i+1}: n={n:.0e}, p={p:.0e}, ψ={psi}V - {status}")
        if not valid:
            all_passed = False
    
    # Test equilibrium values
    u_eq_n, v_eq_n = equilibrium_slotboom_values(1e23, 'n')
    u_eq_p, v_eq_p = equilibrium_slotboom_values(1e23, 'p')
    
    print(f"  Equilibrium values:")
    print(f"    n-type (N_D=1e23): u={u_eq_n:.2e}, v={v_eq_n:.2e}")
    print(f"    p-type (N_A=1e23): u={u_eq_p:.2e}, v={v_eq_p:.2e}")
    
    # Test boundary values
    u_bound, v_bound = boundary_slotboom_values(0.7, 'n')
    print(f"  Boundary values (0.7V, n-type): u={u_bound:.2e}, v={v_bound:.2e}")
    
    return all_passed