"""
Transport models for semiconductor device simulation

Contains mobility models, diffusion coefficients, and transport equations
for electrons and holes in semiconductor materials.
"""

import numpy as np
from .constants import mu_n0, mu_p0, V_T, q

def constant_mobility(n=None, p=None, T=None):
    """
    Constant mobility model (simplest case)
    
    Args:
        n: electron concentration (unused for constant model)
        p: hole concentration (unused for constant model)
        T: temperature (unused for constant model)
    
    Returns:
        (mu_n, mu_p): electron and hole mobilities (m²/V/s)
    """
    return mu_n0, mu_p0

def field_dependent_mobility(E_field, mu_0, v_sat=1e5, beta=2):
    """
    Field-dependent mobility model (velocity saturation)
    
    μ(E) = μ_0 / (1 + (μ_0*E/v_sat)^β)^(1/β)
    
    Args:
        E_field: electric field magnitude (V/m)
        mu_0: low-field mobility (m²/V/s)
        v_sat: saturation velocity (m/s)
        beta: field dependence parameter
    
    Returns:
        Mobility at given field (m²/V/s)
    """
    if np.any(E_field == 0):
        return mu_0
    
    ratio = mu_0 * np.abs(E_field) / v_sat
    return mu_0 / (1 + ratio**beta)**(1/beta)

def concentration_dependent_mobility(n, p, n_ref=1e23, p_ref=1e23, 
                                   mu_n_min=0.05, mu_p_min=0.02,
                                   alpha_n=0.7, alpha_p=0.7):
    """
    Concentration-dependent mobility (impurity scattering)
    
    μ_n = μ_n_min + (μ_n0 - μ_n_min) / (1 + (n/n_ref)^α_n)
    μ_p = μ_p_min + (μ_p0 - μ_p_min) / (1 + (p/p_ref)^α_p)
    
    Args:
        n: electron concentration (m^-3)
        p: hole concentration (m^-3)
        n_ref, p_ref: reference concentrations (m^-3)
        mu_n_min, mu_p_min: minimum mobilities (m²/V/s)
        alpha_n, alpha_p: scattering parameters
    
    Returns:
        (mu_n, mu_p): concentration-dependent mobilities (m²/V/s)
    """
    mu_n = mu_n_min + (mu_n0 - mu_n_min) / (1 + (n/n_ref)**alpha_n)
    mu_p = mu_p_min + (mu_p0 - mu_p_min) / (1 + (p/p_ref)**alpha_p)
    
    return mu_n, mu_p

def diffusion_coefficients(mu_n, mu_p, T=None):
    """
    Calculate diffusion coefficients using Einstein relation
    
    D = μ * k_B * T / q = μ * V_T
    
    Args:
        mu_n: electron mobility (m²/V/s)
        mu_p: hole mobility (m²/V/s)
        T: temperature (K), uses global T if None
    
    Returns:
        (D_n, D_p): electron and hole diffusion coefficients (m²/s)
    """
    if T is not None:
        V_T_local = 1.380649e-23 * T / q  # k_B * T / q
    else:
        V_T_local = V_T
    
    D_n = mu_n * V_T_local
    D_p = mu_p * V_T_local
    
    return D_n, D_p

def einstein_relation_check(mu_n, mu_p, D_n, D_p, tolerance=1e-10):
    """
    Verify Einstein relation: D = μ * V_T
    
    Args:
        mu_n, mu_p: mobilities (m²/V/s)
        D_n, D_p: diffusion coefficients (m²/s)
        tolerance: relative error tolerance
    
    Returns:
        True if Einstein relation is satisfied within tolerance
    """
    expected_D_n = mu_n * V_T
    expected_D_p = mu_p * V_T
    
    error_n = abs(D_n - expected_D_n) / expected_D_n if expected_D_n > 0 else abs(D_n)
    error_p = abs(D_p - expected_D_p) / expected_D_p if expected_D_p > 0 else abs(D_p)
    
    return error_n < tolerance and error_p < tolerance

def current_density_drift_diffusion(n, p, E_field, grad_n, grad_p, mu_n=None, mu_p=None):
    """
    Calculate current densities using drift-diffusion equations
    
    J_n = q * μ_n * n * E + q * D_n * grad_n
    J_p = q * μ_p * p * E - q * D_p * grad_p
    
    Args:
        n, p: carrier concentrations (m^-3)
        E_field: electric field vector (V/m)
        grad_n, grad_p: carrier density gradients (m^-4)
        mu_n, mu_p: mobilities (m²/V/s), use defaults if None
    
    Returns:
        (J_n, J_p): electron and hole current densities (A/m²)
    """
    if mu_n is None or mu_p is None:
        mu_n, mu_p = constant_mobility()
    
    D_n, D_p = diffusion_coefficients(mu_n, mu_p)
    
    # Drift components
    J_n_drift = q * mu_n * n * E_field
    J_p_drift = q * mu_p * p * E_field
    
    # Diffusion components  
    J_n_diff = q * D_n * grad_n
    J_p_diff = -q * D_p * grad_p  # Note negative sign for holes
    
    # Total current densities
    J_n = J_n_drift + J_n_diff
    J_p = J_p_drift + J_p_diff
    
    return J_n, J_p

def transport_test():
    """
    Test transport models for consistency
    """
    print("Testing transport models:")
    
    # Test Einstein relation
    mu_n, mu_p = constant_mobility()
    D_n, D_p = diffusion_coefficients(mu_n, mu_p)
    
    einstein_ok = einstein_relation_check(mu_n, mu_p, D_n, D_p)
    print(f"  Einstein relation: {'✓ PASS' if einstein_ok else '✗ FAIL'}")
    print(f"  μ_n = {mu_n:.3f} m²/V/s, D_n = {D_n*1e4:.3f} cm²/s")
    print(f"  μ_p = {mu_p:.3f} m²/V/s, D_p = {D_p*1e4:.3f} cm²/s")
    
    # Test field dependence
    E_fields = [0, 1e3, 1e4, 1e5, 1e6]  # V/m
    print(f"  Field-dependent mobility (electrons):")
    for E in E_fields:
        mu_n_field = field_dependent_mobility(E, mu_n0)
        print(f"    E = {E:.0e} V/m: μ_n = {mu_n_field:.4f} m²/V/s")
    
    return einstein_ok