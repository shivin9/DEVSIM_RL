"""
Recombination models for semiconductor device simulation

Contains Shockley-Read-Hall (SRH) recombination and other 
generation-recombination mechanisms.
"""

import numpy as np
from .constants import n_i_m3, tau_n0, tau_p0

def srh_recombination_rate(n, p, n_i=None, tau_n=None, tau_p=None):
    """
    Shockley-Read-Hall recombination rate
    
    R = (np - ni²) / (tau_p*(n + ni) + tau_n*(p + ni))
    
    Args:
        n: electron concentration (m^-3)
        p: hole concentration (m^-3) 
        n_i: intrinsic concentration (m^-3), defaults to global n_i_m3
        tau_n: electron lifetime (s), defaults to global tau_n0
        tau_p: hole lifetime (s), defaults to global tau_p0
    
    Returns:
        Recombination rate (m^-3/s)
        Positive = recombination, Negative = generation
    """
    if n_i is None:
        n_i = n_i_m3
    if tau_n is None:
        tau_n = tau_n0
    if tau_p is None:
        tau_p = tau_p0
    
    # Add small regularization to prevent division by zero
    epsilon = 1e-30
    
    numerator = n * p - n_i**2
    denominator = tau_p * (n + n_i) + tau_n * (p + n_i) + epsilon
    
    return numerator / denominator

def auger_recombination_rate(n, p, n_i=None, C_n=1e-43, C_p=1e-43):
    """
    Auger recombination rate (for high injection conditions)
    
    R_auger = (C_n*n + C_p*p) * (np - ni²)
    
    Args:
        n: electron concentration (m^-3)
        p: hole concentration (m^-3)
        n_i: intrinsic concentration (m^-3)
        C_n: electron Auger coefficient (m^6/s)
        C_p: hole Auger coefficient (m^6/s)
    
    Returns:
        Auger recombination rate (m^-3/s)
    """
    if n_i is None:
        n_i = n_i_m3
    
    return (C_n * n + C_p * p) * (n * p - n_i**2)

def radiative_recombination_rate(n, p, n_i=None, B=1e-16):
    """
    Radiative (band-to-band) recombination rate
    
    R_rad = B * (np - ni²)
    
    Args:
        n: electron concentration (m^-3)
        p: hole concentration (m^-3)
        n_i: intrinsic concentration (m^-3)
        B: radiative recombination coefficient (m^3/s)
    
    Returns:
        Radiative recombination rate (m^-3/s)
    """
    if n_i is None:
        n_i = n_i_m3
    
    return B * (n * p - n_i**2)

def total_recombination_rate(n, p, n_i=None, include_auger=False, include_radiative=False):
    """
    Total recombination rate including multiple mechanisms
    
    R_total = R_SRH + R_auger + R_radiative
    
    Args:
        n: electron concentration (m^-3)
        p: hole concentration (m^-3)
        n_i: intrinsic concentration (m^-3)
        include_auger: whether to include Auger recombination
        include_radiative: whether to include radiative recombination
    
    Returns:
        Total recombination rate (m^-3/s)
    """
    R_total = srh_recombination_rate(n, p, n_i)
    
    if include_auger:
        R_total += auger_recombination_rate(n, p, n_i)
    
    if include_radiative:
        R_total += radiative_recombination_rate(n, p, n_i)
    
    return R_total

def equilibrium_test():
    """
    Test recombination models at equilibrium
    Should return very small recombination rates
    """
    print("Testing recombination models at equilibrium:")
    
    # Equilibrium conditions
    n_eq = 1e16  # m^-3
    p_eq = n_i_m3**2 / n_eq  # Maintain np = ni²
    
    R_srh = srh_recombination_rate(n_eq, p_eq)
    R_auger = auger_recombination_rate(n_eq, p_eq)
    R_rad = radiative_recombination_rate(n_eq, p_eq)
    
    print(f"  n = {n_eq:.2e} m^-3, p = {p_eq:.2e} m^-3")
    print(f"  R_SRH = {R_srh:.2e} m^-3/s")
    print(f"  R_Auger = {R_auger:.2e} m^-3/s") 
    print(f"  R_radiative = {R_rad:.2e} m^-3/s")
    
    # Should all be very small (ideally zero)
    total_error = abs(R_srh) + abs(R_auger) + abs(R_rad)
    print(f"  Total equilibrium error: {total_error:.2e} m^-3/s")
    
    return total_error < 1e10  # Reasonable tolerance