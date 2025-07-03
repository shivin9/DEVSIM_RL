"""
Physical constants and material parameters for semiconductor device simulation
"""

import numpy as np

# ====================================================================
# Universal Physical Constants
# ====================================================================
q = 1.60217663e-19  # Elementary charge (C)
epsilon_0 = 8.85418782e-12  # Vacuum permittivity (F/m)
k_B = 1.380649e-23  # Boltzmann constant (J/K)

# ====================================================================
# Material Properties (Silicon)
# ====================================================================
epsilon_si = 11.7  # Relative permittivity of Silicon
epsilon = epsilon_si * epsilon_0  # Permittivity of Silicon (F/m)

# Temperature-dependent parameters
T = 300  # Temperature (K)
V_T = k_B * T / q  # Thermal voltage (V), ~0.0259 V at 300K

# Intrinsic concentration
n_i_cm3 = 1.0e10  # Intrinsic concentration (cm^-3)
n_i_m3 = n_i_cm3 * 1e6  # Intrinsic concentration (m^-3)

# ====================================================================
# Doping Parameters
# ====================================================================
# Maximum doping concentrations
NA_max_cm3 = 5e16  # Acceptor concentration (cm^-3)
ND_max_cm3 = 5e16  # Donor concentration (cm^-3)
NA_max_m3 = NA_max_cm3 * 1e6  # Acceptor concentration (m^-3)
ND_max_m3 = ND_max_cm3 * 1e6  # Donor concentration (m^-3)

# ====================================================================
# Transport Parameters
# ====================================================================
# Mobility parameters (m²/V/s)
mu_n0 = 0.14  # Electron mobility
mu_p0 = 0.045  # Hole mobility

# Diffusion coefficients (Einstein relation: D = μkT/q)
D_n = mu_n0 * V_T  # Electron diffusion coefficient
D_p = mu_p0 * V_T  # Hole diffusion coefficient

# ====================================================================
# Recombination Parameters
# ====================================================================
# SRH Recombination parameters
tau_n0 = 1e-6  # Electron lifetime (s)
tau_p0 = 1e-6  # Hole lifetime (s)
E_trap = 0.0   # Trap energy level relative to intrinsic (eV)

# ====================================================================
# Derived Parameters
# ====================================================================
def built_in_potential():
    """Calculate built-in potential for p-n junction"""
    return V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)

def debye_length():
    """Calculate Debye length"""
    # For p-type side
    L_D_p = np.sqrt(epsilon * V_T / (q * NA_max_m3))
    # For n-type side  
    L_D_n = np.sqrt(epsilon * V_T / (q * ND_max_m3))
    return L_D_p, L_D_n

def print_constants():
    """Print summary of physical constants"""
    print("Physical Constants Summary:")
    print(f"  V_T = {V_T:.4f} V")
    print(f"  n_i = {n_i_m3:.2e} m^-3")
    print(f"  Built-in potential = {built_in_potential():.4f} V")
    L_D_p, L_D_n = debye_length()
    print(f"  Debye lengths: L_D_p = {L_D_p*1e6:.1f} μm, L_D_n = {L_D_n*1e6:.1f} μm")
    print(f"  Mobility: μ_n = {mu_n0:.3f} m²/V/s, μ_p = {mu_p0:.3f} m²/V/s")
    print(f"  Diffusion: D_n = {D_n*1e4:.3f} cm²/s, D_p = {D_p*1e4:.3f} cm²/s")