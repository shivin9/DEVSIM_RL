#!/usr/bin/env python3
"""
Quick test of current scaling in the realistic diode
"""
import numpy as np

# Physical constants
q = 1.60217663e-19  # Elementary charge (C)
epsilon_0 = 8.85418782e-12
epsilon_si = 11.7
epsilon = epsilon_si * epsilon_0
k_B = 1.380649e-23
T = 300
V_T = k_B * T / q

# Device parameters
total_length_um = 10.0
width_um = 5.0
thickness_um = 1.0

total_length_m = total_length_um * 1e-6
width_m = width_um * 1e-6
thickness_m = thickness_um * 1e-6

# Doping
NA_junction_cm3 = 1e16
ND_junction_cm3 = 1e16
n_i_cm3 = 1.0e10

NA_junction_m3 = NA_junction_cm3 * 1e6
ND_junction_m3 = ND_junction_cm3 * 1e6
n_i_m3 = n_i_cm3 * 1e6

# Material parameters
mu_n = 0.14  # m²/V/s
mu_p = 0.045  # m²/V/s

print("Realistic Diode Current Estimation")
print("="*50)
print(f"Device dimensions: {total_length_um} × {width_um} × {thickness_um} μm")
print(f"Device area: {width_m * thickness_m * 1e12:.1f} μm²")
print(f"Junction doping: {NA_junction_cm3:.1e} cm⁻³")

# Estimate typical current for forward bias
V_applied = 0.5  # V
print(f"\nForward bias: {V_applied} V")

# Estimate carrier concentrations (simplified)
# In forward bias, minority carriers are injected
bias_factor = np.exp(V_applied / V_T)
print(f"Bias factor exp(V/VT): {bias_factor:.2e}")

# Estimate current using simplified drift model
# Typical conductivity estimation
avg_n = ND_junction_m3 * bias_factor  # Enhanced by bias
avg_p = NA_junction_m3  # Roughly unchanged
avg_conductivity = q * (avg_n * mu_n + avg_p * mu_p)

print(f"Avg electron density: {avg_n:.2e} m⁻³")
print(f"Avg conductivity: {avg_conductivity:.2e} S/m")

# Current calculation
V_drop = V_applied  # Approximate
device_area = width_m * thickness_m
current_density = avg_conductivity * (V_drop / total_length_m)
total_current = current_density * device_area

print(f"Current density: {current_density:.2e} A/m²")
print(f"Total current: {total_current:.2e} A = {total_current*1e6:.2f} μA")

# For comparison, typical silicon diode at 0.5V forward bias
print(f"\nTypical silicon diode (similar size) at 0.5V: ~1-100 μA")
print(f"Our calculation: {total_current*1e6:.2f} μA")
if total_current*1e6 > 1000:
    print("⚠️  Still too high - need to reduce carrier concentrations")
elif total_current*1e6 < 0.01:
    print("⚠️  Too low - might need higher doping or better model")
else:
    print("✅ Current level looks reasonable")