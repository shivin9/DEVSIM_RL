#!/usr/bin/env python3
"""
2D Diode IV Characteristics Analysis
Based on DEVSIM_examples/diode/diode_2d.py
Generates IV curve data and plots the characteristics
"""

import numpy as np
import matplotlib.pyplot as plt
from devsim import set_parameter, solve, get_contact_current
from devsim.python_packages.simple_physics import GetContactBiasName, PrintCurrents
import sys
import os

# Add the diode directory to path to import diode_common
sys.path.append('/home/shivin/Research/DEVSIM_RL/DEVSIM_examples/diode')
import diode_common

def run_iv_simulation():
    """Run the 2D diode simulation and collect IV data"""
    device = "MyDevice"
    region = "MyRegion"
    
    # Setup the device
    diode_common.Create2DMesh(device, region)
    diode_common.SetParameters(device=device, region=region)
    diode_common.SetNetDoping(device=device, region=region)
    diode_common.InitialSolution(device, region)
    
    # Initial DC solution
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)
    
    diode_common.DriftDiffusionInitialSolution(device, region)
    
    # Drift diffusion simulation at equilibrium
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
    
    # Arrays to store IV data
    voltages = []
    currents = []
    
    # Ramp the bias from 0 to 0.5 V
    v = 0.0
    while v < 0.51:
        set_parameter(device=device, name=GetContactBiasName("top"), value=v)
        solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
        
        # Get current from top contact (total = electron + hole current)
        electron_current = get_contact_current(device=device, contact="top", equation="ElectronContinuityEquation")
        hole_current = get_contact_current(device=device, contact="top", equation="HoleContinuityEquation")
        current = electron_current + hole_current
        
        voltages.append(v)
        currents.append(current)
        
        print(f"Voltage: {v:.2f} V, Current: {current:.2e} A")
        v += 0.1
    
    # Try higher voltages
    test_voltages = [0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2, 2.5, 3, 5]
    for val in test_voltages:
        try:
            set_parameter(device=device, name=GetContactBiasName("top"), value=val)
            data = solve(
                type="dc",
                absolute_error=1e10,
                relative_error=1e-10,
                maximum_iterations=30,
                info=True,
            )
            if data["converged"]:
                electron_current = get_contact_current(device=device, contact="top", equation="ElectronContinuityEquation")
                hole_current = get_contact_current(device=device, contact="top", equation="HoleContinuityEquation")
                current = electron_current + hole_current
                voltages.append(val)
                currents.append(current)
                print(f"Voltage: {val:.2f} V, Current: {current:.2e} A")
            else:
                print(f"Did not converge at {val} V")
                break
        except Exception as e:
            print(f"Error at {val} V: {e}")
            break
    
    return np.array(voltages), np.array(currents)

def plot_iv_characteristics(voltages, currents):
    """Plot IV characteristics"""
    plt.figure(figsize=(10, 6))
    
    # Linear plot
    plt.subplot(1, 2, 1)
    plt.plot(voltages, currents*1000, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (mA)')
    plt.title('2D Diode IV Characteristics (Linear)')
    plt.grid(True, alpha=0.3)
    
    # Semi-log plot
    plt.subplot(1, 2, 2)
    plt.semilogy(voltages, np.abs(currents)*1000, 'r-o', linewidth=2, markersize=6)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (mA)')
    plt.title('2D Diode IV Characteristics (Log)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diode_2d_iv_characteristics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print some statistics
    print(f"\nIV Characteristics Summary:")
    print(f"Voltage range: {voltages.min():.2f} to {voltages.max():.2f} V")
    print(f"Current range: {currents.min():.2e} to {currents.max():.2e} A")
    print(f"Forward voltage at 1mA: {np.interp(1e-3, currents, voltages):.3f} V")

def main():
    """Main function to run simulation and generate plots"""
    print("Running 2D Diode IV Characteristics Simulation...")
    print("=" * 50)
    
    # Run simulation
    voltages, currents = run_iv_simulation()
    
    # Save data to file
    np.savetxt('diode_2d_iv_data.csv', 
               np.column_stack((voltages, currents)), 
               delimiter=',', 
               header='Voltage(V),Current(A)',
               comments='')
    
    # Plot results
    plot_iv_characteristics(voltages, currents)
    
    print("\nSimulation complete!")
    print("Data saved to: diode_2d_iv_data.csv")
    print("Plot saved to: diode_2d_iv_characteristics.png")

if __name__ == "__main__":
    main()