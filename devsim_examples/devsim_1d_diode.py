#!/usr/bin/env python3
"""
1D Silicon Diode Simulation using DEVSIM Library

This code implements a proper 1D p-n junction diode simulation using the DEVSIM
semiconductor device simulator. It follows the standard DEVSIM approach without
artificial constraints or clipping.

Based on DEVSIM examples: diode_1d.py and diode_common.py
"""

import numpy as np
import matplotlib.pyplot as plt

# DEVSIM imports
from devsim import (
    add_1d_contact,
    add_1d_mesh_line, 
    add_1d_region,
    create_1d_mesh,
    create_device,
    finalize_mesh,
    get_contact_list,
    get_contact_current,
    get_node_model_values,
    get_edge_model_values,
    print_node_values,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
    edge_average_model,
)

from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim.python_packages.simple_physics import (
    GetContactBiasName,
    SetSiliconParameters,
    CreateSiliconPotentialOnly,
    CreateSiliconPotentialOnlyContact,
    CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact,
    PrintCurrents,
)

# Device and region names
device = "SiliconDiode"
region = "Silicon"

def create_1d_diode_mesh():
    """
    Create 1D mesh for silicon diode
    
    Mesh structure:
    - 0 to 0.5μm: P-type region (acceptor doped)
    - 0.5μm: Junction location  
    - 0.5μm to 1.0μm: N-type region (donor doped)
    
    Fine mesh around junction for accurate solution
    """
    print("Creating 1D mesh...")
    
    # Create 1D mesh
    create_1d_mesh(mesh="diode")
    
    # Define mesh points with fine spacing around junction
    add_1d_mesh_line(mesh="diode", pos=0.0, ps=1e-7, tag="anode")     # P-contact  
    add_1d_mesh_line(mesh="diode", pos=0.4e-5, ps=1e-8, tag="p_bulk") # P-region
    add_1d_mesh_line(mesh="diode", pos=0.5e-5, ps=1e-9, tag="junction") # Junction
    add_1d_mesh_line(mesh="diode", pos=0.6e-5, ps=1e-8, tag="n_bulk") # N-region  
    add_1d_mesh_line(mesh="diode", pos=1.0e-5, ps=1e-7, tag="cathode") # N-contact
    
    # Create contacts (metal contacts at ends)
    add_1d_contact(mesh="diode", name="anode", tag="anode", material="metal")
    add_1d_contact(mesh="diode", name="cathode", tag="cathode", material="metal")
    
    # Create silicon region
    add_1d_region(mesh="diode", material="Si", region=region, tag1="anode", tag2="cathode")
    
    # Finalize mesh and create device
    finalize_mesh(mesh="diode")
    create_device(mesh="diode", device=device)
    
    print("Mesh created successfully")

def set_silicon_parameters():
    """
    Set physical parameters for silicon at 300K
    Uses DEVSIM built-in silicon parameter library
    """
    print("Setting silicon parameters...")
    
    # Use DEVSIM built-in silicon parameters for 300K
    SetSiliconParameters(device, region, 300)
    
    # Set carrier lifetimes for SRH recombination  
    set_parameter(device=device, region=region, name="taun", value=1e-6)  # Electron lifetime (s)
    set_parameter(device=device, region=region, name="taup", value=1e-6)  # Hole lifetime (s)
    
    print("Silicon parameters set")

def create_doping_profile():
    """
    Create realistic step-junction doping profile
    
    Doping levels:
    - P-region (x < 0.5μm): NA = 1e17 cm^-3 acceptors
    - N-region (x > 0.5μm): ND = 1e17 cm^-3 donors
    """
    print("Creating doping profile...")
    
    # Junction location
    junction_pos = 0.5e-5  # 0.5 μm
    
    # Create acceptor and donor doping profiles using step functions
    # step(a) = 1 if a > 0, else 0
    CreateNodeModel(device, region, "Acceptors", f"1.0e17*step({junction_pos}-x)")
    CreateNodeModel(device, region, "Donors", f"1.0e17*step(x-{junction_pos})")
    
    # Net doping = Donors - Acceptors
    CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
    
    # Print doping profile for verification
    print_node_values(device=device, region=region, name="NetDoping")
    
    print("Doping profile created")

def solve_equilibrium():
    """
    Solve for equilibrium (zero bias) condition
    
    This establishes the built-in potential and carrier distributions
    at thermal equilibrium (no applied voltage).
    """
    print("\n=== Solving Equilibrium ===")
    
    # Create potential solution variable
    CreateSolution(device, region, "Potential")
    
    # Create potential-only physical models (Poisson equation only)
    CreateSiliconPotentialOnly(device, region)
    
    # Set up contacts with zero bias
    for contact_name in get_contact_list(device=device):
        print(f"Setting up contact: {contact_name}")
        set_parameter(device=device, name=GetContactBiasName(contact_name), value=0.0)
        CreateSiliconPotentialOnlyContact(device, region, contact_name)
    
    # Solve Poisson equation for equilibrium
    print("Solving Poisson equation...")
    solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
    
    print("Equilibrium solution complete")

def setup_drift_diffusion():
    """
    Set up drift-diffusion equations for carrier transport
    
    This adds electron and hole continuity equations to the Poisson equation,
    forming the complete drift-diffusion system.
    """
    print("\n=== Setting up Drift-Diffusion ===")
    
    # Create electron and hole concentration solution variables
    CreateSolution(device, region, "Electrons")
    CreateSolution(device, region, "Holes")
    
    # Initialize with intrinsic values from equilibrium solution
    set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
    
    # Create full drift-diffusion physics models
    CreateSiliconDriftDiffusion(device, region)
    
    # Set up drift-diffusion at contacts
    for contact_name in get_contact_list(device=device):
        CreateSiliconDriftDiffusionAtContact(device, region, contact_name)
    
    # Solve drift-diffusion at equilibrium
    print("Solving drift-diffusion at equilibrium...")
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
    
    print("Drift-diffusion setup complete")

def voltage_sweep():
    """
    Perform forward bias voltage sweep
    
    Sweeps from 0V to 0.8V in 0.1V steps and records I-V characteristics
    """
    print("\n=== Voltage Sweep ===")
    
    voltages = []
    currents_anode = []
    currents_cathode = []
    
    # Voltage sweep: -1.0V to +0.8V in 0.1V steps
    voltage = -1.0  # Start from reverse bias
    voltage_step = 0.1
    max_voltage = 0.8
    
    while voltage <= max_voltage + 1e-6:  # Small tolerance for floating point
        print(f"\n--- Applying {voltage:.1f}V ---")
        
        # Set anode bias (cathode remains at 0V)
        set_parameter(device=device, name=GetContactBiasName("anode"), value=voltage)
        
        # Solve at this bias point
        try:
            solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
            
            # Get and print currents
            PrintCurrents(device, "anode")
            PrintCurrents(device, "cathode")
            
            # Store results
            voltages.append(voltage)
            
            # Get contact currents (in Amperes) 
            # Total current = electron current + hole current at contact
            I_anode_e = get_contact_current(device=device, contact="anode", equation="ElectronContinuityEquation")
            I_anode_h = get_contact_current(device=device, contact="anode", equation="HoleContinuityEquation")
            I_anode = I_anode_e + I_anode_h
            
            I_cathode_e = get_contact_current(device=device, contact="cathode", equation="ElectronContinuityEquation") 
            I_cathode_h = get_contact_current(device=device, contact="cathode", equation="HoleContinuityEquation")
            I_cathode = I_cathode_e + I_cathode_h
            
            currents_anode.append(abs(I_anode))
            currents_cathode.append(abs(I_cathode))
            
            print(f"Voltage: {voltage:.1f}V, Current: {abs(I_anode):.2e}A")
            
        except Exception as e:
            print(f"Solution failed at {voltage:.1f}V: {e}")
            break
            
        voltage += voltage_step
    
    return voltages, currents_anode, currents_cathode

def plot_results(voltages, currents):
    """
    Plot I-V characteristics and device profiles
    """
    print("\n=== Plotting Results ===")
    
    # Plot I-V curve
    plt.figure(figsize=(15, 10))
    
    # I-V characteristics - linear scale
    plt.subplot(2, 3, 1)
    plt.plot(voltages, currents, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('Diode I-V Characteristics (Linear)')
    plt.grid(True, alpha=0.3)
    
    # I-V characteristics - log scale for wide dynamic range
    plt.subplot(2, 3, 2)
    plt.semilogy(voltages, np.abs(currents), 'ro-', linewidth=2, markersize=6)
    plt.xlabel('Voltage (V)')
    plt.ylabel('|Current| (A)')
    plt.title('Diode I-V Characteristics (Log Scale)')
    plt.grid(True, alpha=0.3)
    
    # Get spatial profiles at final bias
    x = get_node_model_values(device=device, region=region, name="x")
    potential = get_node_model_values(device=device, region=region, name="Potential")
    electrons = get_node_model_values(device=device, region=region, name="Electrons")
    holes = get_node_model_values(device=device, region=region, name="Holes")
    net_doping = get_node_model_values(device=device, region=region, name="NetDoping")
    
    # Convert x from cm to μm
    x_um = np.array(x) * 1e4
    
    # Plot potential profile
    plt.subplot(2, 3, 3)
    plt.plot(x_um, potential, 'r-', linewidth=2)
    plt.xlabel('Position (μm)')
    plt.ylabel('Potential (V)')
    plt.title('Electrostatic Potential')
    plt.grid(True, alpha=0.3)
    
    # Plot carrier concentrations
    plt.subplot(2, 3, 4)
    plt.semilogy(x_um, electrons, 'b-', linewidth=2, label='Electrons')
    plt.semilogy(x_um, holes, 'r-', linewidth=2, label='Holes')
    plt.semilogy(x_um, np.abs(net_doping), 'k--', linewidth=2, label='|Net Doping|')
    plt.xlabel('Position (μm)')
    plt.ylabel('Concentration (cm⁻³)')
    plt.title('Carrier Concentrations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot current density profiles
    plt.subplot(2, 3, 5)
    # Calculate edge positions for current density
    edge_average_model(device=device, region=region, node_model="x", edge_model="x_edge")
    x_edge = get_edge_model_values(device=device, region=region, name="x_edge")
    J_n = get_edge_model_values(device=device, region=region, name="ElectronCurrent")
    J_p = get_edge_model_values(device=device, region=region, name="HoleCurrent")
    
    x_edge_um = np.array(x_edge) * 1e4
    plt.plot(x_edge_um, J_n, 'b-', linewidth=2, label='Electron Current')
    plt.plot(x_edge_um, J_p, 'r-', linewidth=2, label='Hole Current')
    plt.plot(x_edge_um, np.array(J_n) + np.array(J_p), 'k-', linewidth=2, label='Total Current')
    plt.xlabel('Position (μm)')
    plt.ylabel('Current Density (A/cm²)')
    plt.title('Current Density Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add reverse/forward bias analysis
    plt.subplot(2, 3, 6)
    forward_mask = np.array(voltages) >= 0
    reverse_mask = np.array(voltages) < 0
    
    if np.any(forward_mask):
        plt.semilogy(np.array(voltages)[forward_mask], np.abs(np.array(currents))[forward_mask], 
                    'go-', linewidth=2, markersize=6, label='Forward Bias')
    if np.any(reverse_mask):
        plt.semilogy(np.abs(np.array(voltages))[reverse_mask], np.abs(np.array(currents))[reverse_mask], 
                    'ro-', linewidth=2, markersize=6, label='Reverse Bias')
    
    plt.xlabel('|Voltage| (V)')
    plt.ylabel('|Current| (A)')
    plt.title('Forward vs Reverse Bias')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('devsim_1d_diode_full_iv.png', dpi=300, bbox_inches='tight')
    print("Full I-V results saved to devsim_1d_diode_full_iv.png")
    
    # Show theoretical comparison
    print(f"\nFinal voltage: {voltages[-1]:.1f}V")
    print(f"Final current: {currents[-1]:.2e}A")
    
    # Analysis of forward and reverse characteristics
    forward_currents = [c for v, c in zip(voltages, currents) if v >= 0]
    forward_voltages = [v for v in voltages if v >= 0]
    reverse_currents = [c for v, c in zip(voltages, currents) if v < 0]
    reverse_voltages = [v for v in voltages if v < 0]
    
    if forward_currents:
        print(f"Forward voltage at ~1mA: {np.interp(1e-3, forward_currents, forward_voltages):.2f}V")
    
    if reverse_currents:
        print(f"Reverse current at -1V: {np.abs(reverse_currents[0]):.2e}A")
        print(f"Rectification ratio (I_forward/I_reverse at ±0.5V): {np.abs(forward_currents[5]/reverse_currents[-6]):.1e}" if len(forward_currents) > 5 and len(reverse_currents) > 6 else "N/A")

def main():
    """
    Main simulation function
    """
    print("DEVSIM 1D Silicon Diode Simulation")
    print("="*50)
    
    # Create mesh and set up device
    create_1d_diode_mesh()
    set_silicon_parameters()
    create_doping_profile()
    
    # Solve equilibrium condition
    solve_equilibrium()
    
    # Set up and solve drift-diffusion
    setup_drift_diffusion()
    
    # Perform voltage sweep
    voltages, currents_anode, currents_cathode = voltage_sweep()
    
    # Plot and analyze results
    if voltages:
        plot_results(voltages, currents_anode)
        
        # Save device data
        write_devices(file="devsim_1d_diode.dat", type="tecplot")
        print("Device data saved to devsim_1d_diode.dat")
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()