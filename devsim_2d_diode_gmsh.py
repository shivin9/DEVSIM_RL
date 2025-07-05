#!/usr/bin/env python3
"""
2D Diode IV Characteristics using GMSH mesh
GMSH version of devsim_2d_diode_corrected.py for validation
"""

import numpy as np
import matplotlib.pyplot as plt

# DEVSIM imports
from devsim import (
    create_gmsh_mesh, add_gmsh_region, add_gmsh_contact, 
    finalize_mesh, create_device, set_parameter, solve, get_contact_current
)

from devsim.python_packages.simple_physics import (
    SetSiliconParameters, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName, 
    CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact
)

from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim import set_node_values, get_contact_list

def CreateGMSHMesh(device, region):
    """
    Create 2D mesh from GMSH file
    """
    print("Loading GMSH mesh...")
    
    # Load the GMSH mesh file
    create_gmsh_mesh(mesh="diode_gmsh", file="rectangular_diode_2d.msh")
    
    # Add single bulk region based on Physical Surface names in .geo file
    add_gmsh_region(mesh="diode_gmsh", gmsh_name="Bulk", region=region, material="Silicon")
    
    # Add contacts based on Physical Curve names in .geo file
    add_gmsh_contact(
        mesh="diode_gmsh", 
        gmsh_name="top_contact", 
        region=region,  # Both contacts connect to the same bulk region
        material="metal", 
        name="top"
    )
    add_gmsh_contact(
        mesh="diode_gmsh", 
        gmsh_name="bot_contact", 
        region=region,  # Both contacts connect to the same bulk region
        material="metal", 
        name="bot"
    )
    
    finalize_mesh(mesh="diode_gmsh")
    create_device(mesh="diode_gmsh", device=device)
    
    print("GMSH mesh loaded successfully")
    print(f"Device regions: P_region, N_region")
    print(f"Contacts: top (P+), bot (N+)")

def SetParameters(device, region):
    """
    Set parameters for 300 K for the bulk region
    """
    print("Setting silicon parameters...")
    SetSiliconParameters(device, region, 300)

def SetNetDoping(device, region):
    """
    Create P-N junction doping profile using step function
    Same as original: junction at x = 0.5e-5 (5 μm)
    """
    print("Creating doping profile...")
    
    # Junction at x = 0.5e-5 (5 μm) - same as original
    CreateNodeModel(device, region, "Acceptors", "1.0e18*step(0.5e-5-x)")  # P-side (left)
    CreateNodeModel(device, region, "Donors", "1.0e18*step(x-0.5e-5)")     # N-side (right)
    CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")

def InitialSolution(device, region):
    """
    Initialize potential solution for the bulk region
    """
    print("Initializing potential solution...")
    
    # Create Potential solution variable
    CreateSolution(device, region, "Potential")

    # Create potential-only physical models
    CreateSiliconPotentialOnly(device, region)

    # Set up the contacts
    for contact in get_contact_list(device=device):
        set_parameter(device=device, name=GetContactBiasName(contact), value=0.0)
        CreateSiliconPotentialOnlyContact(device, region, contact)

def DriftDiffusionInitialSolution(device, region):
    """
    Setup drift-diffusion equations for the bulk region
    """
    print("Setting up drift-diffusion...")
    
    # Create drift diffusion solution variables
    CreateSolution(device, region, "Electrons")
    CreateSolution(device, region, "Holes")

    # Create initial guess from dc only solution
    set_node_values(
        device=device, region=region, name="Electrons", init_from="IntrinsicElectrons"
    )
    set_node_values(
        device=device, region=region, name="Holes", init_from="IntrinsicHoles"
    )

    # Set up equations
    CreateSiliconDriftDiffusion(device, region)
    
    # Set up contacts
    for contact in get_contact_list(device=device):
        CreateSiliconDriftDiffusionAtContact(device, region, contact)

def simulate_gmsh_iv_characteristics(device):
    """
    Simulate IV characteristics for GMSH diode
    """
    print("\n=== GMSH IV Characteristics Simulation ===")
    
    # Same voltage sweep as original
    voltages = np.concatenate([
        np.linspace(-1.0, -0.1, 10),  # Reverse bias: -1V to -0.1V
        np.linspace(0.0, 0.5, 6)      # Forward bias: 0V to 0.5V
    ])
    currents_top = []
    currents_bot = []
    
    for v in voltages:
        print(f"\nApplying {v:.1f}V to top contact...")
        
        # Set bias voltage
        set_parameter(device=device, name=GetContactBiasName("top"), value=v)
        
        try:
            # Solve
            data = solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30, info=True)
            
            if data["converged"]:
                # Get currents from both contacts
                i_top_electron = get_contact_current(device=device, contact="top", equation="ElectronContinuityEquation")
                i_top_hole = get_contact_current(device=device, contact="top", equation="HoleContinuityEquation")
                i_top = i_top_electron + i_top_hole
                
                i_bot_electron = get_contact_current(device=device, contact="bot", equation="ElectronContinuityEquation")
                i_bot_hole = get_contact_current(device=device, contact="bot", equation="HoleContinuityEquation")
                i_bot = i_bot_electron + i_bot_hole
                
                currents_top.append(i_top)
                currents_bot.append(i_bot)
                
                print(f"  Top current: {i_top:.2e} A (e: {i_top_electron:.2e}, h: {i_top_hole:.2e})")
                print(f"  Bot current: {i_bot:.2e} A (e: {i_bot_electron:.2e}, h: {i_bot_hole:.2e})")
            else:
                print(f"  Failed to converge")
                currents_top.append(np.nan)
                currents_bot.append(np.nan)
            
        except Exception as e:
            print(f"  Solver error: {e}")
            currents_top.append(np.nan)
            currents_bot.append(np.nan)
    
    return voltages, currents_top, currents_bot

def plot_gmsh_iv_characteristics(voltages, currents_top, currents_bot):
    """
    Plot GMSH diode IV characteristics
    """
    plt.figure(figsize=(15, 10))
    
    # Convert to numpy arrays
    V = np.array(voltages)
    I_top = np.array(currents_top)
    I_bot = np.array(currents_bot)
    I_diode = I_top  # Use top current as primary
    
    # Plot 1: Linear scale IV curve
    plt.subplot(2, 3, 1)
    plt.plot(V, I_diode, 'bo-', linewidth=2, markersize=6, label='GMSH 2D Diode')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('GMSH 2D Diode IV Characteristics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Forward bias log scale
    plt.subplot(2, 3, 2)
    forward_mask = (V > 0) & ~np.isnan(I_diode)
    if np.any(forward_mask):
        V_fwd = V[forward_mask]
        I_fwd = np.abs(I_diode[forward_mask])
        I_fwd = np.maximum(I_fwd, 1e-15)
        plt.semilogy(V_fwd, I_fwd, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('Voltage (V)')
        plt.ylabel('|Current| (A)')
        plt.title('Forward Bias (Log Scale)')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Reverse bias log scale
    plt.subplot(2, 3, 3)
    reverse_mask = (V < 0) & ~np.isnan(I_diode)
    if np.any(reverse_mask):
        V_rev = np.abs(V[reverse_mask])
        I_rev = np.abs(I_diode[reverse_mask])
        I_rev = np.maximum(I_rev, 1e-15)
        plt.semilogy(V_rev, I_rev, 'go-', linewidth=2, markersize=6)
        plt.xlabel('|Voltage| (V)')
        plt.ylabel('|Current| (A)')
        plt.title('Reverse Bias (Log Scale)')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Current comparison
    plt.subplot(2, 3, 4)
    valid_mask = ~np.isnan(I_top) & ~np.isnan(I_bot)
    if np.any(valid_mask):
        plt.plot(V[valid_mask], I_top[valid_mask], 'b.-', label='Top contact')
        plt.plot(V[valid_mask], I_bot[valid_mask], 'r.-', label='Bot contact')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title('Contact Current Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Current conservation
    plt.subplot(2, 3, 5)
    if np.any(valid_mask):
        I_sum = I_top[valid_mask] + I_bot[valid_mask]
        plt.plot(V[valid_mask], np.abs(I_sum), 'mo-', linewidth=2, markersize=6)
        plt.xlabel('Voltage (V)')
        plt.ylabel('|I_top + I_bot| (A)')
        plt.title('Current Conservation Check')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    # Plot 6: Mesh info
    plt.subplot(2, 3, 6)
    plt.text(0.1, 0.8, 'GMSH Mesh Information:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.7, f'Mesh file: rectangular_diode_2d.msh', fontsize=10)
    plt.text(0.1, 0.6, f'Geometry: 10μm × 10μm rectangle', fontsize=10)
    plt.text(0.1, 0.5, f'P-N junction: x = 5μm', fontsize=10)
    plt.text(0.1, 0.4, f'Doping: 1×10¹⁸ cm⁻³', fontsize=10)
    plt.text(0.1, 0.3, f'Contacts: Line contacts', fontsize=10)
    plt.text(0.1, 0.2, f'Elements: ~856k nodes, ~1.7M elements', fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('gmsh_2d_diode_iv.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved GMSH IV plot: gmsh_2d_diode_iv.png")

def main():
    """
    Main function - GMSH version
    """
    print("2D Diode IV Characteristics - GMSH Implementation")
    print("=" * 60)
    
    device = "GMSHDevice"
    region = "MainRegion"
    
    try:
        # Create GMSH mesh and setup
        CreateGMSHMesh(device, region)
        SetParameters(device, region)
        SetNetDoping(device, region)
        InitialSolution(device, region)

        # Initial DC solution (Poisson only)
        print("\nSolving Poisson equation...")
        solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)

        # Setup drift-diffusion
        DriftDiffusionInitialSolution(device, region)
        
        # Drift diffusion equilibrium solution
        print("Solving drift-diffusion at equilibrium...")
        solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)

        # Run IV characteristics
        voltages, currents_top, currents_bot = simulate_gmsh_iv_characteristics(device)
        
        # Plot results
        plot_gmsh_iv_characteristics(voltages, currents_top, currents_bot)
        
        # Save data for comparison
        np.savetxt('gmsh_2d_diode_iv_data.csv', 
                   np.column_stack((voltages, currents_top, currents_bot)), 
                   delimiter=',', 
                   header='Voltage(V),Current_Top(A),Current_Bot(A)',
                   comments='')
        
        # Print summary
        print("\n=== GMSH Summary ===")
        print(f"Device: {device}")
        print(f"Voltage range: {min(voltages):.1f}V to {max(voltages):.1f}V")
        valid_currents = [c for c in currents_top if not np.isnan(c)]
        if valid_currents:
            print(f"Current range: {min(valid_currents):.2e}A to {max(valid_currents):.2e}A")
        print("Data saved to: gmsh_2d_diode_iv_data.csv")
        print("GMSH diode simulation completed successfully!")
        
    except Exception as e:
        print(f"GMSH simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()