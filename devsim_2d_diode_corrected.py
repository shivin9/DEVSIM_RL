#!/usr/bin/env python3
"""
2D Diode IV Characteristics - Corrected Implementation
Following the exact pattern from DEVSIM_examples/diode/diode_common.py
"""

import numpy as np
import matplotlib.pyplot as plt

# DEVSIM imports
from devsim import (
    create_2d_mesh, add_2d_mesh_line, add_2d_contact, add_2d_region,
    finalize_mesh, create_device, set_parameter, solve, get_contact_current
)

from devsim.python_packages.simple_physics import (
    SetSiliconParameters, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName, PrintCurrents
)

from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim import set_node_values, get_contact_list

def Create2DMesh(device, region):
    """
    Create 2D mesh exactly like diode_common.py but with custom dimensions
    """
    print("Creating 2D mesh...")
    
    create_2d_mesh(mesh="dio")
    
    # X-direction mesh lines (device length: 0 to 1e-5 = 10 μm)
    add_2d_mesh_line(mesh="dio", dir="x", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="x", pos=0.5e-5, ps=1e-8)  # Junction at 5 μm
    add_2d_mesh_line(mesh="dio", dir="x", pos=1e-5, ps=1e-6)
    
    # Y-direction mesh lines (device width: 0 to 1e-5 = 10 μm)
    add_2d_mesh_line(mesh="dio", dir="y", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio", dir="y", pos=1e-5, ps=1e-6)

    # Add extra mesh lines for contact regions (exactly like diode_common.py)
    add_2d_mesh_line(mesh="dio", dir="x", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="dio", dir="x", pos=1.001e-5, ps=1e-8)

    # Create regions (exactly like diode_common.py)
    add_2d_region(mesh="dio", material="Si", region=region)
    add_2d_region(mesh="dio", material="Si", region="air1", xl=-1e-8, xh=0)
    add_2d_region(mesh="dio", material="Si", region="air2", xl=1.0e-5, xh=1.001e-5)

    # Create contacts exactly like diode_common.py
    # "top" contact (P+ side) - line contact at high y
    add_2d_contact(
        mesh="dio",
        name="top",
        material="metal",
        region=region,
        yl=0.8e-5,    # Near top edge
        yh=1e-5,      # Top edge
        xl=0,         # Left boundary
        xh=0,         # Line contact (xl=xh)
        bloat=1e-10,
    )
    
    # "bot" contact (N+ side) - line contact at high x  
    add_2d_contact(
        mesh="dio",
        name="bot",
        material="metal",
        region=region,
        xl=1e-5,      # Right boundary
        xh=1e-5,      # Line contact (xl=xh)
        bloat=1e-10,
    )

    finalize_mesh(mesh="dio")
    create_device(mesh="dio", device=device)
    
    print("2D mesh created successfully")

def SetParameters(device, region):
    """
    Set parameters for 300 K (exactly like diode_common.py)
    """
    print("Setting silicon parameters...")
    SetSiliconParameters(device, region, 300)

def SetNetDoping(device, region):
    """
    Create P-N junction doping profile (exactly like diode_common.py)
    """
    print("Creating doping profile...")
    
    # Junction at x = 0.5e-5 (5 μm)
    CreateNodeModel(device, region, "Acceptors", "1.0e18*step(0.5e-5-x)")  # P-side (left)
    CreateNodeModel(device, region, "Donors", "1.0e18*step(x-0.5e-5)")     # N-side (right)
    CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")

def InitialSolution(device, region, circuit_contacts=None):
    """
    Initialize potential solution (exactly like diode_common.py)
    """
    print("Initializing potential solution...")
    
    # Create Potential solution variable
    CreateSolution(device, region, "Potential")

    # Create potential-only physical models
    from devsim.python_packages.simple_physics import CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact

    CreateSiliconPotentialOnly(device, region)

    # Set up the contacts
    for i in get_contact_list(device=device):
        if circuit_contacts and i in circuit_contacts:
            CreateSiliconPotentialOnlyContact(device, region, i, True)
        else:
            set_parameter(device=device, name=GetContactBiasName(i), value=0.0)
            CreateSiliconPotentialOnlyContact(device, region, i)

def DriftDiffusionInitialSolution(device, region, circuit_contacts=None):
    """
    Setup drift-diffusion equations (exactly like diode_common.py)
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
    for i in get_contact_list(device=device):
        if circuit_contacts and i in circuit_contacts:
            CreateSiliconDriftDiffusionAtContact(device, region, i, True)
        else:
            CreateSiliconDriftDiffusionAtContact(device, region, i)

def simulate_2d_iv_characteristics(device, region):
    """
    Simulate IV characteristics for 2D diode
    """
    print("\n=== IV Characteristics Simulation ===")
    
    # Voltage sweep from -1V to +0.5V (more conservative like original example)
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
                # Get currents only if converged (need to specify equation names)
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

def plot_2d_iv_characteristics(voltages, currents_top, currents_bot):
    """
    Plot 2D diode IV characteristics
    """
    plt.figure(figsize=(15, 10))
    
    # Convert to numpy arrays for easier handling
    V = np.array(voltages)
    I_top = np.array(currents_top)
    I_bot = np.array(currents_bot)
    
    # Use top current as primary (it should be negative of bot current)
    I_diode = I_top
    
    # Plot 1: Linear scale IV curve
    plt.subplot(2, 3, 1)
    plt.plot(V, I_diode, 'bo-', linewidth=2, markersize=6, label='2D Diode')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('2D Diode IV Characteristics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 2: Forward bias log scale
    plt.subplot(2, 3, 2)
    forward_mask = (V > 0) & ~np.isnan(I_diode)
    if np.any(forward_mask):
        V_fwd = V[forward_mask]
        I_fwd = np.abs(I_diode[forward_mask])
        I_fwd = np.maximum(I_fwd, 1e-15)  # Avoid log(0)
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
        I_rev = np.maximum(I_rev, 1e-15)  # Avoid log(0)
        plt.semilogy(V_rev, I_rev, 'go-', linewidth=2, markersize=6)
        plt.xlabel('|Voltage| (V)')
        plt.ylabel('|Current| (A)')
        plt.title('Reverse Bias (Log Scale)')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Current comparison (top vs bot)
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
    
    # Plot 5: Current conservation check
    plt.subplot(2, 3, 5)
    if np.any(valid_mask):
        I_sum = I_top[valid_mask] + I_bot[valid_mask]
        plt.plot(V[valid_mask], np.abs(I_sum), 'mo-', linewidth=2, markersize=6)
        plt.xlabel('Voltage (V)')
        plt.ylabel('|I_top + I_bot| (A)')
        plt.title('Current Conservation Check')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    # Plot 6: Full range log-log
    plt.subplot(2, 3, 6)
    valid_mask = ~np.isnan(I_diode) & (V != 0)
    if np.any(valid_mask):
        V_valid = np.abs(V[valid_mask])
        I_valid = np.abs(I_diode[valid_mask])
        I_valid = np.maximum(I_valid, 1e-15)
        plt.loglog(V_valid, I_valid, 'mo-', linewidth=2, markersize=6)
        plt.xlabel('|Voltage| (V)')
        plt.ylabel('|Current| (A)')
        plt.title('Full Range (Log-Log)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2d_diode_iv_corrected.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved IV plot: 2d_diode_iv_corrected.png")

def main():
    """
    Main function - follows exact pattern from diode_2d.py
    """
    print("2D Diode IV Characteristics - Corrected Implementation")
    print("=" * 60)
    
    device = "MyDevice"
    region = "MyRegion"
    
    try:
        # Follow exact sequence from diode_2d.py
        Create2DMesh(device, region)
        SetParameters(device=device, region=region)
        SetNetDoping(device=device, region=region)
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
        voltages, currents_top, currents_bot = simulate_2d_iv_characteristics(device, region)
        
        # Plot results
        plot_2d_iv_characteristics(voltages, currents_top, currents_bot)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Device: {device}")
        print(f"Voltage range: {min(voltages):.1f}V to {max(voltages):.1f}V")
        valid_currents = [c for c in currents_top if not np.isnan(c)]
        if valid_currents:
            print(f"Current range: {min(valid_currents):.2e}A to {max(valid_currents):.2e}A")
        print("2D diode simulation completed successfully!")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()