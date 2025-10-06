#!/usr/bin/env python3
"""
2D Diode IV Characteristics with Custom Geometry
Fixed contact implementation based on DEVSIM examples
"""

import numpy as np
import matplotlib.pyplot as plt

# DEVSIM imports
from devsim import (
    create_2d_mesh, add_2d_mesh_line, add_2d_contact, add_2d_region,
    finalize_mesh, create_device, node_model, set_parameter,
    solve, get_contact_current, get_node_model_values
)

from devsim.python_packages.simple_physics import (
    SetSiliconParameters, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName
)

def create_2d_diode_with_contacts():
    """
    Create 2D diode following DEVSIM example pattern with proper contacts
    """
    print("Creating 2D diode with proper contacts...")
    
    device = "Diode2D"
    region = "Silicon"
    
    # Create mesh following DEVSIM diode_common.py pattern
    create_2d_mesh(mesh="dio2d")
    
    # X-direction (device length) - 10 μm total
    add_2d_mesh_line(mesh="dio2d", dir="x", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio2d", dir="x", pos=4.5e-6, ps=1e-8)   # Fine near junction
    add_2d_mesh_line(mesh="dio2d", dir="x", pos=5e-6, ps=1e-9)     # Junction at 5 μm
    add_2d_mesh_line(mesh="dio2d", dir="x", pos=5.5e-6, ps=1e-8)   # Fine near junction
    add_2d_mesh_line(mesh="dio2d", dir="x", pos=10e-6, ps=1e-6)
    
    # Y-direction (device width) - 5 μm total  
    add_2d_mesh_line(mesh="dio2d", dir="y", pos=0, ps=1e-6)
    add_2d_mesh_line(mesh="dio2d", dir="y", pos=5e-6, ps=1e-6)
    
    # Add extra mesh lines for contact boundaries (like diode_common.py)
    add_2d_mesh_line(mesh="dio2d", dir="x", pos=-1e-8, ps=1e-8)
    add_2d_mesh_line(mesh="dio2d", dir="x", pos=10.01e-6, ps=1e-8)
    
    # Create regions
    add_2d_region(mesh="dio2d", material="Si", region=region)
    # Add extra regions for contacts (like diode_common.py)
    add_2d_region(mesh="dio2d", material="Si", region="air1", xl=-1e-8, xh=0)
    add_2d_region(mesh="dio2d", material="Si", region="air2", xl=10e-6, xh=10.01e-6)
    
    # Create contacts exactly like diode_common.py
    # Top contact (P+ side) - line contact at high y
    add_2d_contact(
        mesh="dio2d",
        name="anode",
        material="metal", 
        region=region,
        yl=4e-6,      # Near top edge
        yh=5e-6,      # Top edge
        xl=0,         # Left boundary
        xh=0,         # Line contact (xl=xh)
        bloat=1e-10
    )
    
    # Bottom contact (N+ side) - line contact at high x
    add_2d_contact(
        mesh="dio2d",
        name="cathode",
        material="metal",
        region=region,
        xl=10e-6,     # Right boundary
        xh=10e-6,     # Line contact (xl=xh)
        bloat=1e-10
    )
    
    # Finalize and create device
    finalize_mesh(mesh="dio2d")
    create_device(mesh="dio2d", device=device)
    
    # Set silicon parameters
    SetSiliconParameters(device, region, 300)
    
    # Create doping profile
    create_2d_doping(device, region)
    
    # Initialize solution following diode_common pattern
    from devsim.python_packages.model_create import CreateSolution
    CreateSolution(device, region, "Potential")
    
    return device, region

def create_2d_doping(device, region):
    """
    Create P-N junction doping profile
    """
    # Junction at x = 5 μm
    junction_pos = 5e-6
    
    # Doping levels (m^-3)
    NA = 1e17 * 1e6  # P-side
    ND = 1e17 * 1e6  # N-side
    
    # Create step junction
    doping_expr = f"ifelse(x < {junction_pos}, {NA}, -{ND})"
    
    node_model(device=device, region=region, name="NetDoping", equation=doping_expr)
    
    print(f"Created 2D doping profile: P-side {NA/1e6:.1e} cm^-3, N-side {ND/1e6:.1e} cm^-3")

def setup_physics(device, region):
    """
    Setup drift-diffusion physics
    """
    print("Setting up physics...")
    
    # Import additional functions
    from devsim.python_packages.model_create import CreateSolution
    from devsim import set_node_values, get_contact_list
    
    # Create electron and hole concentration solution variables
    CreateSolution(device, region, "Electrons")
    CreateSolution(device, region, "Holes")
    
    # Initialize with intrinsic values from equilibrium solution
    set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
    set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
    
    # Create drift-diffusion equations  
    CreateSiliconDriftDiffusion(device, region)
    CreateSiliconDriftDiffusionAtContact(device, region, "anode")
    CreateSiliconDriftDiffusionAtContact(device, region, "cathode")

def simulate_iv_characteristics(device, region):
    """
    Simulate IV characteristics
    """
    print("Simulating IV characteristics...")
    
    # Initial Poisson solution
    solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)
    
    # Setup drift-diffusion physics  
    setup_physics(device, region)
    
    # Initial equilibrium solution with drift-diffusion
    solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
    
    # Voltage sweep
    voltages = np.linspace(-1.0, 0.8, 19)  # -1V to +0.8V
    currents = []
    
    for v in voltages:
        print(f"Solving for V = {v:.2f} V")
        
        # Set bias
        set_parameter(device=device, name=GetContactBiasName("anode"), value=v)
        
        # Solve
        try:
            solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
            
            # Get current
            current = get_contact_current(device=device, contact="anode")
            currents.append(current)
            
            print(f"  Current: {current:.2e} A")
            
        except Exception as e:
            print(f"  Failed to converge at {v:.2f} V: {e}")
            currents.append(np.nan)
    
    return voltages, currents

def plot_iv_characteristics(voltages, currents):
    """
    Plot IV characteristics
    """
    plt.figure(figsize=(12, 8))
    
    # Linear scale
    plt.subplot(2, 2, 1)
    plt.plot(voltages, currents, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('2D Diode IV Characteristics')
    plt.grid(True, alpha=0.3)
    
    # Log scale (forward bias)
    plt.subplot(2, 2, 2)
    forward_mask = np.array(voltages) > 0
    if np.any(forward_mask):
        v_fwd = np.array(voltages)[forward_mask]
        i_fwd = np.array(currents)[forward_mask]
        i_fwd_pos = np.maximum(i_fwd, 1e-12)  # Avoid log(0)
        plt.semilogy(v_fwd, i_fwd_pos, 'ro-', linewidth=2, markersize=6)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title('Forward Bias (Log Scale)')
        plt.grid(True, alpha=0.3)
    
    # Log scale (reverse bias)
    plt.subplot(2, 2, 3)
    reverse_mask = np.array(voltages) < 0
    if np.any(reverse_mask):
        v_rev = np.array(voltages)[reverse_mask]
        i_rev = np.array(currents)[reverse_mask]
        i_rev_abs = np.abs(i_rev)
        i_rev_abs = np.maximum(i_rev_abs, 1e-12)  # Avoid log(0)
        plt.semilogy(np.abs(v_rev), i_rev_abs, 'go-', linewidth=2, markersize=6)
        plt.xlabel('|Voltage| (V)')
        plt.ylabel('|Current| (A)')
        plt.title('Reverse Bias (Log Scale)')
        plt.grid(True, alpha=0.3)
    
    # Full range log-log
    plt.subplot(2, 2, 4)
    valid_mask = ~np.isnan(currents)
    if np.any(valid_mask):
        v_valid = np.array(voltages)[valid_mask]
        i_valid = np.array(currents)[valid_mask]
        i_abs = np.abs(i_valid)
        i_abs = np.maximum(i_abs, 1e-12)
        plt.loglog(np.abs(v_valid), i_abs, 'mo-', linewidth=2, markersize=6)
        plt.xlabel('|Voltage| (V)')
        plt.ylabel('|Current| (A)')
        plt.title('Full Range (Log-Log)')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('2d_diode_iv_characteristics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Saved IV plot: 2d_diode_iv_characteristics.png")

def main():
    """
    Main function to create and simulate 2D diode
    """
    print("2D Diode IV Characteristics Simulation")
    print("=" * 50)
    
    try:
        # Create device
        device, region = create_2d_diode_with_contacts()
        
        # Simulate IV characteristics
        voltages, currents = simulate_iv_characteristics(device, region)
        
        # Plot results
        plot_iv_characteristics(voltages, currents)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Device: {device}")
        print(f"Voltage range: {min(voltages):.1f} V to {max(voltages):.1f} V")
        print(f"Current range: {min(currents):.2e} A to {max(currents):.2e} A")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()