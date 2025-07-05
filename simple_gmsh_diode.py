#!/usr/bin/env python3
"""
Simple 2D GMSH Diode Test
Based on working DEVSIM examples
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

def main():
    """
    Simple GMSH diode test - following exact pattern from working examples
    """
    print("Simple GMSH 2D Diode Test")
    print("=" * 40)
    
    device = "SimpleDevice"
    region = "Bulk"
    
    try:
        # Create GMSH mesh (following gmsh_diode2d.py pattern)
        print("Loading GMSH mesh...")
        create_gmsh_mesh(mesh="simple2d", file="simple_diode_2d.msh")
        add_gmsh_region(mesh="simple2d", gmsh_name="Bulk", region=region, material="Silicon")
        add_gmsh_contact(mesh="simple2d", gmsh_name="Base", region=region, material="metal", name="top")
        add_gmsh_contact(mesh="simple2d", gmsh_name="Emitter", region=region, material="metal", name="bot")
        finalize_mesh(mesh="simple2d")
        create_device(mesh="simple2d", device=device)
        print("GMSH mesh loaded successfully")
        
        # Set parameters (following diode_common.py pattern)
        print("Setting parameters...")
        SetSiliconParameters(device, region, 300)
        
        # Set doping (following diode_common.py pattern)
        print("Setting doping...")
        CreateNodeModel(device, region, "Acceptors", "1.0e18*step(0.5e-5-x)")  # P-side (left)
        CreateNodeModel(device, region, "Donors", "1.0e18*step(x-0.5e-5)")     # N-side (right)
        CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
        
        # Initial solution (following diode_common.py pattern)
        print("Initial solution...")
        CreateSolution(device, region, "Potential")
        CreateSiliconPotentialOnly(device, region)
        
        for contact in get_contact_list(device=device):
            set_parameter(device=device, name=GetContactBiasName(contact), value=0.0)
            CreateSiliconPotentialOnlyContact(device, region, contact)
        
        # Initial DC solution
        print("Solving Poisson...")
        solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)
        
        # Drift diffusion setup (following diode_common.py pattern)
        print("Setting up drift-diffusion...")
        CreateSolution(device, region, "Electrons")
        CreateSolution(device, region, "Holes")
        
        set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
        set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
        
        CreateSiliconDriftDiffusion(device, region)
        for contact in get_contact_list(device=device):
            CreateSiliconDriftDiffusionAtContact(device, region, contact)
        
        # Equilibrium solution
        print("Solving drift-diffusion at equilibrium...")
        solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
        
        # Test a few voltage points
        print("\nTesting voltage sweep...")
        voltages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        currents = []
        
        for v in voltages:
            print(f"V = {v:.1f}V")
            set_parameter(device=device, name=GetContactBiasName("top"), value=v)
            
            try:
                solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
                
                # Get current
                i_electron = get_contact_current(device=device, contact="top", equation="ElectronContinuityEquation")
                i_hole = get_contact_current(device=device, contact="top", equation="HoleContinuityEquation")
                i_total = i_electron + i_hole
                
                currents.append(i_total)
                print(f"  Current: {i_total:.2e} A")
                
            except Exception as e:
                print(f"  Error: {e}")
                currents.append(np.nan)
        
        # Simple plot
        plt.figure(figsize=(8, 6))
        plt.plot(voltages, currents, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (A)')
        plt.title('Simple GMSH 2D Diode IV')
        plt.grid(True, alpha=0.3)
        plt.savefig('simple_gmsh_diode_iv.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nSimple GMSH diode test completed successfully!")
        print("Plot saved: simple_gmsh_diode_iv.png")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()