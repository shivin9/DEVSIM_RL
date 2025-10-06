#!/usr/bin/env python3
"""
DEVSIM Custom Diode Geometries

This script demonstrates different ways to create custom diode geometries in DEVSIM:
1. Programmatic 2D rectangular geometry
2. Complex 2D geometry with graded junctions
3. GMSH integration for arbitrary geometries
4. 3D cylindrical/spherical geometries
"""

import numpy as np
import matplotlib.pyplot as plt

# DEVSIM imports
from devsim import (
    create_1d_mesh, create_2d_mesh, create_gmsh_mesh,
    add_1d_mesh_line, add_2d_mesh_line,
    add_1d_contact, add_2d_contact, add_gmsh_contact,
    add_1d_region, add_2d_region, add_gmsh_region,
    finalize_mesh, create_device,
    node_solution, node_model, set_parameter,
    get_node_model_values, print_node_values
)

from devsim.python_packages.simple_physics import SetSiliconParameters

def create_rectangular_diode_2d():
    """
    Create a 2D rectangular diode with custom dimensions and doping regions
    
    Geometry:
    - P+ contact region (heavily doped)
    - P region (lightly doped) 
    - Junction
    - N region (lightly doped)
    - N+ contact region (heavily doped)
    """
    print("=== Creating 2D Rectangular Diode ===")
    
    # Device dimensions (in cm)
    device_width = 50e-4    # 50 μm
    device_height = 20e-4   # 20 μm
    
    # Region boundaries
    p_contact_width = 10e-4   # 10 μm P+ contact
    p_region_width = 15e-4    # 15 μm P region
    junction_pos = 25e-4      # Junction at 25 μm
    n_region_width = 15e-4    # 15 μm N region  
    n_contact_width = 10e-4   # 10 μm N+ contact
    
    # Mesh spacing
    fine_spacing = 0.5e-4     # 0.5 μm near junction
    coarse_spacing = 2e-4     # 2 μm in bulk regions
    
    device = "RectangularDiode"
    region = "Silicon"
    
    # Create 2D mesh
    create_2d_mesh(mesh="rect_diode")
    
    # X-direction mesh lines (along device length)
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=0, ps=coarse_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=p_contact_width, ps=fine_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=p_contact_width + p_region_width, ps=fine_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=junction_pos - fine_spacing, ps=fine_spacing/2)  # Very fine near junction
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=junction_pos, ps=fine_spacing/4)               # Finest at junction
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=junction_pos + fine_spacing, ps=fine_spacing/2)
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=junction_pos + n_region_width, ps=fine_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="x", pos=device_width, ps=coarse_spacing)
    
    # Y-direction mesh lines (device height)
    add_2d_mesh_line(mesh="rect_diode", dir="y", pos=0, ps=coarse_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="y", pos=device_height/4, ps=fine_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="y", pos=device_height/2, ps=fine_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="y", pos=3*device_height/4, ps=fine_spacing)
    add_2d_mesh_line(mesh="rect_diode", dir="y", pos=device_height, ps=coarse_spacing)
    
    # Create silicon region
    add_2d_region(mesh="rect_diode", material="Si", region=region)
    
    # Create contacts (line contacts at boundaries)
    # P+ contact (left side) - vertical line at x=0
    add_2d_contact(mesh="rect_diode", name="anode", material="metal", region=region,
                   xl=0, xh=0, yl=0, yh=device_height, bloat=1e-10)
    
    # N+ contact (right side) - vertical line at x=device_width
    add_2d_contact(mesh="rect_diode", name="cathode", material="metal", region=region,
                   xl=device_width, xh=device_width, yl=0, yh=device_height, bloat=1e-10)
    
    # Finalize and create device
    finalize_mesh(mesh="rect_diode")
    create_device(mesh="rect_diode", device=device)
    
    # Set silicon parameters
    SetSiliconParameters(device, region, 300)
    
    # Create custom doping profile
    create_rectangular_doping(device, region, junction_pos, device_width)
    
    print(f"Created 2D rectangular diode: {device_width*1e4:.1f} × {device_height*1e4:.1f} μm")
    return device, region

def create_rectangular_doping(device, region, junction_pos, device_width):
    """
    Create spatially varying doping profile for rectangular diode
    """
    # Doping levels (cm^-3)
    NA_contact = 1e19    # P+ contact
    NA_bulk = 5e16       # P bulk
    ND_bulk = 5e16       # N bulk  
    ND_contact = 1e19    # N+ contact
    
    # Convert to m^-3
    NA_contact_m3 = NA_contact * 1e6
    NA_bulk_m3 = NA_bulk * 1e6
    ND_bulk_m3 = ND_bulk * 1e6
    ND_contact_m3 = ND_contact * 1e6
    
    # Define contact region boundaries
    p_contact_end = 10e-4
    n_contact_start = device_width - 10e-4
    
    # Create doping profile with spatial dependence
    doping_expr = f"""ifelse(x < {p_contact_end}, {NA_contact_m3},
                      ifelse(x < {junction_pos}, {NA_bulk_m3}, 
                      ifelse(x < {n_contact_start}, -{ND_bulk_m3}, -{ND_contact_m3})))"""
    
    node_model(device=device, region=region, name="NetDoping", equation=doping_expr)
    
    print("Created custom doping profile:")
    print(f"  P+ contact: {NA_contact:.1e} cm^-3")
    print(f"  P bulk: {NA_bulk:.1e} cm^-3") 
    print(f"  N bulk: {ND_bulk:.1e} cm^-3")
    print(f"  N+ contact: {ND_contact:.1e} cm^-3")

def create_circular_diode_2d():
    """
    Create a 2D circular/cylindrical diode geometry
    Uses DEVSIM's ability to create complex 2D shapes
    """
    print("\n=== Creating 2D Circular Diode ===")
    
    device = "CircularDiode"
    region = "Silicon"
    
    # Circular diode parameters
    radius = 25e-4        # 25 μm radius
    center_x = 30e-4      # Center position
    center_y = 30e-4
    junction_radius = 15e-4  # Inner junction radius
    
    # Create rectangular mesh that encompasses the circle
    create_2d_mesh(mesh="circ_diode")
    
    # Create mesh lines for circular region
    mesh_spacing = 2e-4
    fine_spacing = 0.5e-4
    
    # X-direction
    for i in range(int(2*center_x/mesh_spacing) + 1):
        x_pos = i * mesh_spacing
        spacing = fine_spacing if abs(x_pos - center_x) < radius else mesh_spacing
        add_2d_mesh_line(mesh="circ_diode", dir="x", pos=x_pos, ps=spacing)
    
    # Y-direction  
    for i in range(int(2*center_y/mesh_spacing) + 1):
        y_pos = i * mesh_spacing
        spacing = fine_spacing if abs(y_pos - center_y) < radius else mesh_spacing
        add_2d_mesh_line(mesh="circ_diode", dir="y", pos=y_pos, ps=spacing)
    
    # Create silicon region (will be masked by doping)
    add_2d_region(mesh="circ_diode", material="Si", region=region)
    
    # Create contacts at edges of circular regions
    # Center contact (P+) - point contact at center
    add_2d_contact(mesh="circ_diode", name="anode", material="metal", region=region,
                   xl=center_x, xh=center_x, 
                   yl=center_y, yh=center_y, bloat=1e-10)
    
    # Outer ring contact (N+) - line contact at bottom edge
    add_2d_contact(mesh="circ_diode", name="cathode", material="metal", region=region,
                   xl=0, xh=2*center_x,
                   yl=0, yh=0, bloat=1e-10)
    
    finalize_mesh(mesh="circ_diode")
    create_device(mesh="circ_diode", device=device)
    
    SetSiliconParameters(device, region, 300)
    
    # Create circular doping profile
    create_circular_doping(device, region, center_x, center_y, junction_radius, radius)
    
    print(f"Created 2D circular diode: radius = {radius*1e4:.1f} μm")
    return device, region

def create_circular_doping(device, region, center_x, center_y, junction_radius, outer_radius):
    """
    Create radial doping profile for circular diode
    """
    # Doping levels
    NA_center = 1e18 * 1e6    # P+ center (m^-3)
    ND_outer = 1e17 * 1e6     # N+ outer ring (m^-3)
    
    # Distance from center
    distance_expr = f"((x-{center_x})^2 + (y-{center_y})^2)^0.5"
    
    # Radial doping profile
    doping_expr = f"""ifelse({distance_expr} < {junction_radius}, {NA_center},
                      ifelse({distance_expr} < {outer_radius}, -{ND_outer}, 0))"""
    
    node_model(device=device, region=region, name="NetDoping", equation=doping_expr)
    
    print(f"Created circular doping: junction at r = {junction_radius*1e4:.1f} μm")

def create_graded_junction_diode():
    """
    Create a diode with graded junction (smooth doping transition)
    Demonstrates advanced doping profile control
    """
    print("\n=== Creating Graded Junction Diode ===")
    
    device = "GradedDiode"
    region = "Silicon"
    
    # 1D device for simplicity
    device_length = 5e-4   # 5 μm
    junction_center = 2.5e-4  # 2.5 μm
    junction_width = 0.5e-4   # 0.5 μm transition width
    
    create_1d_mesh(mesh="graded_diode")
    
    # Very fine mesh near junction for graded profile
    add_1d_mesh_line(mesh="graded_diode", pos=0, ps=0.1e-4, tag="left")
    add_1d_mesh_line(mesh="graded_diode", pos=junction_center - junction_width, ps=0.01e-4, tag="pre_junction")
    add_1d_mesh_line(mesh="graded_diode", pos=junction_center, ps=0.005e-4, tag="junction")
    add_1d_mesh_line(mesh="graded_diode", pos=junction_center + junction_width, ps=0.01e-4, tag="post_junction")
    add_1d_mesh_line(mesh="graded_diode", pos=device_length, ps=0.1e-4, tag="right")
    
    add_1d_contact(mesh="graded_diode", name="anode", tag="left", material="metal")
    add_1d_contact(mesh="graded_diode", name="cathode", tag="right", material="metal")
    add_1d_region(mesh="graded_diode", material="Si", region=region, tag1="left", tag2="right")
    
    finalize_mesh(mesh="graded_diode")
    create_device(mesh="graded_diode", device=device)
    
    SetSiliconParameters(device, region, 300)
    
    # Create graded doping profile
    create_graded_doping(device, region, junction_center, junction_width)
    
    print(f"Created graded junction diode: transition width = {junction_width*1e4:.1f} μm")
    return device, region

def create_graded_doping(device, region, junction_center, junction_width):
    """
    Create smooth (graded) junction doping profile using tanh function
    """
    # Doping levels
    NA_max = 1e17 * 1e6    # P-side peak doping (m^-3)
    ND_max = 1e17 * 1e6    # N-side peak doping (m^-3)
    
    # Smooth transition using hyperbolic tangent
    # tanh provides smooth S-curve transition
    transition_sharpness = 2.0 / junction_width  # Controls steepness
    
    # Graded doping profile: NA on left, ND on right, smooth transition
    doping_expr = f"""({NA_max} + {ND_max}) * 0.5 * tanh({transition_sharpness} * ({junction_center} - x)) + 
                      ({NA_max} - {ND_max}) * 0.5"""
    
    node_model(device=device, region=region, name="NetDoping", equation=doping_expr)
    
    print(f"Created graded doping profile:")
    print(f"  P-side: {NA_max/1e6:.1e} cm^-3")
    print(f"  N-side: {ND_max/1e6:.1e} cm^-3")
    print(f"  Transition: tanh profile over {junction_width*1e4:.1f} μm")

def create_gmsh_custom_geometry():
    """
    Demonstrate GMSH integration for arbitrary custom geometries
    This shows how to import complex CAD-designed geometries
    """
    print("\n=== GMSH Custom Geometry Integration ===")
    
    # Create a simple GMSH file for demonstration
    gmsh_content = """
// GMSH geometry file for custom diode shape
// This creates a hexagonal diode geometry

lc = 1e-5;  // Characteristic length

// Define hexagon vertices  
Point(1) = {0, 0, 0, lc};
Point(2) = {2e-4, 0, 0, lc};
Point(3) = {3e-4, 1.73e-4, 0, lc};
Point(4) = {2e-4, 3.46e-4, 0, lc};
Point(5) = {0, 3.46e-4, 0, lc};
Point(6) = {-1e-4, 1.73e-4, 0, lc};

// Create hexagon boundary
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

// Create surface
Line Loop(1) = {1, 2, 3, 4, 5, 6};
Plane Surface(1) = {1};

// Physical entities for DEVSIM
Physical Line("anode") = {6, 1};      // Bottom edges for anode
Physical Line("cathode") = {3, 4};    // Top edges for cathode  
Physical Surface("Silicon") = {1};    // Silicon region

// Generate 2D mesh
Mesh 2;
"""
    
    # Write GMSH file
    with open("custom_diode.geo", "w") as f:
        f.write(gmsh_content)
    
    print("Created GMSH geometry file: custom_diode.geo")
    print("To use this:")
    print("1. Run: gmsh -2 custom_diode.geo -o custom_diode.msh")
    print("2. Use create_gmsh_mesh() in DEVSIM")
    print()
    
    # Show how to use GMSH mesh in DEVSIM (requires .msh file)
    example_gmsh_usage = """
# Example GMSH usage in DEVSIM:
create_gmsh_mesh(mesh="custom_hex", file="custom_diode.msh")
add_gmsh_region(mesh="custom_hex", gmsh_name="Silicon", region="Silicon", material="Silicon")
add_gmsh_contact(mesh="custom_hex", gmsh_name="anode", region="Silicon", material="metal", name="anode")
add_gmsh_contact(mesh="custom_hex", gmsh_name="cathode", region="Silicon", material="metal", name="cathode")
finalize_mesh(mesh="custom_hex")
create_device(mesh="custom_hex", device="HexagonalDiode")
"""
    
    print("GMSH integration code:")
    print(example_gmsh_usage)

def visualize_custom_geometry(device, region):
    """
    Visualize the custom geometry and doping profile
    """
    try:
        # Get mesh coordinates
        x_coords = get_node_model_values(device=device, region=region, name="x")
        if len(x_coords) > 1000:  # 2D mesh
            y_coords = get_node_model_values(device=device, region=region, name="y") 
        else:  # 1D mesh
            y_coords = None
        
        # Get doping profile
        doping = get_node_model_values(device=device, region=region, name="NetDoping")
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        if y_coords is not None:  # 2D plot
            plt.subplot(1, 2, 1)
            scatter = plt.scatter(np.array(x_coords)*1e4, np.array(y_coords)*1e4, 
                                c=np.array(doping), cmap='RdBu', s=1)
            plt.colorbar(scatter, label='Net Doping (m⁻³)')
            plt.xlabel('X (μm)')
            plt.ylabel('Y (μm)')
            plt.title(f'{device} - 2D Doping Profile')
            plt.axis('equal')
            
            plt.subplot(1, 2, 2)
            plt.hist(doping, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Net Doping (m⁻³)')
            plt.ylabel('Number of Nodes')
            plt.title('Doping Distribution')
            
        else:  # 1D plot
            plt.subplot(1, 2, 1)
            plt.plot(np.array(x_coords)*1e4, np.array(doping), 'bo-', linewidth=2, markersize=4)
            plt.xlabel('Position (μm)')
            plt.ylabel('Net Doping (m⁻³)')
            plt.title(f'{device} - 1D Doping Profile')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.semilogy(np.array(x_coords)*1e4, np.abs(np.array(doping)), 'ro-', linewidth=2, markersize=4)
            plt.xlabel('Position (μm)')
            plt.ylabel('|Net Doping| (m⁻³)')
            plt.title('Log Scale Doping')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{device.lower()}_geometry.png', dpi=300, bbox_inches='tight')
        print(f"Saved geometry plot: {device.lower()}_geometry.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def main():
    """
    Demonstrate different custom diode geometries
    """
    print("DEVSIM Custom Diode Geometries Demo")
    print("="*50)
    
    # 1. Rectangular diode with custom doping regions
    try:
        device1, region1 = create_rectangular_diode_2d()
        visualize_custom_geometry(device1, region1)
    except Exception as e:
        print(f"Rectangular diode failed: {e}")
    
    # 2. Circular diode geometry  
    try:
        device2, region2 = create_circular_diode_2d()
        visualize_custom_geometry(device2, region2)
    except Exception as e:
        print(f"Circular diode failed: {e}")
    
    # 3. Graded junction diode
    try:
        device3, region3 = create_graded_junction_diode()
        visualize_custom_geometry(device3, region3)
    except Exception as e:
        print(f"Graded junction diode failed: {e}")
    
    # 4. GMSH integration example
    try:
        create_gmsh_custom_geometry()
    except Exception as e:
        print(f"GMSH example failed: {e}")
    
    print("\n=== Summary ===")
    print("DEVSIM provides multiple ways to create custom diode geometries:")
    print("1. Programmatic 2D mesh creation with add_2d_mesh_line()")
    print("2. Complex doping profiles using mathematical expressions")
    print("3. GMSH integration for CAD-designed geometries")
    print("4. Custom contact shapes and positions")
    print("5. Graded junctions with smooth transitions")

if __name__ == "__main__":
    main()