#!/usr/bin/env python3
"""
Mesh Generator for RL-based Topology Optimization
Converts NumPy grid to GMSH .geo file for semiconductor device simulation

Material codes:
0: Base Silicon (non-optimizable)
1: N-type material
2: P-type material  
3: Contact regions (fixed)
"""

import numpy as np
from collections import deque
import subprocess
import os

def connectivity_check_bfs(grid, material_code):
    """
    Check if all cells with the given material_code form a single connected region
    using Breadth-First Search (BFS).
    
    Args:
        grid: 2D NumPy array representing the device geometry
        material_code: Integer material code to check connectivity for
    
    Returns:
        bool: True if all cells with material_code are connected, False otherwise
    """
    # Find all cells with the specified material
    material_cells = set()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == material_code:
                material_cells.add((i, j))
    
    if not material_cells:
        return True  # No cells of this material - trivially connected
    
    # Start BFS from the first cell found
    start_cell = next(iter(material_cells))
    visited = set()
    queue = deque([start_cell])
    visited.add(start_cell)
    
    # BFS to find all connected cells
    while queue:
        current = queue.popleft()
        row, col = current
        
        # Check all 4 neighbors (up, down, left, right)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if (0 <= new_row < grid.shape[0] and 
                0 <= new_col < grid.shape[1] and
                (new_row, new_col) not in visited and
                grid[new_row, new_col] == material_code):
                
                visited.add((new_row, new_col))
                queue.append((new_row, new_col))
    
    # Check if all material cells were visited
    return len(visited) == len(material_cells)

def extract_interfaces(grid):
    """
    Extract boundaries between different materials using contour detection.
    
    Args:
        grid: 2D NumPy array representing the device geometry
    
    Returns:
        dict: Dictionary mapping material pairs to their interface contours
    """
    try:
        from skimage import measure
    except ImportError:
        raise ImportError("scikit-image is required. Install with: pip install scikit-image")
    
    interfaces = {}
    
    # Find contours for each material boundary
    unique_materials = np.unique(grid)
    
    for i, mat1 in enumerate(unique_materials):
        for mat2 in unique_materials[i+1:]:
            # Create binary mask for this material pair
            mask = (grid == mat1).astype(float)
            
            # Find contours at level 0.5 (boundary between 0 and 1)
            contours = measure.find_contours(mask, 0.5)
            
            if contours:
                # Store the longest contour (main boundary)
                main_contour = max(contours, key=len)
                interfaces[(mat1, mat2)] = main_contour
    
    return interfaces

def generate_mesh_from_grid(grid, output_file="device.geo", 
                          domain_size=(10e-6, 10e-6), 
                          mesh_size=1e-6):
    """
    Generate GMSH .geo file from NumPy grid representation.
    
    Args:
        grid: 2D NumPy array with material codes
        output_file: Path to output .geo file
        domain_size: Physical size of the domain in meters (width, height)
        mesh_size: Default mesh element size in meters
    
    Returns:
        str: Path to generated .geo file
    """
    
    # Check connectivity of critical materials
    if not connectivity_check_bfs(grid, 1):  # N-type
        raise ValueError("N-type material is not connected!")
    if not connectivity_check_bfs(grid, 2):  # P-type
        raise ValueError("P-type material is not connected!")
    
    # Extract interfaces between materials
    interfaces = extract_interfaces(grid)
    
    # Generate GMSH geometry
    geo_content = generate_gmsh_geometry(grid, interfaces, domain_size, mesh_size)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write(geo_content)
    
    return output_file

def generate_gmsh_geometry(grid, interfaces, domain_size, mesh_size):
    """
    Generate the actual GMSH .geo file content.
    
    Args:
        grid: 2D NumPy array with material codes
        interfaces: Dictionary of material interfaces from extract_interfaces
        domain_size: Physical size (width, height) in meters
        mesh_size: Default mesh element size in meters
    
    Returns:
        str: Complete GMSH .geo file content
    """
    
    width, height = domain_size
    rows, cols = grid.shape
    
    # Scale factors to convert grid indices to physical coordinates
    x_scale = width / cols
    y_scale = height / rows
    
    geo_lines = []
    
    # GMSH header
    geo_lines.append("// Generated GMSH geometry for RL topology optimization")
    geo_lines.append("SetFactory(\"OpenCASCADE\");")
    geo_lines.append("")
    
    # Define mesh characteristic length
    geo_lines.append(f"lc = {mesh_size};")
    geo_lines.append("")
    
    # Create overall domain rectangle
    geo_lines.append("// Domain boundary")
    geo_lines.append("Rectangle(1) = {0, 0, 0, %.6e, %.6e, 0};" % (width, height))
    geo_lines.append("")
    
    # Create regions based on material distribution
    regions = create_material_regions(grid, x_scale, y_scale, geo_lines)
    
    # Add P-N junction interface refinement
    add_interface_refinement(interfaces, x_scale, y_scale, mesh_size, geo_lines)
    
    # Define physical regions for DEVSIM
    geo_lines.append("// Physical regions")
    geo_lines.append("Physical Surface(\"Silicon\") = {1};")
    geo_lines.append("")
    
    # Add contacts (assume contacts are at boundaries)
    add_contacts(grid, width, height, geo_lines)
    
    # Mesh refinement
    geo_lines.append("// Mesh refinement")
    geo_lines.append("Mesh.CharacteristicLengthMin = %.6e;" % (mesh_size * 0.1))
    geo_lines.append("Mesh.CharacteristicLengthMax = %.6e;" % (mesh_size * 2.0))
    geo_lines.append("")
    
    return "\n".join(geo_lines)

def create_material_regions(grid, x_scale, y_scale, geo_lines):
    """
    Create GMSH regions for different materials.
    
    Args:
        grid: 2D NumPy array with material codes
        x_scale: Scale factor for x-coordinates
        y_scale: Scale factor for y-coordinates
        geo_lines: List to append GMSH commands to
    
    Returns:
        dict: Mapping of material codes to GMSH surface IDs
    """
    
    regions = {}
    
    # For simplicity, create rectangular regions based on material distribution
    # In practice, you'd use the interfaces for more complex shapes
    
    unique_materials = np.unique(grid)
    
    for material in unique_materials:
        if material == 0:  # Base silicon - use the entire domain
            continue
            
        # Find bounding box of this material
        material_mask = (grid == material)
        rows, cols = np.where(material_mask)
        
        if len(rows) == 0:
            continue
        
        # Get bounding box
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)
        
        # Convert to physical coordinates
        x_min = min_col * x_scale
        x_max = (max_col + 1) * x_scale
        y_min = min_row * y_scale
        y_max = (max_row + 1) * y_scale
        
        geo_lines.append(f"// Material {material} region")
        geo_lines.append(f"Rectangle({material + 10}) = {{%.6e, %.6e, 0, %.6e, %.6e, 0}};" % 
                        (x_min, y_min, x_max - x_min, y_max - y_min))
        
        regions[material] = material + 10
    
    return regions

def add_interface_refinement(interfaces, x_scale, y_scale, mesh_size, geo_lines):
    """
    Add mesh refinement along material interfaces.
    
    Args:
        interfaces: Dictionary of material interfaces
        x_scale: Scale factor for x-coordinates
        y_scale: Scale factor for y-coordinates
        mesh_size: Default mesh size
        geo_lines: List to append GMSH commands to
    """
    
    geo_lines.append("// Interface refinement")
    
    point_id = 1000  # Start point IDs from 1000
    line_id = 1000   # Start line IDs from 1000
    
    for (mat1, mat2), contour in interfaces.items():
        if len(contour) < 2:
            continue
            
        # Create points along the interface
        point_ids = []
        for i, (row, col) in enumerate(contour):
            # Convert to physical coordinates
            x = col * x_scale
            y = row * y_scale
            
            geo_lines.append(f"Point({point_id}) = {{%.6e, %.6e, 0, %.6e}};" % 
                           (x, y, mesh_size * 0.1))
            point_ids.append(point_id)
            point_id += 1
        
        # Create spline through the points
        if len(point_ids) >= 2:
            point_list = ", ".join(map(str, point_ids))
            geo_lines.append(f"Spline({line_id}) = {{{point_list}}};")
            line_id += 1
    
    geo_lines.append("")

def add_contacts(grid, width, height, geo_lines):
    """
    Add contact definitions to the GMSH geometry.
    
    Args:
        grid: 2D NumPy array with material codes
        width: Domain width in meters
        height: Domain height in meters
        geo_lines: List to append GMSH commands to
    """
    
    geo_lines.append("// Contacts")
    
    # Assume contacts are at the boundaries
    # Left contact (anode)
    geo_lines.append("Physical Line(\"anode\") = {4};")  # Left edge of rectangle
    
    # Right contact (cathode)
    geo_lines.append("Physical Line(\"cathode\") = {2};")  # Right edge of rectangle
    
    geo_lines.append("")

def generate_and_mesh(grid, output_prefix="device", domain_size=(10e-6, 10e-6), mesh_size=1e-6):
    """
    Complete pipeline: Generate .geo file and create mesh.
    
    Args:
        grid: 2D NumPy array with material codes
        output_prefix: Prefix for output files
        domain_size: Physical domain size in meters
        mesh_size: Default mesh size in meters
    
    Returns:
        str: Path to generated .msh file
    """
    
    geo_file = f"{output_prefix}.geo"
    msh_file = f"{output_prefix}.msh"
    
    # Generate .geo file
    generate_mesh_from_grid(grid, geo_file, domain_size, mesh_size)
    
    # Call GMSH to generate mesh
    try:
        subprocess.run(["gmsh", geo_file, "-2", "-o", msh_file], 
                      check=True, capture_output=True)
        print(f"Successfully generated mesh: {msh_file}")
        return msh_file
    except subprocess.CalledProcessError as e:
        print(f"GMSH failed: {e}")
        return None
    except FileNotFoundError:
        print("GMSH not found. Please install GMSH and add it to PATH.")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Create a simple test grid (20x20)
    test_grid = np.zeros((20, 20), dtype=int)
    
    # Create a simple P-N junction
    test_grid[:, :10] = 2  # P-type on left
    test_grid[:, 10:] = 1  # N-type on right
    
    # Add some contacts
    test_grid[:, 0] = 3   # Left contact
    test_grid[:, -1] = 3  # Right contact
    
    print("Test grid shape:", test_grid.shape)
    print("Unique materials:", np.unique(test_grid))
    
    # Test connectivity
    print("N-type connected:", connectivity_check_bfs(test_grid, 1))
    print("P-type connected:", connectivity_check_bfs(test_grid, 2))
    
    # Generate mesh
    try:
        msh_file = generate_and_mesh(test_grid, "test_device", 
                                   domain_size=(10e-6, 5e-6), 
                                   mesh_size=1e-6)
        if msh_file:
            print(f"Generated mesh file: {msh_file}")
    except Exception as e:
        print(f"Error generating mesh: {e}")
        
    # Also generate just the .geo file to inspect
    geo_file = generate_mesh_from_grid(test_grid, "test_device.geo", 
                                     domain_size=(10e-6, 5e-6), 
                                     mesh_size=1e-6)
    print(f"\nGenerated .geo file: {geo_file}")
    print("\nGMSH .geo file content:")
    with open(geo_file, 'r') as f:
        print(f.read())