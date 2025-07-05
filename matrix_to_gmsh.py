#!/usr/bin/env python3
"""
Convert numpy material matrices to GMSH geometry files
Handles voids, P-type, and N-type regions with proper meshing
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy import ndimage
from skimage import measure
import subprocess
import os

class MatrixToGMSH:
    """Convert material distribution matrix to GMSH geometry"""
    
    def __init__(self, physical_width: float = 20e-6, physical_height: float = 20e-6):
        """
        Initialize converter
        
        Args:
            physical_width, physical_height: Physical dimensions in meters
        """
        self.physical_width = physical_width
        self.physical_height = physical_height
        
    def convert_matrix_to_gmsh(self, 
                              material_matrix: np.ndarray, 
                              output_file: str,
                              mesh_size: float = 1e-6) -> str:
        """
        Convert material matrix to GMSH .geo file
        
        Args:
            material_matrix: 2D array with material encoding (0=void, 1=N, 2=P)
            output_file: Output .geo filename
            mesh_size: Target mesh size in meters
            
        Returns:
            Path to generated .geo file
        """
        height, width = material_matrix.shape
        
        # Calculate scaling
        dx = self.physical_width / width
        dy = self.physical_height / height
        
        # Extract regions for each material type
        regions = self._extract_regions(material_matrix, dx, dy)
        
        # Generate GMSH geometry code
        geo_code = self._generate_geo_code(regions, mesh_size)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write(geo_code)
        
        return output_file
    
    def _extract_regions(self, material_matrix: np.ndarray, dx: float, dy: float) -> Dict[str, List]:
        """Extract contours for each material type"""
        height, width = material_matrix.shape
        regions = {'void': [], 'n_type': [], 'p_type': []}
        
        # Process each material type
        for material_id, region_name in [(0, 'void'), (1, 'n_type'), (2, 'p_type')]:
            mask = (material_matrix == material_id)
            
            if not np.any(mask):
                continue
            
            # Find contours using marching squares
            try:
                contours = measure.find_contours(mask.astype(float), 0.5)
                
                for contour in contours:
                    # Convert pixel coordinates to physical coordinates
                    physical_contour = []
                    for point in contour:
                        y_pixel, x_pixel = point  # Note: contour gives (row, col)
                        x_phys = x_pixel * dx
                        y_phys = (height - y_pixel) * dy  # Flip Y coordinate
                        physical_contour.append((x_phys, y_phys))
                    
                    # Only keep contours with enough points
                    if len(physical_contour) >= 3:
                        regions[region_name].append(physical_contour)
                        
            except Exception as e:
                print(f"Warning: Could not extract contours for {region_name}: {e}")
        
        return regions
    
    def _generate_geo_code(self, regions: Dict[str, List], mesh_size: float) -> str:
        """Generate GMSH .geo file content"""
        geo_lines = []
        
        # Header
        geo_lines.append("// Auto-generated GMSH geometry from material matrix")
        geo_lines.append(f"// Physical dimensions: {self.physical_width*1e6:.1f} x {self.physical_height*1e6:.1f} μm")
        geo_lines.append("")
        
        # Parameters
        geo_lines.append(f"mesh_size = {mesh_size};")
        geo_lines.append("")
        
        point_id = 1
        line_id = 1
        surface_id = 1
        
        # Create background domain
        geo_lines.append("// Background domain")
        geo_lines.append(f"Point({point_id}) = {{0, 0, 0, mesh_size}};")
        geo_lines.append(f"Point({point_id+1}) = {{{self.physical_width}, 0, 0, mesh_size}};")
        geo_lines.append(f"Point({point_id+2}) = {{{self.physical_width}, {self.physical_height}, 0, mesh_size}};")
        geo_lines.append(f"Point({point_id+3}) = {{0, {self.physical_height}, 0, mesh_size}};")
        
        geo_lines.append(f"Line({line_id}) = {{{point_id}, {point_id+1}}};")  # Bottom
        geo_lines.append(f"Line({line_id+1}) = {{{point_id+1}, {point_id+2}}};")  # Right
        geo_lines.append(f"Line({line_id+2}) = {{{point_id+2}, {point_id+3}}};")  # Top
        geo_lines.append(f"Line({line_id+3}) = {{{point_id+3}, {point_id}}};")  # Left
        
        geo_lines.append(f"Curve Loop(1) = {{{line_id}, {line_id+1}, {line_id+2}, {line_id+3}}};")
        geo_lines.append(f"Plane Surface({surface_id}) = {{1}};")
        
        point_id += 4
        line_id += 4
        surface_id += 1
        
        # Track surfaces for boolean operations
        all_surfaces = [1]  # Background surface
        p_surfaces = []
        void_surfaces = []\n        \n        # Create P-type regions\n        if regions['p_type']:\n            geo_lines.append(\"\\n// P-type regions\")\n            \n            for i, contour in enumerate(regions['p_type']):\n                if len(contour) < 3:\n                    continue\n                    \n                geo_lines.append(f\"// P-type region {i+1}\")\n                \n                # Create points for this contour\n                start_point = point_id\n                for j, (x, y) in enumerate(contour[:-1]):  # Skip last point (same as first)\n                    geo_lines.append(f\"Point({point_id}) = {{{x}, {y}, 0, mesh_size}};\")\n                    point_id += 1\n                \n                # Create lines for this contour\n                start_line = line_id\n                contour_lines = []\n                for j in range(len(contour) - 1):\n                    next_point = start_point + ((j + 1) % (len(contour) - 1))\n                    geo_lines.append(f\"Line({line_id}) = {{{start_point + j}, {next_point}}};\")\n                    contour_lines.append(line_id)\n                    line_id += 1\n                \n                # Create curve loop and surface\n                loop_id = surface_id\n                geo_lines.append(f\"Curve Loop({loop_id}) = {{{', '.join(map(str, contour_lines))}}};\")\n                geo_lines.append(f\"Plane Surface({surface_id}) = {{{loop_id}}};\")\n                \n                p_surfaces.append(surface_id)\n                all_surfaces.append(surface_id)\n                surface_id += 1\n        \n        # Create void regions (holes)\n        if regions['void']:\n            geo_lines.append(\"\\n// Void regions (holes)\")\n            \n            for i, contour in enumerate(regions['void']):\n                if len(contour) < 3:\n                    continue\n                    \n                geo_lines.append(f\"// Void region {i+1}\")\n                \n                # Create points for this contour\n                start_point = point_id\n                for j, (x, y) in enumerate(contour[:-1]):  # Skip last point\n                    geo_lines.append(f\"Point({point_id}) = {{{x}, {y}, 0, mesh_size}};\")\n                    point_id += 1\n                \n                # Create lines for this contour\n                contour_lines = []\n                for j in range(len(contour) - 1):\n                    next_point = start_point + ((j + 1) % (len(contour) - 1))\n                    geo_lines.append(f\"Line({line_id}) = {{{start_point + j}, {next_point}}};\")\n                    contour_lines.append(line_id)\n                    line_id += 1\n                \n                # Create curve loop and surface\n                loop_id = surface_id\n                geo_lines.append(f\"Curve Loop({loop_id}) = {{{', '.join(map(str, contour_lines))}}};\")\n                geo_lines.append(f\"Plane Surface({surface_id}) = {{{loop_id}}};\")\n                \n                void_surfaces.append(surface_id)\n                surface_id += 1\n        \n        # Boolean operations to create final geometry\n        if void_surfaces:\n            geo_lines.append(\"\\n// Boolean operations - subtract voids\")\n            for void_surf in void_surfaces:\n                geo_lines.append(f\"BooleanDifference{{Surface{{1}}; Delete;}}{{Surface{{{void_surf}}}; Delete;}}\")\n        \n        # Contacts (simplified - at left and right edges)\n        geo_lines.append(\"\\n// Contacts\")\n        geo_lines.append(\"Physical Curve(\\\"P_contact\\\") = {4};  // Left edge\")\n        geo_lines.append(\"Physical Curve(\\\"N_contact\\\") = {2};  // Right edge\")\n        \n        # Physical surfaces\n        geo_lines.append(\"\\n// Physical regions\")\n        if p_surfaces:\n            geo_lines.append(f\"Physical Surface(\\\"P_region\\\") = {{{', '.join(map(str, p_surfaces))}}};\")\n        \n        # N-type is the background minus P-type and voids\n        geo_lines.append(\"Physical Surface(\\\"N_region\\\") = {1};  // Background (N-type)\")\n        \n        # Mesh options\n        geo_lines.append(\"\\n// Mesh options\")\n        geo_lines.append(\"Mesh.Algorithm = 6;        // Frontal-Delaunay\")\n        geo_lines.append(\"Mesh.ElementOrder = 1;     // Linear elements\")\n        \n        return \"\\n\".join(geo_lines)\n    \n    def generate_mesh(self, geo_file: str, msh_file: str = None) -> str:\n        \"\"\"Generate mesh from .geo file using GMSH\"\"\"\n        if msh_file is None:\n            msh_file = geo_file.replace('.geo', '.msh')\n        \n        try:\n            # Run GMSH to generate mesh\n            cmd = f\"gmsh -2 {geo_file} -format msh2 -o {msh_file}\"\n            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n            \n            if result.returncode == 0:\n                print(f\"Mesh generated successfully: {msh_file}\")\n                return msh_file\n            else:\n                print(f\"GMSH error: {result.stderr}\")\n                return None\n                \n        except Exception as e:\n            print(f\"Error running GMSH: {e}\")\n            return None\n    \n    def visualize_conversion(self, material_matrix: np.ndarray, title: str = \"Matrix to GMSH\"):\n        \"\"\"Visualize the material matrix and conversion process\"\"\"\n        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n        \n        # Original matrix\n        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])\n        im1 = ax1.imshow(material_matrix, cmap=cmap, vmin=0, vmax=2, origin='lower')\n        ax1.set_title('Original Material Matrix')\n        ax1.set_xlabel('X (pixels)')\n        ax1.set_ylabel('Y (pixels)')\n        \n        # Add colorbar\n        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])\n        cbar1.set_ticklabels(['Void', 'N-type', 'P-type'])\n        \n        # Physical dimensions\n        height, width = material_matrix.shape\n        extent = [0, self.physical_width*1e6, 0, self.physical_height*1e6]\n        im2 = ax2.imshow(material_matrix, cmap=cmap, vmin=0, vmax=2, \n                        extent=extent, origin='lower')\n        ax2.set_title('Physical Scale (for GMSH)')\n        ax2.set_xlabel('X (μm)')\n        ax2.set_ylabel('Y (μm)')\n        \n        # Add contact indicators\n        ax2.axvline(x=0, color='red', linewidth=3, label='P+ Contact')\n        ax2.axvline(x=self.physical_width*1e6, color='blue', linewidth=3, label='N+ Contact')\n        ax2.legend()\n        \n        cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])\n        cbar2.set_ticklabels(['Void', 'N-type', 'P-type'])\n        \n        plt.suptitle(title)\n        plt.tight_layout()\n        return fig\n\ndef test_matrix_to_gmsh():\n    \"\"\"Test the matrix to GMSH conversion\"\"\"\n    print(\"Testing Matrix to GMSH Conversion\")\n    print(\"=\" * 35)\n    \n    # Create converter\n    converter = MatrixToGMSH(physical_width=20e-6, physical_height=20e-6)\n    \n    # Test with different geometries\n    test_cases = [\n        (\"circular\", create_test_circular_matrix()),\n        (\"honeycomb\", create_test_honeycomb_matrix()),\n        (\"complex\", create_test_complex_matrix())\n    ]\n    \n    for name, matrix in test_cases:\n        print(f\"\\nTesting {name} geometry...\")\n        \n        # Visualize matrix\n        fig = converter.visualize_conversion(matrix, f\"{name.title()} Geometry\")\n        plt.savefig(f'{name}_matrix_visualization.png', dpi=150, bbox_inches='tight')\n        plt.close()\n        \n        # Convert to GMSH\n        geo_file = f'{name}_geometry.geo'\n        converter.convert_matrix_to_gmsh(matrix, geo_file)\n        print(f\"Generated: {geo_file}\")\n        \n        # Generate mesh\n        msh_file = converter.generate_mesh(geo_file)\n        if msh_file:\n            print(f\"Generated: {msh_file}\")\n        \n        # Print matrix stats\n        void_frac = np.sum(matrix == 0) / matrix.size\n        n_frac = np.sum(matrix == 1) / matrix.size\n        p_frac = np.sum(matrix == 2) / matrix.size\n        \n        print(f\"  Material fractions: Void={void_frac:.3f}, N={n_frac:.3f}, P={p_frac:.3f}\")\n    \n    print(\"\\nMatrix to GMSH conversion testing complete!\")\n\ndef create_test_circular_matrix(size: int = 32) -> np.ndarray:\n    \"\"\"Create test matrix with circular P-region\"\"\"\n    matrix = np.ones((size, size), dtype=np.uint8)  # N-type background\n    \n    # Create circular P-region\n    center = size // 2\n    radius = size // 4\n    \n    Y, X = np.ogrid[:size, :size]\n    mask = (X - center)**2 + (Y - center)**2 <= radius**2\n    matrix[mask] = 2  # P-type\n    \n    # Add some voids\n    void_centers = [(size//4, size//4), (3*size//4, 3*size//4)]\n    void_radius = size // 8\n    \n    for cx, cy in void_centers:\n        Y, X = np.ogrid[:size, :size]\n        mask = (X - cx)**2 + (Y - cy)**2 <= void_radius**2\n        matrix[mask] = 0  # Void\n    \n    return matrix\n\ndef create_test_honeycomb_matrix(size: int = 32) -> np.ndarray:\n    \"\"\"Create test matrix with honeycomb pattern\"\"\"\n    matrix = np.ones((size, size), dtype=np.uint8)  # N-type background\n    \n    # Create hexagonal P-type cells\n    cell_size = 6\n    for y in range(0, size, cell_size):\n        for x in range(0, size, cell_size):\n            center_x, center_y = x + cell_size//2, y + cell_size//2\n            if center_x < size and center_y < size:\n                radius = cell_size // 3\n                Y, X = np.ogrid[:size, :size]\n                mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2\n                matrix[mask] = 2  # P-type\n    \n    return matrix\n\ndef create_test_complex_matrix(size: int = 32) -> np.ndarray:\n    \"\"\"Create complex test geometry\"\"\"\n    matrix = np.ones((size, size), dtype=np.uint8)  # N-type background\n    \n    # Complex P-type region\n    matrix[:, :size//3] = 2  # Left side P-type\n    \n    # Add some geometric features\n    # Triangle\n    for i in range(size//2):\n        for j in range(i):\n            if i + size//3 < size and j + size//4 < size:\n                matrix[j + size//4, i + size//3] = 2\n    \n    # Voids\n    matrix[size//4:3*size//4, size//8:size//4] = 0\n    matrix[size//8:size//4, 3*size//4:7*size//8] = 0\n    \n    return matrix\n\nif __name__ == \"__main__\":\n    test_matrix_to_gmsh()