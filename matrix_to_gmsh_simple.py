#!/usr/bin/env python3
"""
Simple Matrix to GMSH converter for RL-optimized geometries
Converts numpy material matrices to GMSH geometry files
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from scipy import ndimage
import subprocess
import os

class SimpleMatrixToGMSH:
    """Simple converter from material matrix to GMSH"""
    
    def __init__(self, physical_width: float = 20e-6, physical_height: float = 20e-6):
        self.physical_width = physical_width
        self.physical_height = physical_height
        
    def convert_matrix_to_gmsh(self, material_matrix: np.ndarray, output_file: str) -> str:
        """Convert material matrix to GMSH .geo file"""
        height, width = material_matrix.shape
        
        # Simple approach: create rectangular regions based on material distribution
        geo_code = self._generate_simple_geo(material_matrix)
        
        with open(output_file, 'w') as f:
            f.write(geo_code)
        
        return output_file
    
    def _generate_simple_geo(self, material_matrix: np.ndarray) -> str:
        """Generate simplified GMSH geometry"""
        height, width = material_matrix.shape
        
        lines = []
        lines.append("// Auto-generated GMSH geometry from RL optimization")
        lines.append(f"// Dimensions: {self.physical_width*1e6:.1f} x {self.physical_height*1e6:.1f} μm")
        lines.append("")
        
        # Parameters
        lines.append("mesh_size = 1e-6;")
        lines.append("")
        
        # Create background rectangle (will be mostly N-type)
        lines.append("// Background domain")
        lines.append("Point(1) = {0, 0, 0, mesh_size};")
        lines.append(f"Point(2) = {{{self.physical_width}, 0, 0, mesh_size}};")
        lines.append(f"Point(3) = {{{self.physical_width}, {self.physical_height}, 0, mesh_size}};")
        lines.append(f"Point(4) = {{0, {self.physical_height}, 0, mesh_size}};")
        
        lines.append("Line(1) = {1, 2};")  # Bottom
        lines.append("Line(2) = {2, 3};")  # Right  
        lines.append("Line(3) = {3, 4};")  # Top
        lines.append("Line(4) = {4, 1};")  # Left
        
        lines.append("Curve Loop(1) = {1, 2, 3, 4};")
        lines.append("Plane Surface(1) = {1};")
        lines.append("")
        
        # For simplicity, create a doping profile using a mathematical function
        # instead of complex geometric regions
        lines.append("// Contacts")
        lines.append("Physical Curve(\"P_contact\") = {4};  // Left edge")
        lines.append("Physical Curve(\"N_contact\") = {2};  // Right edge")
        lines.append("")
        
        lines.append("// Physical region (single bulk)")
        lines.append("Physical Surface(\"Bulk\") = {1};")
        lines.append("")
        
        lines.append("// Mesh options")
        lines.append("Mesh.Algorithm = 6;")
        lines.append("Mesh.ElementOrder = 1;")
        
        return "\n".join(lines)
    
    def matrix_to_doping_function(self, material_matrix: np.ndarray) -> str:
        """Convert material matrix to DEVSIM doping function"""
        height, width = material_matrix.shape
        
        # Create piecewise doping function based on matrix
        dx = self.physical_width / width
        dy = self.physical_height / height
        
        # For now, use a simplified approach
        # Find the average P-N boundary
        p_boundary = 0
        for x in range(width):
            col = material_matrix[:, x]
            if np.any(col == 2):  # P-type present
                p_boundary = (x + 1) * dx
        
        # Create step function approximation
        doping_function = f"1e18*step({p_boundary}-x) - 1e18*step(x-{p_boundary})"
        
        return doping_function
    
    def generate_mesh(self, geo_file: str) -> str:
        """Generate mesh from .geo file"""
        msh_file = geo_file.replace('.geo', '.msh')
        
        try:
            cmd = f"gmsh -2 {geo_file} -format msh2 -o {msh_file}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Mesh generated: {msh_file}")
                return msh_file
            else:
                print(f"GMSH error: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def visualize_matrix(self, material_matrix: np.ndarray, title: str = "Material Matrix"):
        """Visualize material matrix"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
        im = ax.imshow(material_matrix, cmap=cmap, vmin=0, vmax=2, origin='lower')
        
        ax.set_title(title)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add contact indicators
        ax.axvline(x=0, color='red', linewidth=3, alpha=0.7, label='P+ Contact')
        ax.axvline(x=material_matrix.shape[1]-1, color='blue', linewidth=3, alpha=0.7, label='N+ Contact')
        ax.legend()
        
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.set_ticklabels(['Void', 'N-type', 'P-type'])
        
        plt.tight_layout()
        return fig

def test_simple_conversion():
    """Test the simple converter"""
    print("Testing Simple Matrix to GMSH Conversion")
    print("=" * 40)
    
    converter = SimpleMatrixToGMSH()
    
    # Create test matrices
    test_matrices = {
        'rectangular': create_rectangular_matrix(),
        'circular': create_circular_matrix(),
        'random': create_random_matrix()
    }
    
    for name, matrix in test_matrices.items():
        print(f"\nTesting {name} geometry...")
        
        # Visualize
        fig = converter.visualize_matrix(matrix, f"{name.title()} Geometry")
        plt.savefig(f'{name}_test_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Convert to GMSH
        geo_file = f'{name}_test.geo'
        converter.convert_matrix_to_gmsh(matrix, geo_file)
        
        # Generate doping function
        doping_func = converter.matrix_to_doping_function(matrix)
        print(f"  Doping function: {doping_func}")
        
        # Generate mesh
        msh_file = converter.generate_mesh(geo_file)
        
        # Print stats
        stats = calculate_matrix_stats(matrix)
        print(f"  Stats: P={stats['p_fraction']:.2f}, N={stats['n_fraction']:.2f}, Void={stats['void_fraction']:.2f}")
        
        print(f"  Files: {geo_file}, {msh_file}")
    
    print("\nSimple conversion test completed!")

def create_rectangular_matrix(size: int = 32) -> np.ndarray:
    """Create rectangular P-N junction"""
    matrix = np.ones((size, size), dtype=np.uint8)
    matrix[:, :size//2] = 2  # P-type left half
    return matrix

def create_circular_matrix(size: int = 32) -> np.ndarray:
    """Create circular P-region"""
    matrix = np.ones((size, size), dtype=np.uint8)  # N-type background
    
    center = size // 2
    radius = size // 3
    
    Y, X = np.ogrid[:size, :size]
    mask = (X - center)**2 + (Y - center)**2 <= radius**2
    matrix[mask] = 2  # P-type circle
    
    return matrix

def create_random_matrix(size: int = 32) -> np.ndarray:
    """Create random geometry (like RL would produce)"""
    matrix = np.ones((size, size), dtype=np.uint8)  # Start with N-type
    
    # Add random P-type regions
    np.random.seed(42)  # For reproducibility
    for _ in range(5):
        x = np.random.randint(size//4, 3*size//4)
        y = np.random.randint(0, size)
        radius = np.random.randint(3, size//4)
        
        Y, X = np.ogrid[:size, :size]
        mask = (X - x)**2 + (Y - y)**2 <= radius**2
        matrix[mask] = 2
    
    # Add some voids
    for _ in range(2):
        x = np.random.randint(size//4, 3*size//4)
        y = np.random.randint(0, size)
        radius = np.random.randint(2, size//6)
        
        Y, X = np.ogrid[:size, :size]
        mask = (X - x)**2 + (Y - y)**2 <= radius**2
        matrix[mask] = 0
    
    return matrix

def calculate_matrix_stats(matrix: np.ndarray) -> Dict[str, float]:
    """Calculate material statistics"""
    total = matrix.size
    return {
        'void_fraction': np.sum(matrix == 0) / total,
        'n_fraction': np.sum(matrix == 1) / total,
        'p_fraction': np.sum(matrix == 2) / total
    }

if __name__ == "__main__":
    test_simple_conversion()