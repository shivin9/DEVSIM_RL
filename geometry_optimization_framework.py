#!/usr/bin/env python3
"""
Geometry Optimization Framework for 2D Diodes
Uses numpy matrices to represent material distribution
Fixed contacts at left (P+) and right (N+) edges
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from scipy import ndimage

class GeometryMatrix:
    """Represents 2D diode geometry as numpy matrix"""
    
    def __init__(self, width: int = 64, height: int = 64, 
                 physical_width: float = 20e-6, physical_height: float = 20e-6):
        """
        Initialize geometry matrix
        
        Args:
            width, height: Grid resolution
            physical_width, physical_height: Physical dimensions in meters
        """
        self.width = width
        self.height = height
        self.physical_width = physical_width
        self.physical_height = physical_height
        
        # Pixel to physical scaling
        self.dx = physical_width / width
        self.dy = physical_height / height
        
        # Material encoding:
        # 0 = Void/Air (insulator)
        # 1 = N-doped Silicon (1e18 cm^-3)
        # 2 = P-doped Silicon (1e18 cm^-3)
        self.material_matrix = np.ones((height, width), dtype=np.uint8)
        
        # Initialize with simple P-N junction at center
        self.material_matrix[:, :width//2] = 2  # P-type (left)
        self.material_matrix[:, width//2:] = 1  # N-type (right)
        
        # Fixed contact regions (cannot be modified)
        self.contact_regions = {
            'P_contact': (slice(None), slice(0, 3)),      # Left edge
            'N_contact': (slice(None), slice(-3, None))   # Right edge
        }
        
    def get_material_matrix(self) -> np.ndarray:
        """Get current material distribution"""
        return self.material_matrix.copy()
    
    def set_material_matrix(self, matrix: np.ndarray):
        """Set material distribution with contact constraints"""
        self.material_matrix = matrix.copy()
        
        # Enforce contact constraints
        self.enforce_contact_constraints()
    
    def enforce_contact_constraints(self):
        """Ensure contacts remain accessible"""
        # P+ contact region must be P-type
        self.material_matrix[self.contact_regions['P_contact']] = 2
        
        # N+ contact region must be N-type  
        self.material_matrix[self.contact_regions['N_contact']] = 1
    
    def apply_modification(self, x: int, y: int, radius: int, material_type: int):
        """Apply circular modification to geometry"""
        # Create circular mask
        Y, X = np.ogrid[:self.height, :self.width]
        mask = (X - x)**2 + (Y - y)**2 <= radius**2
        
        # Apply modification outside contact regions
        modifiable_mask = self.get_modifiable_mask()
        final_mask = mask & modifiable_mask
        
        self.material_matrix[final_mask] = material_type
        
    def get_modifiable_mask(self) -> np.ndarray:
        """Get mask of regions that can be modified (excluding contacts)"""
        mask = np.ones((self.height, self.width), dtype=bool)
        
        # Exclude contact regions
        mask[self.contact_regions['P_contact']] = False
        mask[self.contact_regions['N_contact']] = False
        
        return mask
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate geometry metrics"""
        # Volume fractions
        total_pixels = self.width * self.height
        void_fraction = np.sum(self.material_matrix == 0) / total_pixels
        p_fraction = np.sum(self.material_matrix == 2) / total_pixels
        n_fraction = np.sum(self.material_matrix == 1) / total_pixels
        
        # Interface length (P-N junction length)
        interface_length = self.calculate_interface_length()
        
        # Connectivity
        p_components = self.count_connected_components(self.material_matrix == 2)
        n_components = self.count_connected_components(self.material_matrix == 1)
        
        # Manufacturability score
        min_feature_size = self.calculate_minimum_feature_size()
        
        return {
            'void_fraction': void_fraction,
            'p_fraction': p_fraction,
            'n_fraction': n_fraction,
            'interface_length': interface_length,
            'p_components': p_components,
            'n_components': n_components,
            'min_feature_size': min_feature_size,
            'connectivity_score': 1.0 / (p_components + n_components)
        }
    
    def calculate_interface_length(self) -> float:
        """Calculate total P-N interface length"""
        # Find P-N boundaries
        p_mask = (self.material_matrix == 2)
        n_mask = (self.material_matrix == 1)
        
        # Dilate P regions and find overlap with N
        p_dilated = ndimage.binary_dilation(p_mask)
        interface_pixels = p_dilated & n_mask
        
        # Convert to physical length
        interface_length = np.sum(interface_pixels) * min(self.dx, self.dy)
        return interface_length
    
    def count_connected_components(self, binary_mask: np.ndarray) -> int:
        """Count number of connected components in binary mask"""
        labeled, num_components = ndimage.label(binary_mask)
        return num_components
    
    def calculate_minimum_feature_size(self) -> float:
        """Calculate minimum feature size for manufacturability"""
        min_size = float('inf')
        
        for material_type in [0, 1, 2]:
            mask = (self.material_matrix == material_type)
            if not np.any(mask):
                continue
                
            # Distance transform to find minimum width
            distance = ndimage.distance_transform_edt(mask)
            if np.any(distance > 0):
                min_width_pixels = 2 * np.min(distance[distance > 0])
                min_width_physical = min_width_pixels * min(self.dx, self.dy)
                min_size = min(min_size, min_width_physical)
        
        return min_size if min_size != float('inf') else 0.0
    
    def visualize(self, title: str = "Geometry"):
        """Visualize current geometry"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Material distribution
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
        im1 = ax1.imshow(self.material_matrix, cmap=cmap, vmin=0, vmax=2)
        ax1.set_title(f'{title} - Material Distribution')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
        cbar1.set_ticklabels(['Void', 'N-type', 'P-type'])
        
        # Physical dimensions
        extent = [0, self.physical_width*1e6, 0, self.physical_height*1e6]
        im2 = ax2.imshow(self.material_matrix, cmap=cmap, vmin=0, vmax=2, extent=extent)
        ax2.set_title(f'{title} - Physical Scale')
        ax2.set_xlabel('X (μm)')
        ax2.set_ylabel('Y (μm)')
        
        # Add contact annotations
        ax2.axvline(x=1.5, color='red', linewidth=3, label='P+ Contact')
        ax2.axvline(x=self.physical_width*1e6-1.5, color='blue', linewidth=3, label='N+ Contact')
        ax2.legend()
        
        cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
        cbar2.set_ticklabels(['Void', 'N-type', 'P-type'])
        
        plt.tight_layout()
        return fig
    
    def to_doping_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert material matrix to doping concentrations"""
        N_a = np.zeros_like(self.material_matrix, dtype=float)  # Acceptor concentration
        N_d = np.zeros_like(self.material_matrix, dtype=float)  # Donor concentration
        
        # P-type regions
        N_a[self.material_matrix == 2] = 1e18  # 1e18 cm^-3
        
        # N-type regions  
        N_d[self.material_matrix == 1] = 1e18  # 1e18 cm^-3
        
        # Void regions have zero doping
        
        return N_a, N_d
    
    def save_matrix(self, filename: str):
        """Save geometry matrix to file"""
        np.save(filename, self.material_matrix)
    
    def load_matrix(self, filename: str):
        """Load geometry matrix from file"""
        self.material_matrix = np.load(filename)
        self.enforce_contact_constraints()

class GeometryGenerator:
    """Generate various test geometries"""
    
    @staticmethod
    def create_baseline_rectangular(width: int = 64, height: int = 64) -> GeometryMatrix:
        """Create baseline rectangular P-N junction"""
        geom = GeometryMatrix(width, height)
        # Default initialization is already rectangular
        return geom
    
    @staticmethod
    def create_circular_junction(width: int = 64, height: int = 64, radius_ratio: float = 0.3) -> GeometryMatrix:
        """Create circular P region in N substrate"""
        geom = GeometryMatrix(width, height)
        
        # Start with all N-type
        geom.material_matrix[:, :] = 1
        
        # Create circular P region
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) * radius_ratio
        
        Y, X = np.ogrid[:height, :width]
        mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
        geom.material_matrix[mask] = 2
        
        # Enforce contact constraints
        geom.enforce_contact_constraints()
        
        return geom
    
    @staticmethod
    def create_graded_junction(width: int = 64, height: int = 64, transition_width: int = 10) -> GeometryMatrix:
        """Create graded P-N junction with smooth transition"""
        geom = GeometryMatrix(width, height)
        
        # Create graded transition
        center = width // 2
        for x in range(width):
            if x < center - transition_width // 2:
                geom.material_matrix[:, x] = 2  # P-type
            elif x > center + transition_width // 2:
                geom.material_matrix[:, x] = 1  # N-type
            else:
                # Graded region - for now, just use alternating pattern
                # In actual implementation, this would be continuous doping
                if (x % 2) == 0:
                    geom.material_matrix[:, x] = 2
                else:
                    geom.material_matrix[:, x] = 1
        
        geom.enforce_contact_constraints()
        return geom
    
    @staticmethod
    def create_interdigitated(width: int = 64, height: int = 64, finger_width: int = 4) -> GeometryMatrix:
        """Create interdigitated finger structure"""
        geom = GeometryMatrix(width, height)
        
        # Create alternating fingers
        for x in range(width):
            finger_index = x // finger_width
            if finger_index % 2 == 0:
                geom.material_matrix[:, x] = 2  # P-type
            else:
                geom.material_matrix[:, x] = 1  # N-type
        
        geom.enforce_contact_constraints()
        return geom
    
    @staticmethod
    def create_honeycomb_pattern(width: int = 64, height: int = 64, cell_size: int = 8) -> GeometryMatrix:
        """Create honeycomb-like pattern"""
        geom = GeometryMatrix(width, height)
        
        # Start with N-type background
        geom.material_matrix[:, :] = 1
        
        # Create hexagonal P-type cells
        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                # Create circular P regions
                center_x, center_y = x + cell_size//2, y + cell_size//2
                if center_x < width and center_y < height:
                    radius = cell_size // 3
                    Y, X = np.ogrid[:height, :width]
                    mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
                    geom.material_matrix[mask] = 2
        
        geom.enforce_contact_constraints()
        return geom

def test_geometry_matrix():
    """Test the geometry matrix functionality"""
    print("Testing Geometry Matrix Framework")
    print("=" * 50)
    
    # Test different geometries
    geometries = {
        'Rectangular': GeometryGenerator.create_baseline_rectangular(),
        'Circular': GeometryGenerator.create_circular_junction(),
        'Graded': GeometryGenerator.create_graded_junction(),
        'Interdigitated': GeometryGenerator.create_interdigitated(),
        'Honeycomb': GeometryGenerator.create_honeycomb_pattern()
    }
    
    # Calculate and display metrics
    for name, geom in geometries.items():
        metrics = geom.calculate_metrics()
        print(f"\n{name} Geometry:")
        print(f"  P-fraction: {metrics['p_fraction']:.3f}")
        print(f"  N-fraction: {metrics['n_fraction']:.3f}")
        print(f"  Void-fraction: {metrics['void_fraction']:.3f}")
        print(f"  Interface length: {metrics['interface_length']*1e6:.2f} μm")
        print(f"  Connectivity: {metrics['connectivity_score']:.3f}")
        print(f"  Min feature size: {metrics['min_feature_size']*1e6:.2f} μm")
        
        # Save visualization
        fig = geom.visualize(name)
        plt.savefig(f'{name.lower()}_geometry.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save matrix
        geom.save_matrix(f'{name.lower()}_matrix.npy')
    
    print(f"\nGenerated {len(geometries)} test geometries")
    print("Visualizations saved as PNG files")
    print("Matrices saved as NPY files")

if __name__ == "__main__":
    test_geometry_matrix()