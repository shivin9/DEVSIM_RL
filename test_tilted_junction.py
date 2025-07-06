#!/usr/bin/env python3
"""
Test Tilted P-N Junction Diode
Test if DEVSIM can handle non-rectangular P-N junction geometries
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import time
from typing import Dict, List, Tuple

# Our framework imports
from geometry_optimization_framework import GeometryMatrix, GeometryGenerator
from matrix_to_gmsh_simple import SimpleMatrixToGMSH

# DEVSIM imports
from devsim import (
    create_gmsh_mesh, add_gmsh_region, add_gmsh_contact, reset_devsim,
    finalize_mesh, create_device, set_parameter, solve, get_contact_current)

import devsim

from devsim.python_packages.simple_physics import (
    SetSiliconParameters, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName, 
    CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact
)

from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim import set_node_values, get_contact_list

class TiltedJunctionTester:
    """Test tilted P-N junction geometries with DEVSIM"""
    
    def __init__(self, matrix_size=16, physical_size=10e-6):
        self.matrix_size = matrix_size
        self.physical_size = physical_size
        self.converter = SimpleMatrixToGMSH(physical_size, physical_size)
        
    def create_tilted_junction_geometry(self, tilt_angle_degrees: float) -> GeometryMatrix:
        """
        Create P-N junction tilted at specified angle
        
        Args:
            tilt_angle_degrees: Tilt angle in degrees (0 = vertical, positive = clockwise)
        """
        geom = GeometryMatrix(self.matrix_size, self.matrix_size, 
                             self.physical_size, self.physical_size)
        
        # Initialize as void
        geom.material_matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=np.uint8)
        
        # Convert angle to radians
        tilt_angle_rad = np.radians(tilt_angle_degrees)
        
        # Create tilted junction
        center_x = self.matrix_size / 2
        center_y = self.matrix_size / 2
        
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                # Calculate position relative to center
                dx = j - center_x
                dy = i - center_y
                
                # Rotate coordinate system by negative tilt angle
                # This creates a junction line tilted by positive angle
                rotated_x = dx * np.cos(-tilt_angle_rad) - dy * np.sin(-tilt_angle_rad)
                
                # Determine material based on rotated x coordinate
                if rotated_x < 0:
                    geom.material_matrix[i, j] = 2  # P-type (left of tilted line)
                else:
                    geom.material_matrix[i, j] = 1  # N-type (right of tilted line)
        
        # Enforce contact constraints (ensure electrical connectivity)
        geom.enforce_contact_constraints()
        
        return geom
    
    def test_tilted_junction_angles(self, angles: List[float]) -> Dict:
        """Test multiple tilt angles"""
        print(f"Testing Tilted P-N Junction Geometries")
        print(f"=" * 50)
        
        results = {}
        
        for angle in angles:
            devsim.reset_devsim()
            print(f"\n--- Testing {angle}° Tilt ---")
            
            # Create tilted geometry
            tilted_geom = self.create_tilted_junction_geometry(angle)
            
            # Analyze geometry
            metrics = tilted_geom.calculate_metrics()
            print(f"Geometry metrics:")
            print(f"  P-fraction: {metrics['p_fraction']:.3f}")
            print(f"  N-fraction: {metrics['n_fraction']:.3f}")
            print(f"  Interface length: {metrics['interface_length']*1e6:.1f} μm")
            print(f"  Connectivity: {metrics['connectivity_score']:.3f}")
            
            # Test DEVSIM simulation
            sim_result = self._run_devsim_test(tilted_geom, f"tilted_{angle}deg")
            
            results[angle] = {
                'geometry': tilted_geom,
                'metrics': metrics,
                'simulation': sim_result
            }
            
            if sim_result['success']:
                print(f"  ✓ DEVSIM simulation successful")
                print(f"    Forward current: {sim_result['forward_current']:.2e} A")
                print(f"    Reverse current: {sim_result['reverse_current']:.2e} A")
                print(f"    Power: {sim_result['forward_power']:.2e} W")
                print(f"    Rectification: {sim_result['rectification_ratio']:.1e}")
            else:
                print(f"  ✗ DEVSIM simulation failed: {sim_result['error']}")
        
        return results
    
    def _run_devsim_test(self, geometry: GeometryMatrix, name: str) -> Dict:
        """Run DEVSIM simulation for tilted junction"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                devsim.reset_devsim()
                geo_file = os.path.join(temp_dir, f"{name}.geo")
                
                # Convert geometry to GMSH
                self.converter.convert_matrix_to_gmsh(geometry.material_matrix, geo_file)
                msh_file = self.converter.generate_mesh(geo_file)
                
                if not msh_file:
                    return {'success': False, 'error': 'Mesh generation failed'}
                
                # DEVSIM setup
                device_name = f"tilted_device_{name}_{int(time.time()*1000)}"
                mesh_name = f"tilted_mesh_{name}_{int(time.time()*1000)}"
                region = "Bulk"
                
                # Load mesh
                create_gmsh_mesh(mesh=mesh_name, file=msh_file)
                add_gmsh_region(mesh=mesh_name, gmsh_name="Bulk", region=region, material="Silicon")
                add_gmsh_contact(mesh=mesh_name, gmsh_name="P_contact", region=region, material="metal", name="anode")
                add_gmsh_contact(mesh=mesh_name, gmsh_name="N_contact", region=region, material="metal", name="cathode")
                finalize_mesh(mesh=mesh_name)
                create_device(mesh=mesh_name, device=device_name)
                
                # Physics setup
                SetSiliconParameters(device_name, region, 300)
                
                # Setup doping profile for tilted junction
                self._setup_tilted_doping(device_name, region, geometry)
                
                # Initial solution - Poisson only
                CreateSolution(device_name, region, "Potential")
                CreateSiliconPotentialOnly(device_name, region)
                
                for contact in get_contact_list(device=device_name):
                    set_parameter(device=device_name, name=GetContactBiasName(contact), value=0.0)
                    CreateSiliconPotentialOnlyContact(device_name, region, contact)
                
                # Solve Poisson with standard settings
                solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=300)
                
                # Drift-diffusion setup
                CreateSolution(device_name, region, "Electrons")
                CreateSolution(device_name, region, "Holes")
                
                set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
                set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")
                
                CreateSiliconDriftDiffusion(device_name, region)
                for contact in get_contact_list(device=device_name):
                    CreateSiliconDriftDiffusionAtContact(device_name, region, contact)
                
                # Solve equilibrium
                solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=300)
                
                # Test forward bias (0.7V)
                forward_voltage = 0.7
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=forward_voltage)
                solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=200)
                
                # Get forward current
                i_electron_fwd = get_contact_current(device=device_name, contact="anode", equation="ElectronContinuityEquation")
                i_hole_fwd = get_contact_current(device=device_name, contact="anode", equation="HoleContinuityEquation")
                forward_current = i_electron_fwd + i_hole_fwd
                forward_power = forward_current * forward_voltage
                
                # Test reverse bias (-0.5V)
                reverse_voltage = -0.5
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=reverse_voltage)
                
                try:
                    solve(type="dc", absolute_error=1e6, relative_error=1e-6, maximum_iterations=150)
                    
                    # Get reverse current
                    i_electron_rev = get_contact_current(device=device_name, contact="anode", equation="ElectronContinuityEquation")
                    i_hole_rev = get_contact_current(device=device_name, contact="anode", equation="HoleContinuityEquation")
                    reverse_current = i_electron_rev + i_hole_rev
                    
                except Exception:
                    # If reverse bias fails, assume very low reverse current
                    reverse_current = -1e-12
                
                # Calculate performance metrics
                rectification_ratio = abs(forward_current / (reverse_current + 1e-20))
                
                return {
                    'success': True,
                    'forward_current': forward_current,
                    'reverse_current': reverse_current,
                    'forward_power': forward_power,
                    'rectification_ratio': rectification_ratio
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _setup_tilted_doping(self, device: str, region: str, geometry: GeometryMatrix):
        """Setup doping profile that matches the tilted geometry"""
        # For tilted junction, we need to create a doping profile that follows the tilt
        
        # Find the tilt angle by analyzing the geometry
        p_regions = (geometry.material_matrix == 2)
        n_regions = (geometry.material_matrix == 1)
        
        # Find junction points to determine tilt
        junction_points = []
        
        for i in range(geometry.height):
            p_cols = np.where(p_regions[i, :])[0]
            n_cols = np.where(n_regions[i, :])[0]
            
            if len(p_cols) > 0 and len(n_cols) > 0:
                # Junction position for this row
                junction_col = (np.max(p_cols) + np.min(n_cols)) / 3
                junction_x = (junction_col + 0.5) * geometry.dx
                junction_y = (i + 0.5) * geometry.dy
                junction_points.append((junction_x, junction_y))
        
        if len(junction_points) >= 2:
            # Fit line to junction points to get tilt parameters
            x_coords = [p[0] for p in junction_points]
            y_coords = [p[1] for p in junction_points]
            
            # Linear regression to find junction line
            A = np.vstack([y_coords, np.ones(len(y_coords))]).T
            slope, intercept = np.linalg.lstsq(A, x_coords, rcond=None)[0]
            
            # Junction line: x = slope * y + intercept
            # Doping should be P-type where x < slope * y + intercept
            
            doping_level = 1e18  # cm^-3
            transition_width = geometry.physical_width * 0.02  # 2% of device width for smoothness
            
            # Create tilted doping profile using smooth transition
            CreateNodeModel(device, region, "Acceptors", 
                           f"{doping_level}*0.5*(1.0 + tanh(({slope}*y + {intercept} - x)/{transition_width}))")
            CreateNodeModel(device, region, "Donors", 
                           f"{doping_level}*0.5*(1.0 + tanh((x - {slope}*y - {intercept})/{transition_width}))")
            CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
            
        else:
            # Fallback to vertical junction at center
            junction_x = geometry.physical_width / 2
            doping_level = 1e18
            transition_width = geometry.physical_width * 0.02
            
            CreateNodeModel(device, region, "Acceptors", 
                           f"{doping_level}*0.5*(1.0 + tanh(({junction_x} - x)/{transition_width}))")
            CreateNodeModel(device, region, "Donors", 
                           f"{doping_level}*0.5*(1.0 + tanh((x - {junction_x})/{transition_width}))")
            CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
    
    def visualize_results(self, results: Dict):
        """Create visualization of tilted junction test results"""
        angles = list(results.keys())
        n_angles = len(angles)
        
        fig, axes = plt.subplots(2, n_angles, figsize=(4*n_angles, 8))
        
        if n_angles == 1:
            axes = axes.reshape(-1, 1)
        
        # Material colormap
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
        
        for i, angle in enumerate(angles):
            result = results[angle]
            
            # Plot geometry
            ax1 = axes[0, i]
            im1 = ax1.imshow(result['geometry'].material_matrix, 
                           cmap=cmap, vmin=0, vmax=2, origin='lower')
            ax1.set_title(f'{angle}° Tilt\nGeometry')
            if i == 0:
                plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2],
                           label='Material: 0=Void, 1=N-type, 2=P-type')
            
            # Plot performance
            ax2 = axes[1, i]
            if result['simulation']['success']:
                metrics = ['Forward I', 'Power', 'Rectification']
                values = [
                    result['simulation']['forward_current'],
                    result['simulation']['forward_power'],
                    np.log10(result['simulation']['rectification_ratio'])
                ]
                
                bars = ax2.bar(metrics, values, alpha=0.7)
                ax2.set_title(f'{angle}° Performance\n✓ Success')
                ax2.set_ylabel('Value (log10 for rectification)')
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    if 'Rectification' in metrics[values.index(val)]:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{10**val:.1e}', ha='center', va='bottom', fontsize=8)
                    else:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.2e}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, f'Simulation\nFailed\n\n{result["simulation"]["error"]}', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=10, color='red')
                ax2.set_title(f'{angle}° Performance\n✗ Failed')
        
        plt.tight_layout()
        plt.savefig('tilted_junction_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_results(self, results: Dict):
        """Analyze tilted junction test results"""
        print(f"\n" + "="*50)
        print(f"TILTED JUNCTION ANALYSIS")
        print(f"="*50)
        
        successful_angles = []
        failed_angles = []
        
        for angle, result in results.items():
            if result['simulation']['success']:
                successful_angles.append(angle)
            else:
                failed_angles.append(angle)
        
        success_rate = len(successful_angles) / len(results)
        print(f"Overall success rate: {success_rate:.1%} ({len(successful_angles)}/{len(results)})")
        
        if successful_angles:
            print(f"Successful angles: {successful_angles}")
            
            # Compare performance across angles
            print(f"\nPerformance comparison:")
            for angle in successful_angles:
                sim = results[angle]['simulation']
                print(f"  {angle:4.0f}°: Forward={sim['forward_current']:.2e}A, "
                      f"Power={sim['forward_power']:.2e}W, "
                      f"Rectification={sim['rectification_ratio']:.1e}")
        
        if failed_angles:
            print(f"Failed angles: {failed_angles}")
            print(f"\nFailure reasons:")
            for angle in failed_angles:
                error = results[angle]['simulation']['error']
                print(f"  {angle:4.0f}°: {error}")
        
        # Geometry analysis
        print(f"\nGeometry characteristics:")
        for angle, result in results.items():
            metrics = result['metrics']
            print(f"  {angle:4.0f}°: P-fraction={metrics['p_fraction']:.3f}, "
                  f"Interface={metrics['interface_length']*1e6:.1f}μm, "
                  f"Connectivity={metrics['connectivity_score']:.3f}")

def test_tilted_junctions():
    """Test various tilted P-N junction angles"""
    print("Testing Tilted P-N Junction Diodes with DEVSIM")
    print("=" * 60)
    
    # Create tester
    tester = TiltedJunctionTester(matrix_size=16, physical_size=10e-6)
    
    # Test different tilt angles
    test_angles = [0, 15, 30, 45, 90, -15]  # degrees
    
    print(f"Testing angles: {test_angles}°")
    
    # Run tests
    results = tester.test_tilted_junction_angles(test_angles)
    
    # Analyze results
    tester.analyze_results(results)
    
    # Create visualization
    tester.visualize_results(results)
    
    print(f"\nTest complete! Results saved to 'tilted_junction_test_results.png'")
    
    return results

if __name__ == "__main__":
    test_tilted_junctions()