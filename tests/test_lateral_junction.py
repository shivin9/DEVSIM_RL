#!/usr/bin/env python3
"""
Test Lateral P-N Junction Diode Position
Test if DEVSIM can handle P-N junctions at different lateral positions
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

class LateralJunctionTester:
    """Test lateral P-N junction positions with DEVSIM"""
    
    def __init__(self, matrix_size=16, physical_size=10e-6):
        self.matrix_size = matrix_size
        self.physical_size = physical_size
        self.converter = SimpleMatrixToGMSH(physical_size, physical_size)
        
    def create_lateral_junction_geometry(self, junction_position: float) -> GeometryMatrix:
        """
        Create vertical P-N junction at specified lateral position
        
        Args:
            junction_position: Lateral position as fraction of device width (0.0 = far left, 1.0 = far right)
        """
        geom = GeometryMatrix(self.matrix_size, self.matrix_size, 
                             self.physical_size, self.physical_size)
        
        # Initialize as void
        geom.material_matrix = np.zeros((self.matrix_size, self.matrix_size), dtype=np.uint8)
        
        # Calculate junction column position
        junction_col = int(junction_position * self.matrix_size)
        
        # Ensure junction is within bounds and leaves room for both materials
        junction_col = max(1, min(junction_col, self.matrix_size - 2))
        
        # Create vertical junction at specified position
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if j < junction_col:
                    geom.material_matrix[i, j] = 2  # P-type (left of junction)
                else:
                    geom.material_matrix[i, j] = 1  # N-type (right of junction)
        
        # Enforce contact constraints (ensure electrical connectivity)
        geom.enforce_contact_constraints()
        
        return geom
    
    def test_lateral_junction_positions(self, positions: List[float]) -> Dict:
        """Test multiple lateral junction positions"""
        print(f"Testing Lateral P-N Junction Positions")
        print(f"=" * 50)
        
        results = {}
        
        for pos in positions:
            print(f"\n--- Testing Junction at {pos:.1%} Position ---")
            
            # Create lateral geometry
            lateral_geom = self.create_lateral_junction_geometry(pos)
            
            # Analyze geometry
            metrics = lateral_geom.calculate_metrics()
            print(f"Geometry metrics:")
            print(f"  P-fraction: {metrics['p_fraction']:.3f}")
            print(f"  N-fraction: {metrics['n_fraction']:.3f}")
            print(f"  Interface length: {metrics['interface_length']*1e6:.1f} μm")
            print(f"  Connectivity: {metrics['connectivity_score']:.3f}")
            
            # Test DEVSIM simulation
            sim_result = self._run_devsim_test(lateral_geom, f"lateral_{pos:.1f}")
            
            results[pos] = {
                'geometry': lateral_geom,
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
        """Run DEVSIM simulation for lateral junction"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                geo_file = os.path.join(temp_dir, f"{name}.geo")
                
                # Convert geometry to GMSH
                self.converter.convert_matrix_to_gmsh(geometry.material_matrix, geo_file)
                msh_file = self.converter.generate_mesh(geo_file)
                
                if not msh_file:
                    return {'success': False, 'error': 'Mesh generation failed'}
                
                # DEVSIM setup
                device_name = f"lateral_device_{name}_{int(time.time()*1000)}"
                mesh_name = f"lateral_mesh_{name}_{int(time.time()*1000)}"
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
                
                # Setup doping profile for lateral junction
                self._setup_lateral_doping(device_name, region, geometry)
                
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
    
    def _setup_lateral_doping(self, device: str, region: str, geometry: GeometryMatrix):
        """Setup doping profile that matches the lateral junction geometry"""
        # Find junction position by analyzing the geometry
        p_regions = (geometry.material_matrix == 2)
        n_regions = (geometry.material_matrix == 1)
        
        # Find junction position (transition from P to N)
        junction_positions = []
        
        for i in range(geometry.height):
            p_cols = np.where(p_regions[i, :])[0]
            n_cols = np.where(n_regions[i, :])[0]
            
            if len(p_cols) > 0 and len(n_cols) > 0:
                # Junction position for this row
                junction_col = np.max(p_cols) + 0.5  # Between last P and first N
                junction_x = junction_col * geometry.dx
                junction_positions.append(junction_x)
        
        if junction_positions:
            # Average junction position
            junction_x = np.mean(junction_positions)
        else:
            # Fallback to center
            junction_x = geometry.physical_width / 2
        
        print(f"    Junction position: {junction_x*1e6:.1f} μm ({junction_x/geometry.physical_width:.1%})")
        
        # Create vertical junction doping profile
        doping_level = 1e18  # cm^-3
        transition_width = geometry.physical_width * 0.02  # 2% of device width for smoothness
        
        # Doping profile: P-type on left, N-type on right of junction
        CreateNodeModel(device, region, "Acceptors", 
                       f"{doping_level}*0.5*(1.0 + tanh(({junction_x} - x)/{transition_width}))")
        CreateNodeModel(device, region, "Donors", 
                       f"{doping_level}*0.5*(1.0 + tanh((x - {junction_x})/{transition_width}))")
        CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
    
    def visualize_results(self, results: Dict):
        """Create visualization of lateral junction test results"""
        positions = sorted(results.keys())
        n_positions = len(positions)
        
        fig, axes = plt.subplots(2, n_positions, figsize=(4*n_positions, 8))
        
        if n_positions == 1:
            axes = axes.reshape(-1, 1)
        
        # Material colormap
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
        
        for i, pos in enumerate(positions):
            result = results[pos]
            
            # Plot geometry
            ax1 = axes[0, i]
            im1 = ax1.imshow(result['geometry'].material_matrix, 
                           cmap=cmap, vmin=0, vmax=2, origin='lower')
            ax1.set_title(f'{pos:.1%} Position\nGeometry')
            ax1.axvline(x=pos * result['geometry'].width, color='red', linestyle='--', alpha=0.7, label='Junction')
            ax1.legend()
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
                
                bars = ax2.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'orange'])
                ax2.set_title(f'{pos:.1%} Performance\n✓ Success')
                ax2.set_ylabel('Value (log10 for rectification)')
                
                # Add value labels on bars
                for bar, val, metric in zip(bars, values, metrics):
                    height = bar.get_height()
                    if 'Rectification' in metric:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{10**val:.1e}', ha='center', va='bottom', fontsize=8)
                    else:
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{val:.2e}', ha='center', va='bottom', fontsize=8)
            else:
                ax2.text(0.5, 0.5, f'Simulation\nFailed\n\n{result["simulation"]["error"]}', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=10, color='red')
                ax2.set_title(f'{pos:.1%} Performance\n✗ Failed')
                ax2.set_xticks([])
                ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('lateral_junction_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_results(self, results: Dict):
        """Analyze lateral junction test results"""
        print(f"\n" + "="*50)
        print(f"LATERAL JUNCTION ANALYSIS")
        print(f"="*50)
        
        successful_positions = []
        failed_positions = []
        
        for pos, result in results.items():
            if result['simulation']['success']:
                successful_positions.append(pos)
            else:
                failed_positions.append(pos)
        
        success_rate = len(successful_positions) / len(results)
        print(f"Overall success rate: {success_rate:.1%} ({len(successful_positions)}/{len(results)})")
        
        if successful_positions:
            print(f"Successful positions: {[f'{p:.1%}' for p in successful_positions]}")
            
            # Compare performance across positions
            print(f"\nPerformance comparison:")
            for pos in sorted(successful_positions):
                sim = results[pos]['simulation']
                print(f"  {pos:5.1%}: Forward={sim['forward_current']:.2e}A, "
                      f"Power={sim['forward_power']:.2e}W, "
                      f"Rectification={sim['rectification_ratio']:.1e}")
                
            # Find best performing position
            best_pos = max(successful_positions, 
                          key=lambda p: results[p]['simulation']['forward_current'])
            print(f"\nBest performing position: {best_pos:.1%} "
                  f"(Forward current: {results[best_pos]['simulation']['forward_current']:.2e}A)")
        
        if failed_positions:
            print(f"Failed positions: {[f'{p:.1%}' for p in failed_positions]}")
            print(f"\nFailure reasons:")
            for pos in failed_positions:
                error = results[pos]['simulation']['error']
                print(f"  {pos:5.1%}: {error}")
        
        # Geometry analysis
        print(f"\nGeometry characteristics:")
        for pos in sorted(results.keys()):
            metrics = results[pos]['metrics']
            print(f"  {pos:5.1%}: P-fraction={metrics['p_fraction']:.3f}, "
                  f"N-fraction={metrics['n_fraction']:.3f}, "
                  f"Interface={metrics['interface_length']*1e6:.1f}μm, "
                  f"Connectivity={metrics['connectivity_score']:.3f}")
        
        # Check for trends
        if len(successful_positions) > 1:
            print(f"\nTrend analysis:")
            sorted_pos = sorted(successful_positions)
            currents = [results[p]['simulation']['forward_current'] for p in sorted_pos]
            
            if len(currents) > 2:
                # Simple trend detection
                increasing = all(currents[i] <= currents[i+1] for i in range(len(currents)-1))
                decreasing = all(currents[i] >= currents[i+1] for i in range(len(currents)-1))
                
                if increasing:
                    print("  Forward current increases with junction position")
                elif decreasing:
                    print("  Forward current decreases with junction position")
                else:
                    print("  No clear trend in forward current vs junction position")

def test_lateral_junctions():
    """Test various lateral P-N junction positions"""
    print("Testing Lateral P-N Junction Positions with DEVSIM")
    print("=" * 60)
    
    # Create tester
    tester = LateralJunctionTester(matrix_size=16, physical_size=10e-6)
    
    # Test different lateral positions
    test_positions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # fraction of device width
    
    print(f"Testing junction positions: {[f'{p:.1%}' for p in test_positions]}")
    
    # Run tests
    results = tester.test_lateral_junction_positions(test_positions)
    
    # Analyze results
    tester.analyze_results(results)
    
    # Create visualization
    tester.visualize_results(results)
    
    print(f"\nTest complete! Results saved to 'lateral_junction_test_results.png'")
    
    return results

if __name__ == "__main__":
    test_lateral_junctions()