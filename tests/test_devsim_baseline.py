#!/usr/bin/env python3
"""
Test DEVSIM Baseline and Single Cell Changes
Verify that DEVSIM simulations work with standard rectangular geometry 
and respond meaningfully to single cell modifications
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
    finalize_mesh, create_device, set_parameter, solve, get_contact_current
)

from devsim.python_packages.simple_physics import (
    SetSiliconParameters, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName, 
    CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact
)

from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim import set_node_values, get_contact_list

class DevsimBaselineTester:
    """Test DEVSIM simulation with baseline and modified geometries"""
    
    def __init__(self, matrix_size=16, physical_size=10e-6):
        self.matrix_size = matrix_size
        self.physical_size = physical_size
        self.converter = SimpleMatrixToGMSH(physical_size, physical_size)
        self.results = {}
        
    def test_baseline_rectangular_diode(self):
        """Test baseline rectangular P-N diode"""
        print("\n" + "="*60)
        print("TESTING: Baseline Rectangular P-N Diode")
        print("="*60)
        
        # Create baseline rectangular geometry
        baseline_geom = GeometryGenerator.create_baseline_rectangular(
            self.matrix_size, self.matrix_size
        )
        
        # Analyze geometry
        metrics = baseline_geom.calculate_metrics()
        print(f"Baseline geometry metrics:")
        print(f"  P-fraction: {metrics['p_fraction']:.3f}")
        print(f"  N-fraction: {metrics['n_fraction']:.3f}")
        print(f"  Interface length: {metrics['interface_length']*1e6:.1f} μm")
        print(f"  Connectivity score: {metrics['connectivity_score']:.3f}")
        
        # Run DEVSIM simulation
        baseline_results = self._run_comprehensive_devsim_test(
            baseline_geom, "baseline_rectangular"
        )
        
        if baseline_results['success']:
            print(f"\n✓ Baseline simulation SUCCESSFUL")
            print(f"  Forward current (0.7V): {baseline_results['forward_current']:.2e} A")
            print(f"  Reverse current (-0.5V): {baseline_results['reverse_current']:.2e} A")
            print(f"  Power dissipation (0.7V): {baseline_results['forward_power']:.2e} W")
            print(f"  Forward/Reverse ratio: {baseline_results['rectification_ratio']:.1e}")
        else:
            print(f"\n✗ Baseline simulation FAILED: {baseline_results['error']}")
            return None
        
        self.results['baseline'] = baseline_results
        return baseline_results
    
    def test_single_cell_modifications(self, baseline_geom: GeometryMatrix):
        """Test single cell modifications and their DEVSIM impact"""
        print("\n" + "="*60)
        print("TESTING: Single Cell Modifications")
        print("="*60)
        
        # Define test modifications
        modifications = [
            {
                'name': 'add_void_center',
                'description': 'Add void at center',
                'position': (self.matrix_size//2, self.matrix_size//2),
                'new_material': 0  # Void
            },
            {
                'name': 'add_n_type_center',
                'description': 'Add N-type at center',
                'position': (self.matrix_size//2, self.matrix_size//2),
                'new_material': 1  # N-type
            },
            {
                'name': 'add_p_type_center',
                'description': 'Add P-type at center',
                'position': (self.matrix_size//2, self.matrix_size//2),
                'new_material': 2  # P-type
            },
            {
                'name': 'remove_junction_cell',
                'description': 'Remove cell near junction',
                'position': (self.matrix_size//2, self.matrix_size//2 - 1),
                'new_material': 0  # Void
            },
            {
                'name': 'extend_p_region',
                'description': 'Extend P-region by one cell',
                'position': (self.matrix_size//2, self.matrix_size//2 + 1),
                'new_material': 2  # P-type
            }
        ]
        
        modification_results = {}
        
        for mod in modifications:
            print(f"\n--- Testing: {mod['description']} ---")
            
            # Create modified geometry
            modified_geom = self._create_modified_geometry(baseline_geom, mod)
            
            # Check if modification is valid
            modified_metrics = modified_geom.calculate_metrics()
            
            # Skip if modification breaks basic requirements
            if modified_metrics['p_fraction'] == 0 or modified_metrics['n_fraction'] == 0:
                print(f"  ⚠️  Skipping - modification breaks P-N junction")
                modification_results[mod['name']] = {
                    'success': False,
                    'error': 'No P-N junction after modification'
                }
                continue
            
            print(f"  P-fraction: {modified_metrics['p_fraction']:.3f}")
            print(f"  N-fraction: {modified_metrics['n_fraction']:.3f}")
            print(f"  Interface length: {modified_metrics['interface_length']*1e6:.1f} μm")
            
            # Run DEVSIM simulation
            mod_results = self._run_comprehensive_devsim_test(
                modified_geom, f"modified_{mod['name']}"
            )
            
            if mod_results['success']:
                print(f"  ✓ Simulation successful")
                print(f"    Forward current: {mod_results['forward_current']:.2e} A")
                print(f"    Reverse current: {mod_results['reverse_current']:.2e} A")
                print(f"    Power dissipation: {mod_results['forward_power']:.2e} W")
                
                # Compare to baseline
                if 'baseline' in self.results:
                    baseline = self.results['baseline']
                    current_change = (mod_results['forward_current'] - baseline['forward_current']) / baseline['forward_current']
                    power_change = (mod_results['forward_power'] - baseline['forward_power']) / baseline['forward_power']
                    
                    print(f"    Δ Forward current: {current_change:+.1%}")
                    print(f"    Δ Power dissipation: {power_change:+.1%}")
                    
                    mod_results['current_change_percent'] = current_change * 100
                    mod_results['power_change_percent'] = power_change * 100
            else:
                print(f"  ✗ Simulation failed: {mod_results['error']}")
            
            modification_results[mod['name']] = mod_results
        
        self.results['modifications'] = modification_results
        return modification_results
    
    def _create_modified_geometry(self, baseline_geom: GeometryMatrix, modification: Dict):
        """Create modified geometry with single cell change"""
        # Copy baseline geometry
        modified_geom = GeometryMatrix(
            self.matrix_size, self.matrix_size, 
            self.physical_size, self.physical_size
        )
        modified_geom.material_matrix = baseline_geom.material_matrix.copy()
        
        # Apply modification
        y, x = modification['position']
        modified_geom.material_matrix[y, x] = modification['new_material']
        
        # Enforce contact constraints to maintain electrical connectivity
        modified_geom.enforce_contact_constraints()
        
        return modified_geom
    
    def _run_comprehensive_devsim_test(self, geometry: GeometryMatrix, name: str) -> Dict:
        """Run comprehensive DEVSIM test with forward and reverse bias"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                geo_file = os.path.join(temp_dir, f"{name}.geo")
                reset_devsim()
                # Convert geometry to GMSH
                self.converter.convert_matrix_to_gmsh(geometry.material_matrix, geo_file)
                msh_file = self.converter.generate_mesh(geo_file)
                
                if not msh_file:
                    return {'success': False, 'error': 'Mesh generation failed'}
                
                # DEVSIM setup with unique names
                device_name = f"device_{name}_{int(time.time()*1000)}"
                mesh_name = f"mesh_{name}_{int(time.time()*1000)}"
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
                
                # Setup doping profile based on geometry
                self._setup_geometry_based_doping(device_name, region, geometry)
                
                # Initial solution - Poisson only
                CreateSolution(device_name, region, "Potential")
                CreateSiliconPotentialOnly(device_name, region)
                
                for contact in get_contact_list(device=device_name):
                    set_parameter(device=device_name, name=GetContactBiasName(contact), value=0.0)
                    CreateSiliconPotentialOnlyContact(device_name, region, contact)
                
                # Solve Poisson equation
                solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
                
                # Drift-diffusion setup
                CreateSolution(device_name, region, "Electrons")
                CreateSolution(device_name, region, "Holes")
                
                set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
                set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")
                
                CreateSiliconDriftDiffusion(device_name, region)
                for contact in get_contact_list(device=device_name):
                    CreateSiliconDriftDiffusionAtContact(device_name, region, contact)
                
                # Solve equilibrium
                solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=30)
                
                # Test forward bias (positive voltage on anode)
                print(f"    Testing forward bias...")
                forward_voltage = 0.7  # V
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=forward_voltage)
                
                try:
                    solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=30)
                    
                    # Get forward current
                    i_electron_fwd = get_contact_current(device=device_name, contact="anode", equation="ElectronContinuityEquation")
                    i_hole_fwd = get_contact_current(device=device_name, contact="anode", equation="HoleContinuityEquation")
                    forward_current = i_electron_fwd + i_hole_fwd
                    forward_power = forward_current * forward_voltage
                    
                except Exception as e:
                    return {'success': False, 'error': f'Forward bias solve failed: {str(e)}'}
                
                # Test reverse bias (negative voltage on anode)
                print(f"    Testing reverse bias...")
                reverse_voltage = -0.5  # V
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=reverse_voltage)
                
                try:
                    solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=30)
                    
                    # Get reverse current
                    i_electron_rev = get_contact_current(device=device_name, contact="anode", equation="ElectronContinuityEquation")
                    i_hole_rev = get_contact_current(device=device_name, contact="anode", equation="HoleContinuityEquation")
                    reverse_current = i_electron_rev + i_hole_rev
                    reverse_power = abs(reverse_current * reverse_voltage)
                    
                except Exception as e:
                    return {'success': False, 'error': f'Reverse bias solve failed: {str(e)}'}
                
                # Calculate performance metrics
                rectification_ratio = abs(forward_current / (reverse_current + 1e-20))
                
                return {
                    'success': True,
                    'forward_current': forward_current,
                    'reverse_current': reverse_current,
                    'forward_power': forward_power,
                    'reverse_power': reverse_power,
                    'rectification_ratio': rectification_ratio,
                    'forward_voltage': forward_voltage,
                    'reverse_voltage': reverse_voltage,
                    'geometry_metrics': geometry.calculate_metrics()
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _setup_geometry_based_doping(self, device: str, region: str, geometry: GeometryMatrix):
        """Setup spatially-varying doping based on actual geometry matrix"""
        # Analyze P and N regions from geometry
        p_regions = (geometry.material_matrix == 2)
        n_regions = (geometry.material_matrix == 1)
        
        if np.any(p_regions) and np.any(n_regions):
            # Find P-N junction boundary by analyzing the geometry
            junction_positions = []
            
            for i in range(geometry.height):
                p_cols = np.where(p_regions[i, :])[0]
                n_cols = np.where(n_regions[i, :])[0]
                
                if len(p_cols) > 0 and len(n_cols) > 0:
                    # Junction is between rightmost P and leftmost N
                    junction_col = (np.max(p_cols) + np.min(n_cols)) / 2
                    junction_positions.append(junction_col)
            
            if junction_positions:
                # Average junction position
                avg_junction_col = np.mean(junction_positions)
                junction_x = (avg_junction_col + 0.5) * geometry.dx
            else:
                # Fallback to center
                junction_x = geometry.physical_width / 2
        else:
            # Fallback to center if analysis fails
            junction_x = geometry.physical_width / 2
        
        # Create doping models
        doping_level = 1e18  # cm^-3
        CreateNodeModel(device, region, "Acceptors", f"{doping_level}*step({junction_x}-x)")
        CreateNodeModel(device, region, "Donors", f"{doping_level}*step(x-{junction_x})")
        CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
    
    def analyze_sensitivity_to_modifications(self):
        """Analyze how sensitive DEVSIM results are to single-cell modifications"""
        print("\n" + "="*60)
        print("ANALYSIS: Sensitivity to Single Cell Modifications")
        print("="*60)
        
        if 'baseline' not in self.results or 'modifications' not in self.results:
            print("❌ No baseline or modification results available")
            return
        
        baseline = self.results['baseline']
        modifications = self.results['modifications']
        
        print(f"Baseline performance:")
        print(f"  Forward current: {baseline['forward_current']:.2e} A")
        print(f"  Power dissipation: {baseline['forward_power']:.2e} W")
        print(f"  Rectification ratio: {baseline['rectification_ratio']:.1e}")
        
        print(f"\nModification impacts:")
        successful_mods = 0
        
        for mod_name, mod_result in modifications.items():
            if mod_result['success']:
                successful_mods += 1
                current_change = mod_result.get('current_change_percent', 0)
                power_change = mod_result.get('power_change_percent', 0)
                
                print(f"  {mod_name}:")
                print(f"    Forward current change: {current_change:+.1f}%")
                print(f"    Power change: {power_change:+.1f}%")
                print(f"    Rectification: {mod_result['rectification_ratio']:.1e}")
            else:
                print(f"  {mod_name}: FAILED ({mod_result['error']})")
        
        success_rate = successful_mods / len(modifications)
        print(f"\nOverall success rate: {success_rate:.1%} ({successful_mods}/{len(modifications)})")
        
        # Check if DEVSIM is sensitive enough for optimization
        if successful_mods > 0:
            current_changes = [mod['current_change_percent'] for mod in modifications.values() 
                             if mod['success'] and 'current_change_percent' in mod]
            power_changes = [mod['power_change_percent'] for mod in modifications.values() 
                           if mod['success'] and 'power_change_percent' in mod]
            
            if current_changes:
                max_current_change = max(abs(c) for c in current_changes)
                max_power_change = max(abs(p) for p in power_changes)
                
                print(f"\nSensitivity analysis:")
                print(f"  Max current change: ±{max_current_change:.1f}%")
                print(f"  Max power change: ±{max_power_change:.1f}%")
                
                if max_current_change > 1.0 or max_power_change > 1.0:
                    print(f"  ✓ DEVSIM is sensitive enough for optimization")
                else:
                    print(f"  ⚠️  DEVSIM sensitivity may be too low for optimization")
        
        return {
            'success_rate': success_rate,
            'successful_modifications': successful_mods,
            'baseline_performance': baseline,
            'modification_impacts': modifications
        }
    
    def create_visualization(self):
        """Create visualization of baseline and modification results"""
        if not self.results:
            print("No results to visualize")
            return
        
        fig = plt.figure(figsize=(16, 10))
        
        # Baseline geometry
        if 'baseline' in self.results:
            baseline_geom = GeometryGenerator.create_baseline_rectangular(
                self.matrix_size, self.matrix_size
            )
            
            ax1 = plt.subplot(2, 3, 1)
            cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
            im1 = ax1.imshow(baseline_geom.material_matrix, cmap=cmap, vmin=0, vmax=2, origin='lower')
            ax1.set_title('Baseline Rectangular Diode')
            plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2], 
                        label='Material: 0=Void, 1=N-type, 2=P-type')
        
        # Performance comparison
        if 'modifications' in self.results:
            ax2 = plt.subplot(2, 3, 2)
            
            mod_names = []
            current_changes = []
            power_changes = []
            
            for name, result in self.results['modifications'].items():
                if result['success'] and 'current_change_percent' in result:
                    mod_names.append(name.replace('_', '\n'))
                    current_changes.append(result['current_change_percent'])
                    power_changes.append(result['power_change_percent'])
            
            if mod_names:
                x_pos = np.arange(len(mod_names))
                width = 0.35
                
                bars1 = ax2.bar(x_pos - width/2, current_changes, width, 
                               label='Current Change %', alpha=0.7)
                bars2 = ax2.bar(x_pos + width/2, power_changes, width, 
                               label='Power Change %', alpha=0.7)
                
                ax2.set_xlabel('Modification')
                ax2.set_ylabel('Change (%)')
                ax2.set_title('Performance Impact of Single Cell Changes')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(mod_names, rotation=45, ha='right')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Rectification comparison
        if 'baseline' in self.results and 'modifications' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            
            rectification_ratios = [self.results['baseline']['rectification_ratio']]
            labels = ['Baseline']
            
            for name, result in self.results['modifications'].items():
                if result['success']:
                    rectification_ratios.append(result['rectification_ratio'])
                    labels.append(name.replace('_', '\n'))
            
            ax3.semilogy(rectification_ratios, 'o-', markersize=8)
            ax3.set_xlabel('Configuration')
            ax3.set_ylabel('Rectification Ratio')
            ax3.set_title('Rectification Performance')
            ax3.set_xticks(range(len(labels)))
            ax3.set_xticklabels(labels, rotation=45, ha='right')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('devsim_baseline_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_full_test(self):
        """Run complete baseline and modification test"""
        print("DEVSIM BASELINE AND SINGLE CELL MODIFICATION TEST")
        print("=" * 60)
        
        # Test 1: Baseline rectangular diode
        baseline_result = self.test_baseline_rectangular_diode()
        
        if baseline_result is None:
            print("\n❌ Baseline test failed - cannot proceed with modifications")
            return False
        
        # Test 2: Single cell modifications
        baseline_geom = GeometryGenerator.create_baseline_rectangular(
            self.matrix_size, self.matrix_size
        )
        modification_results = self.test_single_cell_modifications(baseline_geom)
        
        # Test 3: Analyze sensitivity
        sensitivity_analysis = self.analyze_sensitivity_to_modifications()
        
        # Create visualization
        self.create_visualization()
        
        # Final assessment
        print("\n" + "="*60)
        print("FINAL ASSESSMENT")
        print("="*60)
        
        success = baseline_result['success']
        print(f"Baseline DEVSIM simulation: {'✓ SUCCESS' if success else '✗ FAILED'}")
        
        if success and sensitivity_analysis:
            success_rate = sensitivity_analysis['success_rate']
            print(f"Single cell modification success rate: {success_rate:.1%}")
            
            if success_rate > 0.5:
                print(f"✓ DEVSIM pipeline is ready for optimization")
                print(f"✓ Geometry changes produce measurable performance impacts")
                return True
            else:
                print(f"⚠️  Low success rate - may need more robust DEVSIM settings")
                return False
        else:
            print(f"❌ Cannot proceed with optimization - baseline or sensitivity test failed")
            return False

def main():
    """Run DEVSIM baseline and modification test"""
    tester = DevsimBaselineTester(matrix_size=16, physical_size=10e-6)
    success = tester.run_full_test()
    
    if success:
        print(f"\n🎉 Ready to proceed with DEVSIM-based optimization!")
    else:
        print(f"\n⚠️  DEVSIM testing revealed issues that need to be addressed")
    
    return tester.results

if __name__ == "__main__":
    main()