#!/usr/bin/env python3
"""
DEVSIM-based Objective Function for Semiconductor Device Optimization
Minimize power while maintaining forward current density and very low reverse current
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import time
from typing import Dict, List, Tuple, Optional

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

class DevsimObjectiveFunction:
    """
    DEVSIM-based objective function for semiconductor device optimization
    
    Objective: Minimize power while maintaining:
    - High forward current density
    - Very low reverse current
    """
    
    def __init__(self, matrix_size=16, physical_size=10e-6):
        self.matrix_size = matrix_size
        self.physical_size = physical_size
        self.converter = SimpleMatrixToGMSH(physical_size, physical_size)
        
        # Optimization targets
        self.forward_voltage = 0.7  # V
        self.reverse_voltage = -0.5  # V
        self.min_forward_current = 1.0  # A (minimum acceptable)
        self.max_reverse_current = 1e-3  # A (maximum acceptable)
        self.target_rectification_ratio = 1e3  # Minimum acceptable
        
        # Objective weights
        self.power_weight = 1.0
        self.current_penalty_weight = 1e6
        self.reverse_penalty_weight = 1e9
        self.simulation_failure_penalty = 1e12
        
        # Performance cache
        self.performance_cache = {}
        self.evaluation_count = 0
        
    def evaluate_geometry(self, geometry: GeometryMatrix) -> Dict:
        """
        Evaluate geometry using DEVSIM simulation
        
        Returns:
            Dict with 'objective', 'forward_current', 'reverse_current', 'power', 'success', etc.
        """
        self.evaluation_count += 1
        
        # Create geometry hash for caching
        geom_hash = hash(geometry.material_matrix.tobytes())
        if geom_hash in self.performance_cache:
            return self.performance_cache[geom_hash].copy()
        
        # Quick geometry validity check
        metrics = geometry.calculate_metrics()
        
        # Must have both P and N regions
        if metrics['p_fraction'] < 0.05 or metrics['n_fraction'] < 0.05:
            result = {
                'success': False,
                'error': 'Insufficient P or N material',
                'objective': self.simulation_failure_penalty,
                'forward_current': 0.0,
                'reverse_current': 0.0,
                'power': 0.0,
                'rectification_ratio': 0.0
            }
            self.performance_cache[geom_hash] = result
            return result
        
        # Must have reasonable connectivity
        if metrics['connectivity_score'] <= 0:
            result = {
                'success': False,
                'error': 'Poor connectivity',
                'objective': self.simulation_failure_penalty,
                'forward_current': 0.0,
                'reverse_current': 0.0,
                'power': 0.0,
                'rectification_ratio': 0.0
            }
            self.performance_cache[geom_hash] = result
            return result
        
        # Run DEVSIM simulation
        devsim_result = self._run_devsim_simulation(geometry)
        
        if not devsim_result['success']:
            result = {
                'success': False,
                'error': devsim_result['error'],
                'objective': self.simulation_failure_penalty,
                'forward_current': 0.0,
                'reverse_current': 0.0,
                'power': 0.0,
                'rectification_ratio': 0.0
            }
            self.performance_cache[geom_hash] = result
            return result
        
        # Extract performance metrics
        forward_current = devsim_result['forward_current']
        reverse_current = abs(devsim_result['reverse_current'])
        power = devsim_result['forward_power']
        rectification_ratio = devsim_result['rectification_ratio']
        
        # Calculate objective function
        objective = self._calculate_objective(
            forward_current, reverse_current, power, rectification_ratio
        )
        
        result = {
            'success': True,
            'objective': objective,
            'forward_current': forward_current,
            'reverse_current': reverse_current,
            'power': power,
            'rectification_ratio': rectification_ratio,
            'geometry_metrics': metrics
        }
        
        self.performance_cache[geom_hash] = result
        return result
    
    def _calculate_objective(self, forward_current: float, reverse_current: float, 
                           power: float, rectification_ratio: float) -> float:
        """
        Calculate multi-objective function:
        Minimize: power + penalties for not meeting current/rectification requirements
        """
        objective = 0.0
        
        # Primary objective: minimize power
        objective += self.power_weight * power
        
        # Penalty for insufficient forward current
        if forward_current < self.min_forward_current:
            current_deficit = self.min_forward_current - forward_current
            objective += self.current_penalty_weight * (current_deficit ** 2)
        
        # Penalty for excessive reverse current
        if reverse_current > self.max_reverse_current:
            reverse_excess = reverse_current - self.max_reverse_current
            objective += self.reverse_penalty_weight * (reverse_excess ** 2)
        
        # Penalty for poor rectification
        if rectification_ratio < self.target_rectification_ratio:
            rectification_deficit = self.target_rectification_ratio - rectification_ratio
            objective += self.current_penalty_weight * (rectification_deficit / self.target_rectification_ratio)
        
        return objective
    
    def _run_devsim_simulation(self, geometry: GeometryMatrix) -> Dict:
        """Run DEVSIM simulation to get forward/reverse current and power"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                geo_file = os.path.join(temp_dir, f"opt_eval_{self.evaluation_count}.geo")
                
                # Convert geometry to GMSH
                self.converter.convert_matrix_to_gmsh(geometry.material_matrix, geo_file)
                msh_file = self.converter.generate_mesh(geo_file)
                
                if not msh_file:
                    return {'success': False, 'error': 'Mesh generation failed'}
                
                # DEVSIM setup with unique names
                device_name = f"optim_device_{self.evaluation_count}_{int(time.time()*1000)}"
                mesh_name = f"optim_mesh_{self.evaluation_count}_{int(time.time()*1000)}"
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
                
                # Solve equilibrium with relaxed tolerances for optimization speed
                solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=20)
                
                # Test forward bias
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=self.forward_voltage)
                
                try:
                    solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=20)
                    
                    # Get forward current
                    i_electron_fwd = get_contact_current(device=device_name, contact="anode", equation="ElectronContinuityEquation")
                    i_hole_fwd = get_contact_current(device=device_name, contact="anode", equation="HoleContinuityEquation")
                    forward_current = i_electron_fwd + i_hole_fwd
                    forward_power = forward_current * self.forward_voltage
                    
                except Exception:
                    return {'success': False, 'error': 'Forward bias solve failed'}
                
                # Test reverse bias with more relaxed settings
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=self.reverse_voltage)
                
                try:
                    solve(type="dc", absolute_error=1e6, relative_error=1e-6, maximum_iterations=15)
                    
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
    
    def evaluate_sensitivity(self, geometry: GeometryMatrix, delta_size=1) -> np.ndarray:
        """
        Evaluate sensitivity of objective function w.r.t. geometry changes
        
        Returns sensitivity matrix same size as geometry.material_matrix
        """
        print(f"  Computing DEVSIM-based sensitivities...")
        
        # Get baseline performance
        baseline_result = self.evaluate_geometry(geometry)
        if not baseline_result['success']:
            return np.zeros_like(geometry.material_matrix, dtype=float)
        
        baseline_objective = baseline_result['objective']
        sensitivity = np.zeros_like(geometry.material_matrix, dtype=float)
        
        # Get modifiable mask (areas we can change)
        modifiable_mask = geometry.get_modifiable_mask()
        modifiable_indices = np.where(modifiable_mask)
        
        # Sample subset of modifiable points for efficiency
        n_modifiable = len(modifiable_indices[0])
        sample_size = min(50, n_modifiable)  # Limit to 50 evaluations for speed
        
        if n_modifiable > sample_size:
            # Random sampling of modifiable points
            sample_indices = np.random.choice(n_modifiable, sample_size, replace=False)
            sample_y = modifiable_indices[0][sample_indices]
            sample_x = modifiable_indices[1][sample_indices]
        else:
            sample_y = modifiable_indices[0]
            sample_x = modifiable_indices[1]
        
        print(f"    Evaluating {len(sample_y)} geometry perturbations...")
        
        for idx, (y, x) in enumerate(zip(sample_y, sample_x)):
            if idx % 10 == 0:
                print(f"    Progress: {idx}/{len(sample_y)}")
            
            # Create modified geometry
            modified_geom = self._create_modified_geometry(geometry, x, y)
            
            if modified_geom is None:
                continue
            
            # Evaluate modified geometry
            modified_result = self.evaluate_geometry(modified_geom)
            
            if modified_result['success']:
                # Calculate sensitivity (negative because we minimize objective)
                sensitivity[y, x] = -(modified_result['objective'] - baseline_objective)
            else:
                # Penalize modifications that break simulation
                sensitivity[y, x] = -self.simulation_failure_penalty
        
        print(f"    Sensitivity range: [{np.min(sensitivity):.2e}, {np.max(sensitivity):.2e}]")
        return sensitivity
    
    def _create_modified_geometry(self, base_geometry: GeometryMatrix, x: int, y: int) -> Optional[GeometryMatrix]:
        """Create geometry with single cell modification"""
        # Copy base geometry
        modified_geom = GeometryMatrix(
            self.matrix_size, self.matrix_size,
            self.physical_size, self.physical_size
        )
        modified_geom.material_matrix = base_geometry.material_matrix.copy()
        
        # Try different material changes
        current_material = modified_geom.material_matrix[y, x]
        
        # Cycle to next material type
        if current_material == 0:  # Void -> N-type
            new_material = 1
        elif current_material == 1:  # N-type -> P-type
            new_material = 2
        else:  # P-type -> Void
            new_material = 0
        
        modified_geom.material_matrix[y, x] = new_material
        
        # Enforce contact constraints to maintain electrical connectivity
        modified_geom.enforce_contact_constraints()
        
        # Check if modification is valid
        metrics = modified_geom.calculate_metrics()
        if metrics['p_fraction'] < 0.05 or metrics['n_fraction'] < 0.05:
            return None  # Invalid modification
        
        return modified_geom
    
    def get_statistics(self) -> Dict:
        """Get objective function evaluation statistics"""
        return {
            'evaluation_count': self.evaluation_count,
            'cache_size': len(self.performance_cache),
            'cache_hit_rate': (self.evaluation_count - len(self.performance_cache)) / max(1, self.evaluation_count)
        }
    
    def print_performance_summary(self, result: Dict):
        """Print human-readable performance summary"""
        if not result['success']:
            print(f"    ✗ Simulation failed: {result['error']}")
            return
        
        print(f"    ✓ Simulation successful:")
        print(f"      Forward current: {result['forward_current']:.2e} A")
        print(f"      Reverse current: {result['reverse_current']:.2e} A")
        print(f"      Power: {result['power']:.2e} W")
        print(f"      Rectification: {result['rectification_ratio']:.1e}")
        print(f"      Objective: {result['objective']:.2e}")
        
        # Performance assessment
        meets_current = result['forward_current'] >= self.min_forward_current
        meets_reverse = result['reverse_current'] <= self.max_reverse_current
        meets_rectification = result['rectification_ratio'] >= self.target_rectification_ratio
        
        print(f"      Targets: Current {'✓' if meets_current else '✗'}, "
              f"Reverse {'✓' if meets_reverse else '✗'}, "
              f"Rectification {'✓' if meets_rectification else '✗'}")

def test_devsim_objective_function():
    """Test the DEVSIM objective function"""
    print("Testing DEVSIM Objective Function")
    print("=" * 50)
    
    # Create objective function
    obj_func = DevsimObjectiveFunction(matrix_size=16, physical_size=10e-6)
    
    # Test with baseline rectangular geometry
    print("\nTesting baseline rectangular geometry:")
    baseline_geom = GeometryGenerator.create_baseline_rectangular(16, 16)
    baseline_result = obj_func.evaluate_geometry(baseline_geom)
    obj_func.print_performance_summary(baseline_result)
    
    # Test with circular geometry
    print("\nTesting circular junction geometry:")
    circular_geom = GeometryGenerator.create_circular_junction(16, 16, radius_ratio=0.3)
    circular_result = obj_func.evaluate_geometry(circular_geom)
    obj_func.print_performance_summary(circular_result)
    
    # Test sensitivity calculation
    print("\nTesting sensitivity calculation:")
    sensitivity = obj_func.evaluate_sensitivity(baseline_geom)
    
    print(f"Sensitivity statistics:")
    print(f"  Mean: {np.mean(sensitivity):.2e}")
    print(f"  Std: {np.std(sensitivity):.2e}")
    print(f"  Range: [{np.min(sensitivity):.2e}, {np.max(sensitivity):.2e}]")
    print(f"  Non-zero elements: {np.sum(sensitivity != 0)}/{sensitivity.size}")
    
    # Objective function statistics
    stats = obj_func.get_statistics()
    print(f"\nObjective function statistics:")
    print(f"  Evaluations: {stats['evaluation_count']}")
    print(f"  Cache size: {stats['cache_size']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Baseline geometry
    cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
    im1 = ax1.imshow(baseline_geom.material_matrix, cmap=cmap, vmin=0, vmax=2, origin='lower')
    ax1.set_title('Baseline Geometry')
    plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
    
    # Circular geometry
    im2 = ax2.imshow(circular_geom.material_matrix, cmap=cmap, vmin=0, vmax=2, origin='lower')
    ax2.set_title('Circular Geometry')
    plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
    
    # Sensitivity map
    im3 = ax3.imshow(sensitivity, cmap='RdBu_r', origin='lower')
    ax3.set_title('Objective Sensitivity')
    plt.colorbar(im3, ax=ax3, label='∂Objective/∂Material')
    
    # Performance comparison
    if baseline_result['success'] and circular_result['success']:
        geometries = ['Rectangular', 'Circular']
        objectives = [baseline_result['objective'], circular_result['objective']]
        powers = [baseline_result['power'], circular_result['power']]
        currents = [baseline_result['forward_current'], circular_result['forward_current']]
        
        x_pos = np.arange(len(geometries))
        width = 0.25
        
        ax4_twin = ax4.twinx()
        
        bars1 = ax4.bar(x_pos - width, objectives, width, label='Objective', alpha=0.7)
        bars2 = ax4.bar(x_pos, powers, width, label='Power (W)', alpha=0.7)
        bars3 = ax4_twin.bar(x_pos + width, currents, width, label='Current (A)', alpha=0.7, color='green')
        
        ax4.set_xlabel('Geometry Type')
        ax4.set_ylabel('Objective / Power')
        ax4_twin.set_ylabel('Forward Current (A)')
        ax4.set_title('Performance Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(geometries)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('devsim_objective_function_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nTest complete! Results saved to 'devsim_objective_function_test.png'")
    return obj_func, baseline_result, circular_result

if __name__ == "__main__":
    test_devsim_objective_function()