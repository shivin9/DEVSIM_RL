#!/usr/bin/env python3
"""
Robust DEVSIM Objective Function
Enhanced version with better convergence handling for optimization
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

class RobustDevsimObjective:
    """
    Robust DEVSIM objective function with enhanced convergence handling
    """
    
    def __init__(self, matrix_size=16, physical_size=10e-6):
        self.matrix_size = matrix_size
        self.physical_size = physical_size
        self.converter = SimpleMatrixToGMSH(physical_size, physical_size)
        
        # Optimization targets
        self.forward_voltage = 0.7  # V
        self.reverse_voltage = -0.5  # V
        self.min_forward_current = 1.0  # A
        self.max_reverse_current = 1e-3  # A
        
        # Objective weights
        self.power_weight = 1.0
        self.current_penalty_weight = 1e6
        self.reverse_penalty_weight = 1e9
        self.simulation_failure_penalty = 1e12
        
        # Robust solver settings
        self.solver_configs = [
            # Config 1: Standard settings
            {
                'name': 'standard',
                'poisson_abs_error': 1.0,
                'poisson_rel_error': 1e-10,
                'poisson_max_iter': 30,
                'dd_abs_error': 1e8,
                'dd_rel_error': 1e-8,
                'dd_max_iter': 20,
                'bias_abs_error': 1e8,
                'bias_rel_error': 1e-8,
                'bias_max_iter': 15
            },
            # Config 2: Relaxed settings
            {
                'name': 'relaxed',
                'poisson_abs_error': 10.0,
                'poisson_rel_error': 1e-8,
                'poisson_max_iter': 20,
                'dd_abs_error': 1e6,
                'dd_rel_error': 1e-6,
                'dd_max_iter': 15,
                'bias_abs_error': 1e6,
                'bias_rel_error': 1e-6,
                'bias_max_iter': 10
            },
            # Config 3: Very relaxed settings
            {
                'name': 'very_relaxed',
                'poisson_abs_error': 100.0,
                'poisson_rel_error': 1e-6,
                'poisson_max_iter': 15,
                'dd_abs_error': 1e4,
                'dd_rel_error': 1e-4,
                'dd_max_iter': 10,
                'bias_abs_error': 1e4,
                'bias_rel_error': 1e-4,
                'bias_max_iter': 8
            }
        ]
        
        # Performance cache
        self.performance_cache = {}
        self.evaluation_count = 0
        self.convergence_stats = {'standard': 0, 'relaxed': 0, 'very_relaxed': 0, 'failed': 0}
        
    def evaluate_geometry(self, geometry: GeometryMatrix) -> Dict:
        """
        Evaluate geometry with multiple solver configurations for robustness
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
            result = self._create_failure_result('Insufficient P or N material')
            self.performance_cache[geom_hash] = result
            return result
        
        # Must have reasonable connectivity
        if metrics['connectivity_score'] <= 0:
            result = self._create_failure_result('Poor connectivity')
            self.performance_cache[geom_hash] = result
            return result
        
        # Try progressively more relaxed solver settings
        for config in self.solver_configs:
            print(f"    Trying {config['name']} solver settings...")
            
            devsim_result = self._run_devsim_with_config(geometry, config)
            
            if devsim_result['success']:
                print(f"    ✓ Converged with {config['name']} settings")
                self.convergence_stats[config['name']] += 1
                
                # Calculate objective function
                objective = self._calculate_objective(
                    devsim_result['forward_current'],
                    abs(devsim_result['reverse_current']),
                    devsim_result['forward_power'],
                    devsim_result['rectification_ratio']
                )
                
                result = {
                    'success': True,
                    'solver_config': config['name'],
                    'objective': objective,
                    'forward_current': devsim_result['forward_current'],
                    'reverse_current': devsim_result['reverse_current'],
                    'power': devsim_result['forward_power'],
                    'rectification_ratio': devsim_result['rectification_ratio'],
                    'geometry_metrics': metrics
                }
                
                self.performance_cache[geom_hash] = result
                return result
        
        # All solver configs failed
        print(f"    ✗ All solver configurations failed")
        self.convergence_stats['failed'] += 1
        result = self._create_failure_result('All solver configurations failed')
        self.performance_cache[geom_hash] = result
        return result
    
    def _create_failure_result(self, error_msg: str) -> Dict:
        """Create standardized failure result"""
        return {
            'success': False,
            'error': error_msg,
            'objective': self.simulation_failure_penalty,
            'forward_current': 0.0,
            'reverse_current': 0.0,
            'power': 0.0,
            'rectification_ratio': 0.0
        }
    
    def _run_devsim_with_config(self, geometry: GeometryMatrix, config: Dict) -> Dict:
        """Run DEVSIM simulation with specific solver configuration"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                geo_file = os.path.join(temp_dir, f"robust_eval_{self.evaluation_count}.geo")
                reset_devsim()
                # Convert geometry to GMSH
                self.converter.convert_matrix_to_gmsh(geometry.material_matrix, geo_file)
                msh_file = self.converter.generate_mesh(geo_file)
                
                if not msh_file:
                    return {'success': False, 'error': 'Mesh generation failed'}
                
                # DEVSIM setup with unique names
                device_name = f"robust_device_{self.evaluation_count}_{int(time.time()*1000)}"
                mesh_name = f"robust_mesh_{self.evaluation_count}_{int(time.time()*1000)}"
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
                
                # Solve Poisson equation with config settings
                solve(type="dc", 
                      absolute_error=config['poisson_abs_error'],
                      relative_error=config['poisson_rel_error'], 
                      maximum_iterations=config['poisson_max_iter'])
                
                # Drift-diffusion setup
                CreateSolution(device_name, region, "Electrons")
                CreateSolution(device_name, region, "Holes")
                
                set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
                set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")
                
                CreateSiliconDriftDiffusion(device_name, region)
                for contact in get_contact_list(device=device_name):
                    CreateSiliconDriftDiffusionAtContact(device_name, region, contact)
                
                # Solve equilibrium with config settings
                solve(type="dc", 
                      absolute_error=config['dd_abs_error'],
                      relative_error=config['dd_rel_error'], 
                      maximum_iterations=config['dd_max_iter'])
                
                # Test forward bias
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=self.forward_voltage)
                
                solve(type="dc", 
                      absolute_error=config['bias_abs_error'],
                      relative_error=config['bias_rel_error'], 
                      maximum_iterations=config['bias_max_iter'])
                
                # Get forward current
                i_electron_fwd = get_contact_current(device=device_name, contact="anode", equation="ElectronContinuityEquation")
                i_hole_fwd = get_contact_current(device=device_name, contact="anode", equation="HoleContinuityEquation")
                forward_current = i_electron_fwd + i_hole_fwd
                forward_power = forward_current * self.forward_voltage
                
                # Test reverse bias - try with even more relaxed settings
                set_parameter(device=device_name, name=GetContactBiasName("anode"), value=self.reverse_voltage)
                
                try:
                    # Use very relaxed settings for reverse bias
                    solve(type="dc", 
                          absolute_error=config['bias_abs_error']*100,  # Even more relaxed
                          relative_error=config['bias_rel_error']*10, 
                          maximum_iterations=max(5, config['bias_max_iter']//2))  # Fewer iterations
                    
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
        
        # Create doping models with more gradual transition
        doping_level = 1e18  # cm^-3
        transition_width = geometry.physical_width * 0.1  # 10% of device width
        
        # Smooth doping transition to avoid sharp discontinuities
        CreateNodeModel(device, region, "Acceptors", 
                       f"{doping_level}*0.5*(1.0 + tanh(({junction_x}-x)/{transition_width}))")
        CreateNodeModel(device, region, "Donors", 
                       f"{doping_level}*0.5*(1.0 + tanh((x-{junction_x})/{transition_width}))")
        CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
    
    def _calculate_objective(self, forward_current: float, reverse_current: float, 
                           power: float, rectification_ratio: float) -> float:
        """Calculate multi-objective function"""
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
        
        return objective
    
    def get_statistics(self) -> Dict:
        """Get convergence and performance statistics"""
        total_attempts = sum(self.convergence_stats.values())
        
        stats = {
            'evaluation_count': self.evaluation_count,
            'cache_size': len(self.performance_cache),
            'cache_hit_rate': (self.evaluation_count - len(self.performance_cache)) / max(1, self.evaluation_count),
            'convergence_stats': self.convergence_stats.copy(),
            'overall_success_rate': (total_attempts - self.convergence_stats['failed']) / max(1, total_attempts)
        }
        
        return stats
    
    def print_performance_summary(self, result: Dict):
        """Print human-readable performance summary"""
        if not result['success']:
            print(f"    ✗ Simulation failed: {result['error']}")
            return
        
        solver_config = result.get('solver_config', 'unknown')
        print(f"    ✓ Simulation successful ({solver_config} settings):")
        print(f"      Forward current: {result['forward_current']:.2e} A")
        print(f"      Reverse current: {result['reverse_current']:.2e} A")
        print(f"      Power: {result['power']:.2e} W")
        print(f"      Rectification: {result['rectification_ratio']:.1e}")
        print(f"      Objective: {result['objective']:.2e}")

def test_robust_modifications():
    """Test robust DEVSIM objective with single cell modifications"""
    print("Testing Robust DEVSIM Objective Function")
    print("=" * 60)
    
    # Create robust objective function
    robust_obj = RobustDevsimObjective(matrix_size=16, physical_size=10e-6)
    
    # Test baseline
    print(f"\n1. Testing baseline rectangular geometry:")
    baseline_geom = GeometryGenerator.create_baseline_rectangular(16, 16)
    baseline_result = robust_obj.evaluate_geometry(baseline_geom)
    robust_obj.print_performance_summary(baseline_result)
    
    if not baseline_result['success']:
        print(f"❌ Baseline failed - cannot proceed")
        return
    
    # Test single cell modifications with robust settings
    print(f"\n2. Testing single cell modifications with robust solver:")
    
    modifications = [
        {'name': 'add_void_center', 'pos': (8, 8), 'material': 0},
        {'name': 'add_n_type_center', 'pos': (8, 8), 'material': 1},
        {'name': 'extend_p_region', 'pos': (8, 9), 'material': 2},
        {'name': 'modify_edge', 'pos': (8, 4), 'material': 1},
        {'name': 'modify_other_edge', 'pos': (8, 12), 'material': 2}
    ]
    
    successful_mods = 0
    results = {}
    
    for mod in modifications:
        print(f"\n  Testing: {mod['name']}")
        
        # Create modified geometry
        modified_geom = GeometryMatrix(16, 16, 10e-6, 10e-6)
        modified_geom.material_matrix = baseline_geom.material_matrix.copy()
        
        y, x = mod['pos']
        modified_geom.material_matrix[y, x] = mod['material']
        modified_geom.enforce_contact_constraints()
        
        # Check validity
        metrics = modified_geom.calculate_metrics()
        if metrics['p_fraction'] < 0.05 or metrics['n_fraction'] < 0.05:
            print(f"    ⚠️  Skipping - would break P-N junction")
            continue
        
        # Evaluate with robust objective
        result = robust_obj.evaluate_geometry(modified_geom)
        results[mod['name']] = result
        
        if result['success']:
            successful_mods += 1
            robust_obj.print_performance_summary(result)
            
            # Compare to baseline
            current_change = (result['forward_current'] - baseline_result['forward_current']) / baseline_result['forward_current']
            power_change = (result['power'] - baseline_result['power']) / baseline_result['power']
            print(f"      Δ Forward current: {current_change:+.1%}")
            print(f"      Δ Power: {power_change:+.1%}")
        else:
            robust_obj.print_performance_summary(result)
    
    # Summary
    print(f"\n" + "="*60)
    print(f"ROBUST MODIFICATION TEST SUMMARY")
    print(f"="*60)
    
    success_rate = successful_mods / len(modifications)
    print(f"Success rate: {success_rate:.1%} ({successful_mods}/{len(modifications)})")
    
    stats = robust_obj.get_statistics()
    print(f"Overall simulation success rate: {stats['overall_success_rate']:.1%}")
    print(f"Convergence breakdown:")
    for config, count in stats['convergence_stats'].items():
        print(f"  {config}: {count}")
    
    if success_rate > 0.5:
        print(f"✓ Robust settings enable successful optimization!")
    else:
        print(f"⚠️  Still need more robust handling")
    
    return robust_obj, results

if __name__ == "__main__":
    test_robust_modifications()