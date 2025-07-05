#!/usr/bin/env python3
"""
Simple RL Environment for Topology Optimization
Basic optimization loop with matrix-based geometries and robust DEVSIM simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import tempfile
import os
from typing import Dict, List, Tuple, Optional
import time

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

class TopologyOptimizationEnvironment:
    """
    Simple RL environment for topology optimization of semiconductor devices
    """
    
    def __init__(self, matrix_size: int = 16, physical_size: float = 20e-6):
        self.matrix_size = matrix_size
        self.physical_size = physical_size
        self.converter = SimpleMatrixToGMSH(physical_size, physical_size)
        
        # Optimization parameters
        self.max_iterations = 50
        self.convergence_timeout = 30  # seconds
        
        # Current state
        self.current_geometry = None
        self.current_performance = -1e15
        self.iteration_count = 0
        self.optimization_history = []
        
        # Statistics
        self.success_count = 0
        self.failure_count = 0
        
    def reset(self) -> np.ndarray:
        """Reset environment with initial geometry"""
        # Start with rectangular geometry
        self.current_geometry = GeometryGenerator.create_baseline_rectangular(
            self.matrix_size, self.matrix_size
        )
        self.current_performance = self._evaluate_geometry(self.current_geometry)
        self.iteration_count = 0
        self.optimization_history = []
        
        return self.current_geometry.material_matrix.copy()
    
    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Apply action and return new state, reward, done, info
        
        Args:
            action: Dictionary with 'type' and parameters
                    - 'random_mutation': {'strength': float}
                    - 'local_modification': {'x': int, 'y': int, 'radius': int, 'material': int}
                    - 'predefined_geometry': {'geometry_type': str}
        """
        info = {'action_type': action.get('type', 'unknown')}
        
        # Apply action to create new geometry
        new_geometry = self._apply_action(action)
        
        # Evaluate new geometry
        new_performance = self._evaluate_geometry(new_geometry)
        
        # Calculate reward (improvement in performance)
        reward = new_performance - self.current_performance
        
        # Accept if better (greedy for now)
        if new_performance > self.current_performance:
            self.current_geometry = new_geometry
            self.current_performance = new_performance
            info['accepted'] = True
        else:
            info['accepted'] = False
        
        # Update iteration count
        self.iteration_count += 1
        
        # Record history
        self.optimization_history.append({
            'iteration': self.iteration_count,
            'performance': self.current_performance,
            'reward': reward,
            'action': action,
            'accepted': info['accepted']
        })
        
        # Check if done
        done = self.iteration_count >= self.max_iterations
        
        info['performance'] = self.current_performance
        info['iteration'] = self.iteration_count
        
        return self.current_geometry.material_matrix.copy(), reward, done, info
    
    def _apply_action(self, action: Dict) -> GeometryMatrix:
        """Apply action to current geometry"""
        action_type = action.get('type', 'random_mutation')
        
        if action_type == 'random_mutation':
            return self._random_mutation(action.get('strength', 0.1))
        elif action_type == 'local_modification':
            return self._local_modification(action)
        elif action_type == 'predefined_geometry':
            return self._predefined_geometry(action.get('geometry_type', 'circular'))
        else:
            # Default: small random mutation
            return self._random_mutation(0.05)
    
    def _random_mutation(self, strength: float) -> GeometryMatrix:
        """Apply random mutation to current geometry"""
        new_geom = GeometryMatrix(
            self.matrix_size, self.matrix_size, 
            self.physical_size, self.physical_size
        )
        
        # Copy current geometry
        new_geom.material_matrix = self.current_geometry.material_matrix.copy()
        
        # Apply random mutations
        modifiable_mask = new_geom.get_modifiable_mask()
        modifiable_indices = np.where(modifiable_mask)
        
        num_modifications = max(1, int(strength * len(modifiable_indices[0])))
        
        for _ in range(num_modifications):
            # Random position in modifiable region
            idx = random.randint(0, len(modifiable_indices[0]) - 1)
            y, x = modifiable_indices[0][idx], modifiable_indices[1][idx]
            
            # Random material change
            current_material = new_geom.material_matrix[y, x]
            new_material = random.choice([0, 1, 2])  # Void, N-type, P-type
            
            if new_material != current_material:
                new_geom.material_matrix[y, x] = new_material
        
        # Enforce contact constraints
        new_geom.enforce_contact_constraints()
        
        return new_geom
    
    def _local_modification(self, action: Dict) -> GeometryMatrix:
        """Apply local circular modification"""
        new_geom = GeometryMatrix(
            self.matrix_size, self.matrix_size,
            self.physical_size, self.physical_size
        )
        
        # Copy current geometry
        new_geom.material_matrix = self.current_geometry.material_matrix.copy()
        
        # Apply modification
        x = action.get('x', self.matrix_size // 2)
        y = action.get('y', self.matrix_size // 2)
        radius = action.get('radius', 2)
        material = action.get('material', 1)
        
        new_geom.apply_modification(x, y, radius, material)
        
        return new_geom
    
    def _predefined_geometry(self, geometry_type: str) -> GeometryMatrix:
        """Generate predefined geometry"""
        if geometry_type == 'circular':
            return GeometryGenerator.create_circular_junction(
                self.matrix_size, self.matrix_size, radius_ratio=0.3
            )
        elif geometry_type == 'interdigitated':
            return GeometryGenerator.create_interdigitated(
                self.matrix_size, self.matrix_size, finger_width=2
            )
        elif geometry_type == 'honeycomb':
            return GeometryGenerator.create_honeycomb_pattern(
                self.matrix_size, self.matrix_size, cell_size=4
            )
        else:
            return GeometryGenerator.create_baseline_rectangular(
                self.matrix_size, self.matrix_size
            )
    
    def _evaluate_geometry(self, geometry: GeometryMatrix) -> float:
        """
        Evaluate geometry performance using DEVSIM simulation
        Returns large negative value for failed simulations
        """
        try:
            # Quick geometry validity check
            metrics = geometry.calculate_metrics()
            
            # Must have both P and N regions
            if metrics['p_fraction'] == 0 or metrics['n_fraction'] == 0:
                return -1e15
            
            # Must have reasonable connectivity
            if metrics['connectivity_score'] <= 0:
                return -1e15
            
            # Run DEVSIM simulation with timeout
            performance = self._run_devsim_simulation_with_timeout(geometry)
            
            if performance is not None:
                self.success_count += 1
                return performance
            else:
                self.failure_count += 1
                return -1e15
                
        except Exception as e:
            self.failure_count += 1
            return -1e15
    
    def _run_devsim_simulation_with_timeout(self, geometry: GeometryMatrix) -> Optional[float]:
        """Run DEVSIM simulation with timeout protection"""
        start_time = time.time()
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                geo_file = os.path.join(temp_dir, "optim_device.geo")
                
                # Convert geometry to GMSH
                self.converter.convert_matrix_to_gmsh(geometry.material_matrix, geo_file)
                msh_file = self.converter.generate_mesh(geo_file)
                
                if not msh_file:
                    return None
                
                # Check timeout
                if time.time() - start_time > self.convergence_timeout:
                    return None
                
                # DEVSIM setup with unique names
                device_name = f"device_{self.iteration_count}_{random.randint(1000, 9999)}"
                mesh_name = f"mesh_{self.iteration_count}_{random.randint(1000, 9999)}"
                region = "Bulk"
                
                # Load mesh
                create_gmsh_mesh(mesh=mesh_name, file=msh_file)
                add_gmsh_region(mesh=mesh_name, gmsh_name="Bulk", region=region, material="Silicon")
                add_gmsh_contact(mesh=mesh_name, gmsh_name="P_contact", region=region, material="metal", name="base")
                add_gmsh_contact(mesh=mesh_name, gmsh_name="N_contact", region=region, material="metal", name="emitter")
                finalize_mesh(mesh=mesh_name)
                create_device(mesh=mesh_name, device=device_name)
                
                # Physics setup
                SetSiliconParameters(device_name, region, 300)
                
                # Simple doping profile
                junction_pos = geometry.physical_width / 2
                CreateNodeModel(device_name, region, "Acceptors", f"1.0e18*step({junction_pos}-x)")
                CreateNodeModel(device_name, region, "Donors", f"1.0e18*step(x-{junction_pos})")
                CreateNodeModel(device_name, region, "NetDoping", "Donors-Acceptors")
                
                # Initial solution
                CreateSolution(device_name, region, "Potential")
                CreateSiliconPotentialOnly(device_name, region)
                
                for contact in get_contact_list(device=device_name):
                    set_parameter(device=device_name, name=GetContactBiasName(contact), value=0.0)
                    CreateSiliconPotentialOnlyContact(device_name, region, contact)
                
                # Check timeout
                if time.time() - start_time > self.convergence_timeout:
                    return None
                
                # Solve Poisson with limited iterations
                solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=15)
                
                # Drift-diffusion setup
                CreateSolution(device_name, region, "Electrons")
                CreateSolution(device_name, region, "Holes")
                
                set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
                set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")
                
                CreateSiliconDriftDiffusion(device_name, region)
                for contact in get_contact_list(device=device_name):
                    CreateSiliconDriftDiffusionAtContact(device_name, region, contact)
                
                # Check timeout
                if time.time() - start_time > self.convergence_timeout:
                    return None
                
                # Solve equilibrium with limited iterations
                solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=15)
                
                # Quick performance test at 0.5V
                set_parameter(device=device_name, name=GetContactBiasName("base"), value=0.5)
                
                # Check timeout
                if time.time() - start_time > self.convergence_timeout:
                    return None
                
                solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=10)
                
                # Get current
                i_electron = get_contact_current(device=device_name, contact="base", equation="ElectronContinuityEquation")
                i_hole = get_contact_current(device=device_name, contact="base", equation="HoleContinuityEquation")
                i_total = i_electron + i_hole
                
                # Include geometry metrics in performance
                interface_length = geometry.calculate_metrics()['interface_length']
                performance = i_total * interface_length * 1e6  # Scale factor
                
                return performance
                
        except Exception as e:
            return None
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics"""
        return {
            'iteration': self.iteration_count,
            'success_rate': self.success_count / max(1, self.success_count + self.failure_count),
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'current_performance': self.current_performance,
            'best_performance': max([h['performance'] for h in self.optimization_history], default=-1e15),
            'history_length': len(self.optimization_history)
        }
    
    def plot_optimization_history(self):
        """Plot optimization progress"""
        if not self.optimization_history:
            print("No optimization history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        iterations = [h['iteration'] for h in self.optimization_history]
        performances = [h['performance'] for h in self.optimization_history]
        accepted = [h['accepted'] for h in self.optimization_history]
        
        # Performance over time
        ax1.plot(iterations, performances, 'b-', linewidth=2, label='Performance')
        ax1.scatter([i for i, a in zip(iterations, accepted) if a], 
                   [p for p, a in zip(performances, accepted) if a], 
                   color='green', s=50, alpha=0.7, label='Accepted')
        ax1.scatter([i for i, a in zip(iterations, accepted) if not a], 
                   [p for p, a in zip(performances, accepted) if not a], 
                   color='red', s=30, alpha=0.5, label='Rejected')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Performance')
        ax1.set_title('Optimization Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Success rate
        window_size = 10
        success_rates = []
        for i in range(len(accepted)):
            start = max(0, i - window_size)
            window = accepted[start:i+1]
            success_rates.append(sum(window) / len(window))
        
        ax2.plot(iterations, success_rates, 'g-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Acceptance Rate (windowed)')
        ax2.set_title('Acceptance Rate Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('optimization_progress.png', dpi=300, bbox_inches='tight')
        plt.show()

class SimpleOptimizer:
    """Simple optimization algorithms for testing"""
    
    @staticmethod
    def random_search(env: TopologyOptimizationEnvironment, num_iterations: int = 20) -> Dict:
        """Random search optimizer"""
        print(f"Running Random Search Optimization ({num_iterations} iterations)")
        print("=" * 60)
        
        state = env.reset()
        best_performance = env.current_performance
        best_geometry = env.current_geometry
        
        for i in range(num_iterations):
            # Random action
            action = {
                'type': 'random_mutation',
                'strength': random.uniform(0.05, 0.2)
            }
            
            state, reward, done, info = env.step(action)
            
            if info['accepted']:
                print(f"Iteration {i+1}: ✓ Improved performance: {info['performance']:.2e}")
                if info['performance'] > best_performance:
                    best_performance = info['performance']
                    best_geometry = env.current_geometry
            else:
                print(f"Iteration {i+1}: → No improvement (current: {info['performance']:.2e})")
            
            if done:
                break
        
        stats = env.get_statistics()
        print(f"\nRandom Search Complete:")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Best performance: {best_performance:.2e}")
        
        return {'best_geometry': best_geometry, 'best_performance': best_performance, 'stats': stats}
    
    @staticmethod
    def hill_climbing(env: TopologyOptimizationEnvironment, num_iterations: int = 20) -> Dict:
        """Hill climbing optimizer"""
        print(f"Running Hill Climbing Optimization ({num_iterations} iterations)")
        print("=" * 60)
        
        state = env.reset()
        best_performance = env.current_performance
        best_geometry = env.current_geometry
        
        for i in range(num_iterations):
            # Try different action types
            action_types = ['random_mutation', 'local_modification', 'predefined_geometry']
            
            if i % 10 == 0:  # Every 10 iterations, try predefined geometry
                action = {
                    'type': 'predefined_geometry',
                    'geometry_type': random.choice(['circular', 'interdigitated', 'honeycomb'])
                }
            elif i % 3 == 0:  # Every 3 iterations, try local modification
                action = {
                    'type': 'local_modification',
                    'x': random.randint(4, env.matrix_size - 4),
                    'y': random.randint(4, env.matrix_size - 4),
                    'radius': random.randint(1, 3),
                    'material': random.choice([0, 1, 2])
                }
            else:  # Random mutation
                action = {
                    'type': 'random_mutation',
                    'strength': random.uniform(0.02, 0.1)
                }
            
            state, reward, done, info = env.step(action)
            
            if info['accepted']:
                print(f"Iteration {i+1}: ✓ Improved performance: {info['performance']:.2e} (action: {action['type']})")
                if info['performance'] > best_performance:
                    best_performance = info['performance']
                    best_geometry = env.current_geometry
            else:
                print(f"Iteration {i+1}: → No improvement (action: {action['type']})")
            
            if done:
                break
        
        stats = env.get_statistics()
        print(f"\nHill Climbing Complete:")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Best performance: {best_performance:.2e}")
        
        return {'best_geometry': best_geometry, 'best_performance': best_performance, 'stats': stats}

def test_simple_optimization():
    """Test the simple optimization environment"""
    print("Testing Simple RL Environment for Topology Optimization")
    print("=" * 70)
    
    # Create environment
    env = TopologyOptimizationEnvironment(matrix_size=1024, physical_size=20e-6)
    
    # Test 1: Random Search
    print("\nTEST 1: Random Search")
    print("-" * 30)
    random_results = SimpleOptimizer.random_search(env, num_iterations=10)
    
    # Test 2: Hill Climbing
    print("\nTEST 2: Hill Climbing")
    print("-" * 30)
    hill_results = SimpleOptimizer.hill_climbing(env, num_iterations=10)
    
    # Compare results
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPARISON")
    print("=" * 70)
    print(f"Random Search - Best Performance: {random_results['best_performance']:.2e}")
    print(f"Hill Climbing - Best Performance: {hill_results['best_performance']:.2e}")
    
    # Plot results
    env.plot_optimization_history()
    
    # Visualize best geometry
    if hill_results['best_performance'] > random_results['best_performance']:
        best_result = hill_results
        method = "Hill Climbing"
    else:
        best_result = random_results
        method = "Random Search"
    
    print(f"\nBest result from: {method}")
    fig = best_result['best_geometry'].visualize(f"Best Geometry ({method})")
    plt.savefig('best_optimized_geometry.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nOptimization test complete!")
    print(f"Plots saved: optimization_progress.png, best_optimized_geometry.png")
    
    return best_result

if __name__ == "__main__":
    test_simple_optimization()