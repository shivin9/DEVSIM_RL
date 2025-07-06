#!/usr/bin/env python3
"""
DEVSIM-SIMP Integrated Optimization
Combine SIMP topology optimization with DEVSIM-based objective function
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

from geometry_optimization_framework import GeometryMatrix, GeometryGenerator
from devsim_objective_function import DevsimObjectiveFunction

class DevsimSIMPOptimizer:
    """
    SIMP topology optimization with DEVSIM-based objective function
    
    Instead of using simplified electrical analysis, use full DEVSIM simulation
    to evaluate power, forward current, and reverse current
    """
    
    def __init__(self, nelx=16, nely=16, physical_size=10e-6, volfrac=0.5):
        self.nelx = nelx
        self.nely = nely
        self.physical_size = physical_size
        self.volfrac = volfrac
        
        # DEVSIM objective function
        self.devsim_obj = DevsimObjectiveFunction(
            matrix_size=max(nelx, nely), 
            physical_size=physical_size
        )
        
        # Optimization parameters
        self.move_limit = 0.1  # Conservative move limit
        self.max_iterations = 20  # Limited iterations due to DEVSIM cost
        
        # History tracking
        self.objective_history = []
        self.performance_history = []
        self.geometry_history = []
        
        # Current geometry
        self.current_geometry = None
        self.current_performance = None
        
    def optimize(self, initial_geometry: GeometryMatrix = None) -> GeometryMatrix:
        """
        Run DEVSIM-SIMP optimization
        
        Args:
            initial_geometry: Starting geometry (default: rectangular)
        
        Returns:
            Optimized geometry
        """
        print(f"Starting DEVSIM-SIMP Optimization")
        print(f"Design domain: {self.nelx} × {self.nely} elements")
        print(f"Physical size: {self.physical_size*1e6:.1f} μm")
        print(f"Target volume fraction: {self.volfrac}")
        print(f"Max iterations: {self.max_iterations}")
        print("=" * 60)
        
        # Initialize with baseline rectangular geometry if not provided
        if initial_geometry is None:
            self.current_geometry = GeometryGenerator.create_baseline_rectangular(
                self.nelx, self.nely
            )
            # Scale to desired physical size
            self.current_geometry.physical_width = self.physical_size
            self.current_geometry.physical_height = self.physical_size
            self.current_geometry.dx = self.physical_size / self.nelx
            self.current_geometry.dy = self.physical_size / self.nely
        else:
            self.current_geometry = initial_geometry
        
        # Evaluate initial geometry
        print(f"Evaluating initial geometry...")
        self.current_performance = self.devsim_obj.evaluate_geometry(self.current_geometry)
        self._log_iteration(0, self.current_performance)
        
        if not self.current_performance['success']:
            print(f"❌ Initial geometry failed DEVSIM simulation!")
            return self.current_geometry
        
        # Optimization loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration} ---")
            start_time = time.time()
            
            # Get sensitivities from DEVSIM (expensive!)
            print(f"Computing DEVSIM sensitivities...")
            sensitivities = self.devsim_obj.evaluate_sensitivity(
                self.current_geometry, delta_size=1
            )
            
            # Apply material-based optimization update
            new_geometry = self._apply_material_update(sensitivities)
            
            # Evaluate new geometry
            print(f"Evaluating new geometry...")
            new_performance = self.devsim_obj.evaluate_geometry(new_geometry)
            
            iteration_time = time.time() - start_time
            
            # Accept if better
            if new_performance['success'] and (
                not self.current_performance['success'] or 
                new_performance['objective'] < self.current_performance['objective']
            ):
                improvement = self.current_performance['objective'] - new_performance['objective']
                print(f"✓ Improvement: {improvement:.2e}")
                
                self.current_geometry = new_geometry
                self.current_performance = new_performance
                accepted = True
            else:
                print(f"→ No improvement")
                accepted = False
            
            # Log iteration
            self._log_iteration(iteration, self.current_performance, iteration_time, accepted)
            
            # Print current performance
            self.devsim_obj.print_performance_summary(self.current_performance)
        
        print(f"\n" + "="*60)
        print(f"OPTIMIZATION COMPLETE")
        print(f"="*60)
        
        # Final summary
        stats = self.devsim_obj.get_statistics()
        print(f"DEVSIM evaluations: {stats['evaluation_count']}")
        print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
        if self.current_performance['success']:
            print(f"Final performance:")
            self.devsim_obj.print_performance_summary(self.current_performance)
        else:
            print(f"❌ Final geometry failed simulation")
        
        return self.current_geometry
    
    def _apply_material_update(self, sensitivities: np.ndarray) -> GeometryMatrix:
        """
        Apply material-based optimization update
        
        Instead of density updates, directly modify material assignments
        based on sensitivity guidance
        """
        # Copy current geometry
        new_geometry = GeometryMatrix(
            self.nelx, self.nely, self.physical_size, self.physical_size
        )
        new_geometry.material_matrix = self.current_geometry.material_matrix.copy()
        
        # Get modifiable regions (excluding contacts)
        modifiable_mask = new_geometry.get_modifiable_mask()
        
        # Find most promising modifications
        modifiable_sensitivities = sensitivities * modifiable_mask
        
        # Select top candidates for modification
        flat_sensitivities = modifiable_sensitivities.flatten()
        flat_indices = np.argsort(flat_sensitivities)
        
        # Modify top candidates (positive sensitivity = beneficial change)
        n_modifications = max(1, int(self.move_limit * np.sum(modifiable_mask)))
        
        for i in range(min(n_modifications, 5)):  # Limit to 5 changes per iteration
            idx = flat_indices[-(i+1)]  # Start from highest sensitivity
            
            if flat_sensitivities[idx] <= 0:
                break  # No more beneficial changes
            
            # Convert flat index back to 2D
            y, x = np.unravel_index(idx, modifiable_sensitivities.shape)
            
            # Apply material change
            current_material = new_geometry.material_matrix[y, x]
            
            # Cycle to next material type (0->1->2->0)
            if current_material == 0:  # Void -> N-type
                new_material = 1
            elif current_material == 1:  # N-type -> P-type
                new_material = 2
            else:  # P-type -> Void
                new_material = 0
            
            new_geometry.material_matrix[y, x] = new_material
        
        # Enforce contact constraints
        new_geometry.enforce_contact_constraints()
        
        # Check that geometry is still valid
        metrics = new_geometry.calculate_metrics()
        if metrics['p_fraction'] < 0.05 or metrics['n_fraction'] < 0.05:
            print(f"  ⚠️  Modification would break P-N junction, reverting...")
            return self.current_geometry
        
        n_changes = np.sum(new_geometry.material_matrix != self.current_geometry.material_matrix)
        print(f"  Applied {n_changes} material changes")
        
        return new_geometry
    
    def _log_iteration(self, iteration: int, performance: Dict, 
                      iteration_time: float = 0, accepted: bool = True):
        """Log iteration results"""
        
        self.objective_history.append(performance.get('objective', 1e12))
        self.performance_history.append(performance)
        self.geometry_history.append(self.current_geometry.material_matrix.copy())
        
        if iteration == 0:
            print(f"Initial: Objective = {performance.get('objective', 'N/A'):.2e}")
        else:
            status = "✓" if accepted else "→"
            print(f"Iter {iteration:2d}: {status} Objective = {performance.get('objective', 'N/A'):.2e}, "
                  f"Time = {iteration_time:.1f}s")
        
        if performance['success']:
            print(f"         Forward: {performance['forward_current']:.2e} A, "
                  f"Power: {performance['power']:.2e} W, "
                  f"Rectification: {performance['rectification_ratio']:.1e}")
    
    def plot_optimization_results(self):
        """Plot optimization progress and final result"""
        fig = plt.figure(figsize=(16, 10))
        
        # Optimization progress
        ax1 = plt.subplot(2, 3, 1)
        plt.semilogy(self.objective_history, 'b-o', linewidth=2, markersize=6)
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function')
        plt.title('Optimization Progress')
        plt.grid(True, alpha=0.3)
        
        # Performance metrics over time
        ax2 = plt.subplot(2, 3, 2)
        iterations = range(len(self.performance_history))
        
        forward_currents = [p.get('forward_current', 0) for p in self.performance_history]
        powers = [p.get('power', 0) for p in self.performance_history]
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(iterations, forward_currents, 'g-o', label='Forward Current (A)')
        line2 = ax2_twin.plot(iterations, powers, 'r-s', label='Power (W)')
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Forward Current (A)', color='g')
        ax2_twin.set_ylabel('Power (W)', color='r')
        ax2.set_title('Performance Metrics')
        ax2.grid(True, alpha=0.3)
        
        # Add legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        
        # Initial vs final geometry
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
        
        ax3 = plt.subplot(2, 3, 4)
        initial_geom = self.geometry_history[0] if self.geometry_history else np.zeros((self.nely, self.nelx))
        im3 = ax3.imshow(initial_geom, cmap=cmap, vmin=0, vmax=2, origin='lower')
        ax3.set_title('Initial Geometry')
        plt.colorbar(im3, ax=ax3, ticks=[0, 1, 2], 
                    label='Material: 0=Void, 1=N-type, 2=P-type')
        
        ax4 = plt.subplot(2, 3, 5)
        final_geom = self.geometry_history[-1] if self.geometry_history else np.zeros((self.nely, self.nelx))
        im4 = ax4.imshow(final_geom, cmap=cmap, vmin=0, vmax=2, origin='lower')
        ax4.set_title('Final Geometry')
        plt.colorbar(im4, ax=ax4, ticks=[0, 1, 2],
                    label='Material: 0=Void, 1=N-type, 2=P-type')
        
        # Performance comparison
        ax5 = plt.subplot(2, 3, 3)
        if len(self.performance_history) >= 2:
            initial_perf = self.performance_history[0]
            final_perf = self.performance_history[-1]
            
            metrics = ['Objective', 'Forward Current', 'Power', 'Rectification']
            initial_values = [
                initial_perf.get('objective', 0),
                initial_perf.get('forward_current', 0),
                initial_perf.get('power', 0),
                np.log10(initial_perf.get('rectification_ratio', 1))
            ]
            final_values = [
                final_perf.get('objective', 0),
                final_perf.get('forward_current', 0),
                final_perf.get('power', 0),
                np.log10(final_perf.get('rectification_ratio', 1))
            ]
            
            x_pos = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax5.bar(x_pos - width/2, initial_values, width, 
                           label='Initial', alpha=0.7)
            bars2 = ax5.bar(x_pos + width/2, final_values, width, 
                           label='Final', alpha=0.7)
            
            ax5.set_xlabel('Metric')
            ax5.set_ylabel('Value')
            ax5.set_title('Initial vs Final Performance')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(metrics, rotation=45, ha='right')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Geometry evolution animation data
        ax6 = plt.subplot(2, 3, 6)
        if len(self.geometry_history) > 1:
            # Show difference between initial and final
            diff = final_geom.astype(float) - initial_geom.astype(float)
            im6 = ax6.imshow(diff, cmap='RdBu_r', origin='lower')
            ax6.set_title('Geometry Changes')
            plt.colorbar(im6, ax=ax6, label='Material Change')
        else:
            ax6.text(0.5, 0.5, 'No Changes', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Geometry Changes')
        
        plt.tight_layout()
        plt.savefig('devsim_simp_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def test_devsim_simp_optimization():
    """Test DEVSIM-SIMP integrated optimization"""
    print("Testing DEVSIM-SIMP Integrated Optimization")
    print("=" * 60)
    
    # Create optimizer
    optimizer = DevsimSIMPOptimizer(
        nelx=12, nely=12,  # Small for fast testing
        physical_size=10e-6,
        volfrac=0.5
    )
    
    # Test with rectangular initial geometry
    print(f"\nStarting optimization from rectangular baseline...")
    optimized_geometry = optimizer.optimize()
    
    # Plot results
    optimizer.plot_optimization_results()
    
    # Final assessment
    print(f"\n" + "="*60)
    print(f"OPTIMIZATION ASSESSMENT")
    print(f"="*60)
    
    if optimizer.performance_history:
        initial_perf = optimizer.performance_history[0]
        final_perf = optimizer.performance_history[-1]
        
        if initial_perf['success'] and final_perf['success']:
            obj_improvement = initial_perf['objective'] - final_perf['objective']
            power_change = (final_perf['power'] - initial_perf['power']) / initial_perf['power']
            current_change = (final_perf['forward_current'] - initial_perf['forward_current']) / initial_perf['forward_current']
            
            print(f"Objective improvement: {obj_improvement:.2e}")
            print(f"Power change: {power_change:+.1%}")
            print(f"Forward current change: {current_change:+.1%}")
            
            if obj_improvement > 0:
                print(f"✓ Optimization successful!")
            else:
                print(f"→ No significant improvement")
        else:
            print(f"⚠️  Simulation issues encountered")
    
    stats = optimizer.devsim_obj.get_statistics()
    print(f"\nComputational efficiency:")
    print(f"Total DEVSIM evaluations: {stats['evaluation_count']}")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
    
    return optimizer

if __name__ == "__main__":
    test_devsim_simp_optimization()