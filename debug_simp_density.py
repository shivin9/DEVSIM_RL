#!/usr/bin/env python3
"""
Debug SIMP Density Convergence Issues
Systematic debugging of why SIMP converges to zero density
"""

import numpy as np
import matplotlib.pyplot as plt
from simp_topology_optimization import SIMPOptimizer

class SIMPDebugger:
    """Debugger for SIMP density convergence issues"""
    
    def __init__(self):
        self.debug_results = {}
        
    def debug_volume_constraint_enforcement(self):
        """Test if volume constraint is being enforced correctly"""
        print("\n" + "="*60)
        print("DEBUGGING: Volume Constraint Enforcement")
        print("="*60)
        
        # Test different volume fractions
        test_volfrac = [0.3, 0.5, 0.8]
        results = {}
        
        for vf in test_volfrac:
            print(f"\nTesting volume fraction: {vf}")
            
            simp = SIMPOptimizer(nelx=8, nely=8, volfrac=vf, penal=3.0, rmin=1.0)
            
            # Run short optimization
            x_opt = simp.optimize(max_iter=10, tol=1e-6)
            
            final_vf = np.sum(x_opt) / (simp.nelx * simp.nely)
            
            print(f"  Target VF: {vf:.3f}")
            print(f"  Final VF:  {final_vf:.3f}")
            print(f"  Error:     {abs(final_vf - vf):.3f}")
            print(f"  Final compliance: {simp.compliance_history[-1]:.2e}")
            
            results[vf] = {
                'target_vf': vf,
                'final_vf': final_vf,
                'error': abs(final_vf - vf),
                'compliance_history': simp.compliance_history.copy(),
                'volume_history': simp.volume_history.copy(),
                'final_design': x_opt.copy()
            }
        
        # Analysis
        print(f"\n" + "-"*40)
        print("VOLUME CONSTRAINT ANALYSIS:")
        for vf, result in results.items():
            success = result['error'] < 0.05  # Within 5%
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"  VF {vf}: {status} (error: {result['error']:.3f})")
        
        self.debug_results['volume_constraint'] = results
        return results
    
    def debug_objective_function_incentives(self):
        """Test if objective function creates proper material incentives"""
        print("\n" + "="*60)
        print("DEBUGGING: Objective Function Incentives")
        print("="*60)
        
        simp = SIMPOptimizer(nelx=8, nely=8, volfrac=0.5, penal=3.0, rmin=1.0)
        
        # Test different density patterns
        test_patterns = {
            'uniform_low': np.ones((8, 8)) * 0.1,
            'uniform_mid': np.ones((8, 8)) * 0.5,
            'uniform_high': np.ones((8, 8)) * 0.9,
            'left_right': np.concatenate([np.ones((8, 4)) * 0.9, np.ones((8, 4)) * 0.1], axis=1),
            'checkerboard': np.array([[0.9 if (i+j)%2 == 0 else 0.1 for j in range(8)] for i in range(8)])
        }
        
        results = {}
        
        for name, pattern in test_patterns.items():
            print(f"\nTesting pattern: {name}")
            
            # Evaluate compliance for this pattern
            compliance, V = simp._simplified_electrical_analysis(pattern)
            dc, _ = simp._analytical_sensitivity_analysis(pattern)
            
            # Material statistics
            avg_density = np.mean(pattern)
            density_std = np.std(pattern)
            
            # Sensitivity statistics
            avg_sensitivity = np.mean(dc)
            sensitivity_range = np.max(dc) - np.min(dc)
            
            print(f"  Avg density: {avg_density:.3f} ± {density_std:.3f}")
            print(f"  Compliance: {compliance:.2e}")
            print(f"  Avg sensitivity: {avg_sensitivity:.2e}")
            print(f"  Sensitivity range: {sensitivity_range:.2e}")
            
            # Check if sensitivities encourage material
            encouraging_material = np.sum(dc < 0) > np.sum(dc > 0)
            print(f"  Encouraging material: {encouraging_material}")
            
            results[name] = {
                'pattern': pattern,
                'compliance': compliance,
                'avg_density': avg_density,
                'avg_sensitivity': avg_sensitivity,
                'sensitivity_range': sensitivity_range,
                'encouraging_material': encouraging_material,
                'voltage': V
            }
        
        self.debug_results['objective_incentives'] = results
        return results
    
    def debug_material_interpolation(self):
        """Test material interpolation function"""
        print("\n" + "="*60)
        print("DEBUGGING: Material Interpolation")
        print("="*60)
        
        simp = SIMPOptimizer(nelx=8, nely=8, volfrac=0.5, penal=3.0, rmin=1.0)
        
        # Test range of densities
        densities = np.linspace(0, 1, 11)
        
        print("Density → Conductivity mapping:")
        print("Density  | Conductivity | Relative")
        print("-" * 35)
        
        conductivities = []
        for rho in densities:
            sigma = simp._material_interpolation(np.array([[rho]]))[0, 0]
            relative = sigma / simp.E1 if simp.E1 > 0 else 0
            conductivities.append(sigma)
            print(f"{rho:6.1f}   | {sigma:.2e}  | {relative:6.3f}")
        
        # Check if interpolation is reasonable
        print(f"\nMaterial properties:")
        print(f"  E0 (void): {simp.E0:.2e}")
        print(f"  E1 (solid): {simp.E1:.2e}")
        print(f"  Ratio E1/E0: {simp.E1/simp.E0:.1e}")
        print(f"  Penalty parameter: {simp.penal}")
        
        # Test derivative
        print(f"\nTesting derivative calculation...")
        test_densities = [0.1, 0.5, 0.9]
        
        for rho in test_densities:
            # Analytical derivative
            dE_drho_analytical = simp.penal * (simp.E1 - simp.E0) * rho**(simp.penal - 1)
            
            # Numerical derivative
            delta = 1e-6
            sigma_plus = simp._material_interpolation(np.array([[rho + delta]]))[0, 0]
            sigma_minus = simp._material_interpolation(np.array([[rho - delta]]))[0, 0]
            dE_drho_numerical = (sigma_plus - sigma_minus) / (2 * delta)
            
            error = abs(dE_drho_analytical - dE_drho_numerical) / abs(dE_drho_analytical + 1e-16)
            
            print(f"  ρ={rho}: analytical={dE_drho_analytical:.2e}, numerical={dE_drho_numerical:.2e}, error={error:.2e}")
        
        return {
            'densities': densities,
            'conductivities': conductivities,
            'E0': simp.E0,
            'E1': simp.E1,
            'penal': simp.penal
        }
    
    def debug_optimality_criteria_update(self):
        """Test Optimality Criteria update mechanism"""
        print("\n" + "="*60)
        print("DEBUGGING: Optimality Criteria Update")
        print("="*60)
        
        simp = SIMPOptimizer(nelx=8, nely=8, volfrac=0.5, penal=3.0, rmin=1.0)
        
        # Test with known density and sensitivity
        x_current = np.ones((8, 8)) * 0.5  # Start at target volume
        
        # Get sensitivities
        dc, compliance = simp._analytical_sensitivity_analysis(x_current)
        
        print(f"Before update:")
        print(f"  Volume fraction: {np.sum(x_current)/(8*8):.3f}")
        print(f"  Compliance: {compliance:.2e}")
        print(f"  Sensitivity range: [{np.min(dc):.2e}, {np.max(dc):.2e}]")
        
        # Apply OC update
        x_new = simp._optimality_criteria_update(x_current, dc)
        
        print(f"\nAfter update:")
        print(f"  Volume fraction: {np.sum(x_new)/(8*8):.3f}")
        print(f"  Change: {np.max(np.abs(x_new - x_current)):.3f}")
        print(f"  Density range: [{np.min(x_new):.3f}, {np.max(x_new):.3f}]")
        
        # Test volume conservation
        target_volume = simp.volfrac * simp.nelx * simp.nely
        actual_volume = np.sum(x_new)
        volume_error = abs(actual_volume - target_volume) / target_volume
        
        print(f"\nVolume conservation:")
        print(f"  Target volume: {target_volume:.1f}")
        print(f"  Actual volume: {actual_volume:.1f}")
        print(f"  Error: {volume_error:.3f}")
        
        # Test multiple updates
        print(f"\nTesting sequential updates:")
        x = x_current.copy()
        for i in range(5):
            dc, compliance = simp._analytical_sensitivity_analysis(x)
            x = simp._optimality_criteria_update(x, dc)
            vf = np.sum(x) / (8*8)
            print(f"  Update {i+1}: VF={vf:.3f}, Compliance={compliance:.2e}")
        
        return {
            'initial_density': x_current,
            'final_density': x_new,
            'volume_error': volume_error
        }
    
    def debug_boundary_conditions(self):
        """Test if boundary conditions create meaningful gradients"""
        print("\n" + "="*60)
        print("DEBUGGING: Boundary Conditions")
        print("="*60)
        
        simp = SIMPOptimizer(nelx=8, nely=8, volfrac=0.5, penal=3.0, rmin=1.0)
        
        # Test with uniform density
        x_uniform = np.ones((8, 8)) * 0.5
        compliance, V = simp._simplified_electrical_analysis(x_uniform)
        
        print(f"Uniform density test:")
        print(f"  Voltage boundary values:")
        print(f"    Left (V=1): {V[:, 0]}")
        print(f"    Right (V=0): {V[:, -1]}")
        print(f"  Voltage range: [{np.min(V):.3f}, {np.max(V):.3f}]")
        print(f"  Compliance: {compliance:.2e}")
        
        # Check voltage gradient
        V_grad_x = np.gradient(V, axis=1)
        V_grad_y = np.gradient(V, axis=0)
        grad_magnitude = np.sqrt(V_grad_x**2 + V_grad_y**2)
        
        print(f"  Voltage gradient magnitude: [{np.min(grad_magnitude):.3f}, {np.max(grad_magnitude):.3f}]")
        print(f"  Average gradient: {np.mean(grad_magnitude):.3f}")
        
        # Test if gradient creates current flow
        has_meaningful_gradient = np.mean(grad_magnitude) > 0.01
        print(f"  Meaningful gradient: {has_meaningful_gradient}")
        
        return {
            'voltage': V,
            'gradient_magnitude': grad_magnitude,
            'has_meaningful_gradient': has_meaningful_gradient
        }
    
    def run_full_debug_session(self):
        """Run complete debugging session"""
        print("SIMP DENSITY CONVERGENCE DEBUGGING SESSION")
        print("=" * 60)
        
        # Run all debug tests
        vol_results = self.debug_volume_constraint_enforcement()
        obj_results = self.debug_objective_function_incentives()
        mat_results = self.debug_material_interpolation()
        oc_results = self.debug_optimality_criteria_update()
        bc_results = self.debug_boundary_conditions()
        
        # Summary analysis
        print("\n" + "="*60)
        print("DEBUGGING SUMMARY")
        print("="*60)
        
        # Volume constraint issues
        vol_failures = sum(1 for vf, result in vol_results.items() if result['error'] > 0.05)
        print(f"Volume constraint failures: {vol_failures}/{len(vol_results)}")
        
        # Material incentive issues
        encouraging_count = sum(1 for result in obj_results.values() if result['encouraging_material'])
        print(f"Patterns encouraging material: {encouraging_count}/{len(obj_results)}")
        
        # Boundary condition issues
        print(f"Meaningful voltage gradient: {bc_results['has_meaningful_gradient']}")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        if vol_failures > 0:
            print("- Fix volume constraint enforcement in Optimality Criteria")
        if encouraging_count < len(obj_results) // 2:
            print("- Adjust material interpolation or objective function")
        if not bc_results['has_meaningful_gradient']:
            print("- Fix boundary conditions to create meaningful voltage gradients")
        
        # Create visualization
        self._create_debug_visualization()
        
        return {
            'volume_constraint': vol_results,
            'objective_incentives': obj_results,
            'material_interpolation': mat_results,
            'optimality_criteria': oc_results,
            'boundary_conditions': bc_results
        }
    
    def _create_debug_visualization(self):
        """Create comprehensive debugging visualization"""
        fig = plt.figure(figsize=(16, 12))
        
        # Volume constraint test
        if 'volume_constraint' in self.debug_results:
            vol_results = self.debug_results['volume_constraint']
            
            ax1 = plt.subplot(2, 3, 1)
            target_vfs = list(vol_results.keys())
            final_vfs = [result['final_vf'] for result in vol_results.values()]
            plt.plot(target_vfs, target_vfs, 'k--', label='Perfect')
            plt.plot(target_vfs, final_vfs, 'ro-', label='Actual')
            plt.xlabel('Target Volume Fraction')
            plt.ylabel('Final Volume Fraction')
            plt.title('Volume Constraint Enforcement')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Volume histories
            ax2 = plt.subplot(2, 3, 2)
            for vf, result in vol_results.items():
                plt.plot(result['volume_history'], label=f'VF {vf}')
            plt.xlabel('Iteration')
            plt.ylabel('Volume Fraction')
            plt.title('Volume Fraction Histories')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Final designs
            for i, (vf, result) in enumerate(vol_results.items()):
                ax = plt.subplot(2, 3, 3 + i)
                plt.imshow(result['final_design'], cmap='viridis', origin='lower')
                plt.title(f'Final Design VF={vf}')
                plt.colorbar()
        
        plt.tight_layout()
        plt.savefig('simp_debug_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Run SIMP debugging session"""
    debugger = SIMPDebugger()
    results = debugger.run_full_debug_session()
    
    print(f"\nDebugging complete! Check 'simp_debug_analysis.png' for visualizations.")
    return results

if __name__ == "__main__":
    main()