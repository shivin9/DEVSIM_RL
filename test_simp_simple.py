#!/usr/bin/env python3
"""
Simple test of SIMP implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from simp_topology_optimization import SIMPOptimizer

def simple_simp_test():
    """Simple test with small problem size"""
    print("Simple SIMP Test")
    print("=" * 30)
    
    # Small test case
    simp = SIMPOptimizer(nelx=10, nely=10, volfrac=0.5, penal=3.0, rmin=1.0)
    
    print(f"Initialized: {simp.nelx}x{simp.nely} elements")
    print(f"Initial density sum: {np.sum(simp.x)}")
    
    # Test individual components
    print("\n1. Testing material interpolation...")
    test_densities = np.array([0.0, 0.5, 1.0])
    for rho in test_densities:
        sigma = simp._material_interpolation(np.array([[rho]]))
        print(f"  ρ={rho} → σ={sigma[0,0]:.2e}")
    
    print("\n2. Testing electrical analysis...")
    x_test = np.ones((simp.nely, simp.nelx)) * 0.5
    compliance, V = simp._simplified_electrical_analysis(x_test)
    print(f"  Compliance: {compliance:.2e}")
    print(f"  Voltage range: {np.min(V):.3f} to {np.max(V):.3f}")
    
    print("\n3. Testing sensitivity analysis...")
    dc, c0 = simp._sensitivity_analysis(x_test)
    print(f"  Base compliance: {c0:.2e}")
    print(f"  Sensitivity range: {np.min(dc):.2e} to {np.max(dc):.2e}")
    
    print("\n4. Running short optimization...")
    x_opt = simp.optimize(max_iter=5, tol=1e-6)
    
    print(f"\nOptimization complete!")
    print(f"Final volume fraction: {np.sum(x_opt)/(simp.nelx*simp.nely):.3f}")
    
    # Simple visualization
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.imshow(simp.x, cmap='gray', origin='lower')
    plt.title('Final Density')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    if simp.compliance_history:
        plt.plot(simp.compliance_history, 'b-o')
        plt.title('Compliance History')
        plt.xlabel('Iteration')
        plt.ylabel('Compliance')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_simp_test.png', dpi=150)
    plt.show()
    
    return simp

if __name__ == "__main__":
    simple_simp_test()