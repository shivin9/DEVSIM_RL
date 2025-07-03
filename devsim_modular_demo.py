#!/usr/bin/env python3
"""
Modular Drift-Diffusion Solver Demonstration

This demonstrates the new modular structure while keeping the original
implementation intact. Shows how components can be imported and used
in a cleaner, more maintainable way.

Usage: python devsim_modular_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Import modular components
from devsim_solver.physics import constants, recombination, transport
from devsim_solver.numerics import stability, slotboom

def demonstrate_modular_components():
    """Demonstrate the modular component functionality"""
    
    print("="*60)
    print("MODULAR DRIFT-DIFFUSION SOLVER DEMONSTRATION")
    print("="*60)
    
    print("\n1. PHYSICS COMPONENTS")
    print("-" * 30)
    constants.print_constants()
    
    print("\n2. NUMERICAL METHODS")
    print("-" * 30)
    
    # Demonstrate Bernoulli function
    print("Bernoulli function behavior:")
    x_vals = np.linspace(-5, 5, 11)
    B_vals = [stability.bernoulli_function(x) for x in x_vals]
    
    for x, B in zip(x_vals, B_vals):
        print(f"  B({x:4.1f}) = {B:8.4f}")
    
    # Demonstrate Scharfetter-Gummel weights
    print("\nScharfetter-Gummel discretization weights:")
    sg_vals = [0.1, 1.0, 2.0, 5.0]
    for x in sg_vals:
        w_up, w_down = stability.scharfetter_gummel_weights(x)
        print(f"  x={x}: w_up={w_up:.4f}, w_down={w_down:.4f}, sum={w_up+w_down:.4f}")
    
    print("\n3. SLOTBOOM TRANSFORMATIONS")
    print("-" * 30)
    
    # Demonstrate Slotboom variables
    test_conditions = [
        ("Equilibrium", 1e16, 1e16, 0.0),
        ("Forward bias", 5e16, 2e15, 0.6),
        ("Reverse bias", 2e15, 5e16, -0.5),
    ]
    
    for name, n, p, psi in test_conditions:
        u, v = slotboom.density_to_slotboom(n, p, psi)
        n_back, p_back = slotboom.slotboom_to_density(u, v, psi)
        
        print(f"  {name}:")
        print(f"    Input:  n={n:.1e}, p={p:.1e}, ψ={psi:.1f}V")
        print(f"    Slotboom: u={u:.2e}, v={v:.2e}")
        print(f"    Recovered: n={n_back:.1e}, p={p_back:.1e}")
        
        # Check accuracy
        n_err = abs(n_back - n) / n
        p_err = abs(p_back - p) / p
        print(f"    Errors: Δn={n_err:.1e}, Δp={p_err:.1e}")
    
    print("\n4. RECOMBINATION MODELS")
    print("-" * 30)
    
    # Demonstrate recombination under different conditions
    recomb_conditions = [
        ("Equilibrium", 1e16, 1e16),
        ("High injection", 1e18, 1e18),
        ("Low injection", 1e12, 1e12),
        ("Non-equilibrium", 1e17, 1e14),
    ]
    
    for name, n, p in recomb_conditions:
        R_srh = recombination.srh_recombination_rate(n, p)
        R_auger = recombination.auger_recombination_rate(n, p)
        R_rad = recombination.radiative_recombination_rate(n, p)
        R_total = recombination.total_recombination_rate(n, p, include_auger=True, include_radiative=True)
        
        print(f"  {name}: n={n:.0e}, p={p:.0e}")
        print(f"    R_SRH = {R_srh:.2e} m⁻³/s")
        print(f"    R_Auger = {R_auger:.2e} m⁻³/s")
        print(f"    R_radiative = {R_rad:.2e} m⁻³/s")
        print(f"    R_total = {R_total:.2e} m⁻³/s")
    
    print("\n5. TRANSPORT MODELS")
    print("-" * 30)
    
    # Demonstrate mobility models
    print("Mobility models:")
    mu_n_const, mu_p_const = transport.constant_mobility()
    print(f"  Constant: μ_n = {mu_n_const:.3f}, μ_p = {mu_p_const:.3f} m²/V/s")
    
    # Field-dependent mobility
    E_fields = [1e3, 1e4, 1e5, 1e6]  # V/m
    print("  Field-dependent (electrons):")
    for E in E_fields:
        mu_n_field = transport.field_dependent_mobility(E, mu_n_const)
        print(f"    E = {E:.0e} V/m: μ_n = {mu_n_field:.4f} m²/V/s")
    
    # Einstein relation verification
    D_n, D_p = transport.diffusion_coefficients(mu_n_const, mu_p_const)
    einstein_ok = transport.einstein_relation_check(mu_n_const, mu_p_const, D_n, D_p)
    print(f"  Einstein relation: {'✓ Valid' if einstein_ok else '✗ Invalid'}")
    print(f"    D_n = {D_n*1e4:.1f} cm²/s, D_p = {D_p*1e4:.1f} cm²/s")

def plot_modular_functions():
    """Create plots demonstrating modular functions"""
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Bernoulli function
        x = np.linspace(-10, 10, 1000)
        B = np.array([stability.bernoulli_function(xi) for xi in x])
        
        ax1.plot(x, B, 'b-', linewidth=2, label='B(x) = x/(exp(x)-1)')
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('B(x)')
        ax1.set_title('Bernoulli Function')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(-2, 10)
        
        # Plot 2: Slotboom transformation
        psi_vals = np.linspace(-1, 1, 100)
        n_fixed = 1e16
        p_fixed = 1e15
        
        u_vals = []
        v_vals = []
        for psi in psi_vals:
            u, v = slotboom.density_to_slotboom(n_fixed, p_fixed, psi)
            u_vals.append(u)
            v_vals.append(v)
        
        ax2.semilogy(psi_vals, u_vals, 'r-', label='u (electrons)', linewidth=2)
        ax2.semilogy(psi_vals, v_vals, 'b-', label='v (holes)', linewidth=2)
        ax2.set_xlabel('Potential ψ (V)')
        ax2.set_ylabel('Slotboom Variables')
        ax2.set_title('Slotboom Variables vs Potential')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Recombination rates
        n_vals = np.logspace(12, 20, 100)
        p_fixed = 1e15
        
        R_srh = [recombination.srh_recombination_rate(n, p_fixed) for n in n_vals]
        R_auger = [recombination.auger_recombination_rate(n, p_fixed) for n in n_vals]
        
        ax3.semilogx(n_vals, R_srh, 'g-', label='SRH', linewidth=2)
        ax3.semilogx(n_vals, R_auger, 'm-', label='Auger', linewidth=2)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Electron Density (m⁻³)')
        ax3.set_ylabel('Recombination Rate (m⁻³/s)')
        ax3.set_title('Recombination Mechanisms')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Field-dependent mobility
        E_vals = np.logspace(2, 7, 100)
        mu_n_field = [transport.field_dependent_mobility(E, constants.mu_n0) for E in E_vals]
        mu_p_field = [transport.field_dependent_mobility(E, constants.mu_p0) for E in E_vals]
        
        ax4.loglog(E_vals, mu_n_field, 'r-', label='Electrons', linewidth=2)
        ax4.loglog(E_vals, mu_p_field, 'b-', label='Holes', linewidth=2)
        ax4.axhline(y=constants.mu_n0, color='r', linestyle='--', alpha=0.5)
        ax4.axhline(y=constants.mu_p0, color='b', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Electric Field (V/m)')
        ax4.set_ylabel('Mobility (m²/V/s)')
        ax4.set_title('Field-Dependent Mobility')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Plots generated successfully")
        
    except ImportError:
        print("\n⚠ Matplotlib not available - skipping plots")

def run_component_tests():
    """Run tests on all modular components"""
    
    print("\n" + "="*60)
    print("COMPONENT TESTING")
    print("="*60)
    
    tests = [
        ("Physics - Constants", lambda: True),  # Constants don't need testing
        ("Physics - Recombination", recombination.equilibrium_test),
        ("Physics - Transport", transport.transport_test),
        ("Numerics - Stability", stability.stability_test),
        ("Numerics - Slotboom", slotboom.slotboom_test),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{test_name:25} {status}")
            if result:
                passed += 1
        except Exception as e:
            print(f"{test_name:25} ✗ ERROR: {e}")
    
    print(f"\nTest Summary: {passed}/{total} passed ({100*passed/total:.0f}%)")
    return passed == total

if __name__ == "__main__":
    # Demonstrate modular components
    demonstrate_modular_components()
    
    # Run component tests
    all_passed = run_component_tests()
    
    # Generate plots
    plot_modular_functions()
    
    print("\n" + "="*60)
    print("MODULAR DEMONSTRATION COMPLETE")
    print("="*60)
    print(f"Status: {'✓ SUCCESS' if all_passed else '✗ SOME TESTS FAILED'}")
    print("\nThe modular structure provides:")
    print("  • Clean separation of physics, numerics, and solvers")
    print("  • Reusable components for different device types")
    print("  • Better testing and validation capabilities")
    print("  • Easier extension with new physics models")
    print("  • Improved maintainability and documentation")
    print("\nOriginal monolithic solver remains intact in devsim_diode_working.py")