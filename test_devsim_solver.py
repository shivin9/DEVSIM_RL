#!/usr/bin/env python3
"""
Test suite for drift-diffusion semiconductor device simulator

Tests various aspects of the solver including:
- Physics validation (equilibrium, recombination, transport)
- Boundary conditions 
- Numerical stability
- Known analytical cases

Usage: python test_devsim_solver.py
"""

import numpy as np
import sys
import traceback
from typing import Tuple, Dict, Any

# Import the main solver
exec(open('devsim_diode_working.py').read().split('if __name__')[0])

class TestRunner:
    """Simple test runner for the drift-diffusion solver"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = {}
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results"""
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print('='*60)
        
        try:
            result = test_func()
            if result:
                print(f"✓ PASS: {test_name}")
                self.tests_passed += 1
                self.test_results[test_name] = "PASS"
            else:
                print(f"✗ FAIL: {test_name}")
                self.tests_failed += 1
                self.test_results[test_name] = "FAIL"
            return result
        except Exception as e:
            print(f"✗ ERROR in {test_name}: {e}")
            traceback.print_exc()
            self.tests_failed += 1
            self.test_results[test_name] = f"ERROR: {e}"
            return False
    
    def summary(self):
        """Print test summary"""
        total = self.tests_passed + self.tests_failed
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print('='*60)
        print(f"Total tests: {total}")
        print(f"Passed: {self.tests_passed}")
        print(f"Failed: {self.tests_failed}")
        print(f"Success rate: {100*self.tests_passed/total:.1f}%" if total > 0 else "No tests run")
        
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result == "PASS" else "✗"
            print(f"  {status_symbol} {test_name}: {result}")

# Test functions
def test_physical_constants():
    """Test that physical constants are reasonable"""
    print("Testing physical constants...")
    
    # Check thermal voltage at 300K
    expected_vt = 0.0259  # V at 300K
    if abs(V_T - expected_vt) > 0.001:
        print(f"✗ V_T = {V_T:.4f} V, expected ~{expected_vt:.4f} V")
        return False
    print(f"✓ V_T = {V_T:.4f} V (correct)")
    
    # Check built-in potential calculation
    expected_vbi = V_T * np.log(NA_max_m3 * ND_max_m3 / n_i_m3**2)
    if expected_vbi < 0.5 or expected_vbi > 1.2:
        print(f"✗ Built-in potential = {expected_vbi:.3f} V (suspicious)")
        return False
    print(f"✓ Built-in potential = {expected_vbi:.3f} V (reasonable)")
    
    # Check intrinsic concentration
    if n_i_m3 <= 0 or n_i_m3 > 1e20:
        print(f"✗ Intrinsic concentration = {n_i_m3:.2e} m^-3 (suspicious)")
        return False
    print(f"✓ Intrinsic concentration = {n_i_m3:.2e} m^-3 (reasonable)")
    
    return True

def test_bernoulli_function():
    """Test Bernoulli function for numerical stability"""
    print("Testing Bernoulli function...")
    
    # Test small argument (should use Taylor expansion)
    x_small = 1e-12
    B_small = bernoulli_function(x_small)
    expected_small = 1.0 - x_small/2.0  # First-order Taylor
    if abs(B_small - expected_small) > 1e-10:
        print(f"✗ B({x_small}) = {B_small}, expected ~{expected_small}")
        return False
    print(f"✓ B({x_small}) = {B_small:.10f} (Taylor expansion correct)")
    
    # Test medium argument
    x_med = 0.1
    B_med = bernoulli_function(x_med)
    expected_med = x_med / (np.exp(x_med) - 1.0)
    if abs(B_med - expected_med) > 1e-10:
        print(f"✗ B({x_med}) = {B_med}, expected {expected_med}")
        return False
    print(f"✓ B({x_med}) = {B_med:.6f} (exact formula correct)")
    
    # Test array input
    x_array = np.array([1e-15, 0.1, 1.0])
    B_array = bernoulli_function(x_array)
    if len(B_array) != 3:
        print(f"✗ Array input failed")
        return False
    print(f"✓ Array input works: B([1e-15, 0.1, 1.0]) = {B_array}")
    
    return True

def test_srh_recombination():
    """Test SRH recombination model"""
    print("Testing SRH recombination...")
    
    # Test equilibrium (R should be zero)
    n_eq = 1e16  # m^-3
    p_eq = n_i_m3**2 / n_eq  # Equilibrium condition
    R_eq = srh_recombination_rate(n_eq, p_eq)
    if abs(R_eq) > 1e10:  # Should be very small
        print(f"✗ Equilibrium recombination R = {R_eq:.2e} m^-3/s (should be ~0)")
        return False
    print(f"✓ Equilibrium recombination R = {R_eq:.2e} m^-3/s")
    
    # Test high injection (R should be positive)
    n_high = 1e18  # Much higher than equilibrium
    p_high = 1e18
    R_high = srh_recombination_rate(n_high, p_high)
    if R_high <= 0:
        print(f"✗ High injection recombination R = {R_high:.2e} m^-3/s (should be > 0)")
        return False
    print(f"✓ High injection recombination R = {R_high:.2e} m^-3/s")
    
    # Test low injection (R should be negative - generation)
    n_low = 1e12  # Much lower than equilibrium  
    p_low = 1e12
    R_low = srh_recombination_rate(n_low, p_low)
    if R_low >= 0:
        print(f"✗ Low injection recombination R = {R_low:.2e} m^-3/s (should be < 0)")
        return False
    print(f"✓ Low injection recombination R = {R_low:.2e} m^-3/s")
    
    return True

def test_slotboom_transformations():
    """Test Slotboom variable transformations"""
    print("Testing Slotboom transformations...")
    
    # Test round-trip conversion
    n_orig = 1e16  # m^-3
    p_orig = 1e14  # m^-3
    psi_test = 0.5  # V
    
    # Convert to Slotboom variables
    u, v = density_to_slotboom(n_orig, p_orig, psi_test)
    print(f"  Original: n={n_orig:.0e}, p={p_orig:.0e}, psi={psi_test}V")
    print(f"  Slotboom: u={u:.2e}, v={v:.2e}")
    
    # Convert back
    n_back, p_back = slotboom_to_density(u, v, psi_test)
    
    # Check accuracy
    n_error = abs(n_orig - n_back) / n_orig
    p_error = abs(p_orig - p_back) / p_orig
    
    if n_error > 1e-12 or p_error > 1e-12:
        print(f"✗ Conversion error: n_error={n_error:.2e}, p_error={p_error:.2e}")
        return False
    
    print(f"✓ Round-trip conversion: n_error={n_error:.2e}, p_error={p_error:.2e}")
    
    # Test with different potentials
    for psi in [-0.5, 0.0, 0.5, 1.0]:
        u, v = density_to_slotboom(n_orig, p_orig, psi)
        n_check, p_check = slotboom_to_density(u, v, psi)
        if abs(n_check - n_orig)/n_orig > 1e-12:
            print(f"✗ Failed at psi={psi}V")
            return False
    
    print(f"✓ Multiple potential values tested")
    return True

def test_equilibrium_solver():
    """Test equilibrium solver properties"""
    print("Testing equilibrium solver...")
    
    # Solve equilibrium
    psi_eq, n_eq, p_eq = solve_equilibrium()
    
    # Check potential boundary conditions (both contacts at 0V)
    psi_vals = psi_eq.x.array
    if abs(np.max(psi_vals)) > 1e-10 or abs(np.min(psi_vals)) > 1e-10:
        print(f"✗ Potential not zero everywhere: range=[{np.min(psi_vals):.2e}, {np.max(psi_vals):.2e}]V")
        return False
    print(f"✓ Equilibrium potential = 0V everywhere (correct for zero bias)")
    
    # Check charge neutrality (approximately)
    n_vals = n_eq.x.array
    p_vals = p_eq.x.array
    C_vals = C.x.array
    
    # Net charge density: rho = q(p - n + C)
    rho = q * (p_vals - n_vals + C_vals)
    max_rho = np.max(np.abs(rho))
    
    # Should be small (charge neutrality)
    if max_rho > 1e-10:
        print(f"✗ Charge neutrality violated: max|rho| = {max_rho:.2e} C/m^3")
        return False
    print(f"✓ Charge neutrality satisfied: max|rho| = {max_rho:.2e} C/m^3")
    
    # Check carrier concentrations are positive
    if np.min(n_vals) <= 0 or np.min(p_vals) <= 0:
        print(f"✗ Negative carrier concentrations found")
        return False
    print(f"✓ All carrier concentrations positive")
    
    print(f"  n range: [{np.min(n_vals):.2e}, {np.max(n_vals):.2e}] m^-3")
    print(f"  p range: [{np.min(p_vals):.2e}, {np.max(p_vals):.2e}] m^-3")
    
    return True

def test_continuity_solvers():
    """Test individual continuity equation solvers"""
    print("Testing continuity equation solvers...")
    
    # Create test potential and initial carriers
    psi_test = Function(V_psi)
    u_init = Function(V_psi)
    v_init = Function(V_psi)
    
    # Set up simple test case
    psi_test.x.array[:] = 0.1  # Small bias
    u_init.x.array[:] = 1.0    # Equilibrium Slotboom variables
    v_init.x.array[:] = 1.0
    
    # Test electron continuity solver
    print("  Testing electron continuity solver...")
    u_result = solve_electron_continuity(psi_test, v_init, applied_voltage=0.1)
    
    # Check result is reasonable
    u_vals = u_result.x.array
    if np.min(u_vals) <= 0 or np.max(u_vals) > 1e15:
        print(f"✗ Electron Slotboom variable out of range: [{np.min(u_vals):.2e}, {np.max(u_vals):.2e}]")
        return False
    print(f"    ✓ Electron solver converged: u range=[{np.min(u_vals):.2e}, {np.max(u_vals):.2e}]")
    
    # Test hole continuity solver
    print("  Testing hole continuity solver...")
    v_result = solve_hole_continuity(psi_test, u_result, applied_voltage=0.1)
    
    # Check result is reasonable
    v_vals = v_result.x.array
    if np.min(v_vals) <= 0 or np.max(v_vals) > 1e15:
        print(f"✗ Hole Slotboom variable out of range: [{np.min(v_vals):.2e}, {np.max(v_vals):.2e}]")
        return False
    print(f"    ✓ Hole solver converged: v range=[{np.min(v_vals):.2e}, {np.max(v_vals):.2e}]")
    
    # Test coupled solve (electron then hole)
    print("  Testing coupled solve...")
    for i in range(3):  # Few Gummel-like iterations
        u_result = solve_electron_continuity(psi_test, v_result, u_result, applied_voltage=0.1)
        v_result = solve_hole_continuity(psi_test, u_result, v_result, applied_voltage=0.1)
    
    print(f"    ✓ Coupled iterations completed")
    
    return True

def test_boundary_current_calculation():
    """Test boundary current calculation"""
    print("Testing boundary current calculation...")
    
    # Create simple test case with known gradients
    psi_test = Function(V_psi)
    n_test = Function(fem.functionspace(domain, ('DG', 0)))
    p_test = Function(fem.functionspace(domain, ('DG', 0)))
    
    # Set up linear potential drop
    def linear_potential(x):
        return 0.1 * x[1] / height_m  # Linear from 0 to 0.1V
    
    psi_test.interpolate(linear_potential)
    
    # Set uniform carrier densities
    n_test.x.array[:] = 1e16  # m^-3
    p_test.x.array[:] = 1e14  # m^-3
    
    # Calculate current at bottom boundary (cathode)
    try:
        I_boundary = calculate_boundary_current(psi_test, n_test, p_test, boundary_id=2)
        print(f"  Calculated boundary current: {I_boundary:.2e} A/m")
        
        # Should be non-zero due to electric field
        if abs(I_boundary) < 1e-20:
            print(f"✗ Current too small (possibly zero)")
            return False
        
        print(f"✓ Boundary current calculation successful")
        return True
        
    except Exception as e:
        print(f"✗ Error in boundary current calculation: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability under various conditions"""
    print("Testing numerical stability...")
    
    # Test Bernoulli function at extreme values
    extreme_vals = [-50, -10, -1e-15, 0, 1e-15, 10, 50]
    for x in extreme_vals:
        try:
            B = bernoulli_function(x)
            if not np.isfinite(B):
                print(f"✗ Bernoulli function not finite at x={x}: B={B}")
                return False
        except Exception as e:
            print(f"✗ Bernoulli function failed at x={x}: {e}")
            return False
    
    print(f"✓ Bernoulli function stable at extreme values")
    
    # Test SRH recombination with extreme carrier densities
    extreme_carriers = [1e5, 1e10, 1e15, 1e20, 1e25]
    for n in extreme_carriers:
        for p in extreme_carriers:
            try:
                R = srh_recombination_rate(n, p)
                if not np.isfinite(R):
                    print(f"✗ SRH recombination not finite: n={n:.0e}, p={p:.0e}, R={R}")
                    return False
            except Exception as e:
                print(f"✗ SRH recombination failed: n={n:.0e}, p={p:.0e}: {e}")
                return False
    
    print(f"✓ SRH recombination stable with extreme carrier densities")
    
    return True

def test_main_simulation_runs():
    """Test that main simulation runs without crashing"""
    print("Testing main simulation execution...")
    
    # Run a very short simulation
    test_voltages = [0.05, 0.1]  # Just 2 points
    
    try:
        voltages, currents, drift_currents, diffusion_currents, conductivities = voltage_stepping_simulation(
            test_voltages, plot_results=False)
        
        if len(voltages) < 2:  # Should have equilibrium + test points
            print(f"✗ Too few simulation points: {len(voltages)}")
            return False
        
        if len(currents) != len(voltages):
            print(f"✗ Inconsistent result arrays: V={len(voltages)}, I={len(currents)}")
            return False
        
        # Check for reasonable current values
        max_current = max(np.abs(currents))
        if max_current == 0 or not np.isfinite(max_current):
            print(f"✗ Unreasonable current values: max={max_current}")
            return False
        
        print(f"✓ Main simulation completed: {len(voltages)} points, max current={max_current:.2e} A/m")
        return True
        
    except Exception as e:
        print(f"✗ Main simulation failed: {e}")
        return False

# Main test execution
if __name__ == "__main__":
    print("DRIFT-DIFFUSION SOLVER TEST SUITE")
    print("="*60)
    
    runner = TestRunner()
    
    # Run all tests
    runner.run_test("Physical Constants", test_physical_constants)
    runner.run_test("Bernoulli Function", test_bernoulli_function)
    runner.run_test("SRH Recombination", test_srh_recombination)
    runner.run_test("Slotboom Transformations", test_slotboom_transformations)
    runner.run_test("Equilibrium Solver", test_equilibrium_solver)
    runner.run_test("Continuity Solvers", test_continuity_solvers)
    runner.run_test("Boundary Current Calculation", test_boundary_current_calculation)
    runner.run_test("Numerical Stability", test_numerical_stability)
    runner.run_test("Main Simulation Execution", test_main_simulation_runs)
    
    # Print summary
    runner.summary()
    
    # Exit with appropriate code
    sys.exit(0 if runner.tests_failed == 0 else 1)