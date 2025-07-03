# Drift-Diffusion Solver Test Suite

This directory contains comprehensive tests for the drift-diffusion semiconductor device simulator.

## Test Files

- **`test_devsim_solver.py`** - Main test suite with comprehensive physics and numerical tests
- **`run_tests.py`** - Simple test runner script
- **`TEST_README.md`** - This documentation file

## Running Tests

### Quick Test Run
```bash
python run_tests.py --quick
```

### Full Test Suite
```bash
python test_devsim_solver.py
```
or
```bash
python run_tests.py
```

## Test Categories

### 1. Physics Validation Tests
- **Physical Constants**: Validates thermal voltage, built-in potential, intrinsic concentration
- **SRH Recombination**: Tests recombination model under equilibrium, high/low injection
- **Slotboom Transformations**: Validates round-trip conversion accuracy

### 2. Numerical Stability Tests  
- **Bernoulli Function**: Tests numerical stability with Taylor expansion for small arguments
- **Extreme Values**: Tests solver behavior with extreme carrier densities and potentials
- **Convergence**: Validates solver convergence under various conditions

### 3. Solver Component Tests
- **Equilibrium Solver**: Tests charge neutrality, boundary conditions, carrier positivity
- **Continuity Solvers**: Tests individual electron/hole continuity equation solvers
- **Boundary Current**: Validates current integration at device terminals

### 4. Integration Tests
- **Main Simulation**: Tests full voltage stepping simulation execution
- **Coupled Solving**: Tests electron-hole solver coupling (Gummel-like iterations)

## Test Results

All tests should pass with the following expected behaviors:

✅ **Physical Constants** - Validates V_T ≈ 0.0259V, reasonable built-in potential
✅ **Bernoulli Function** - Numerical stability with Taylor expansion for |x| < 1e-10  
✅ **SRH Recombination** - Zero at equilibrium, positive for high injection, negative for low injection
✅ **Slotboom Transformations** - Perfect round-trip conversion (error < 1e-12)
✅ **Equilibrium Solver** - Zero potential, charge neutrality, positive carriers
✅ **Continuity Solvers** - Convergence in 1-2 iterations for test cases
✅ **Boundary Current** - Non-zero current with applied field
✅ **Numerical Stability** - No NaN/Inf values under extreme conditions
✅ **Main Simulation** - Successful voltage stepping with reasonable currents

## Test Coverage

The test suite covers:

- ✅ All helper functions (Bernoulli, SRH, Slotboom)
- ✅ Individual equation solvers (Poisson, electron continuity, hole continuity)
- ✅ Boundary condition handling
- ✅ Current calculation and integration
- ✅ Numerical stability and edge cases
- ✅ Full system integration
- ✅ Physics validation against analytical results

## Known Issues

1. **Convergence Warning**: Occasionally one Newton iteration may not converge in coupled solving, but fallback mechanisms ensure stability
2. **Large Potential Values**: Under extreme bias, potentials can become very large (this indicates need for better physics models)
3. **Current Magnitude**: Current values are often very large due to simplified carrier models (will improve with full drift-diffusion implementation)

## Adding New Tests

To add new tests:

1. Add test function to `test_devsim_solver.py`
2. Follow naming convention: `test_feature_name()`
3. Return `True` for pass, `False` for fail
4. Add test to main execution block
5. Include descriptive print statements for debugging

Example test function:
```python
def test_new_feature():
    """Test description"""
    print("Testing new feature...")
    
    # Test implementation
    result = some_calculation()
    
    if result_is_correct:
        print("✓ Test passed")
        return True
    else:
        print("✗ Test failed")
        return False
```

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```bash
# Exit code 0 = all tests pass
# Exit code 1 = some tests failed
python run_tests.py
echo $?  # Check exit code
```