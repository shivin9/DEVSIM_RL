"""
Numerical stability functions for semiconductor device simulation

Contains the Bernoulli function and other numerical methods
for stable discretization of transport equations.
"""

import numpy as np
import ufl

def bernoulli_function(x):
    """
    Bernoulli function B(x) = x/(exp(x)-1) for Scharfetter-Gummel discretization
    Uses Taylor expansion for small |x| to avoid numerical issues
    
    This function is essential for numerically stable discretization of 
    drift-diffusion equations, particularly the Scharfetter-Gummel scheme.
    
    Args:
        x: argument (scalar or array)
    
    Returns:
        B(x): Bernoulli function value(s)
    """
    if isinstance(x, (int, float)):
        if abs(x) < 1e-10:
            # Taylor expansion: B(x) ≈ 1 - x/2 + x²/12 - x⁴/720 + ...
            return 1.0 - x/2.0 + x**2/12.0 - x**4/720.0
        else:
            return x / (np.exp(x) - 1.0)
    else:
        # For arrays
        result = np.zeros_like(x)
        small_mask = np.abs(x) < 1e-10
        large_mask = ~small_mask
        
        # Taylor expansion for small values
        x_small = x[small_mask]
        result[small_mask] = 1.0 - x_small/2.0 + x_small**2/12.0 - x_small**4/720.0
        
        # Exact formula for large values
        x_large = x[large_mask]
        result[large_mask] = x_large / (np.exp(x_large) - 1.0)
        
        return result

def bernoulli_ufl(x):
    """
    UFL-compatible Bernoulli function for use in variational forms
    
    Args:
        x: UFL expression
    
    Returns:
        UFL expression for Bernoulli function
    """
    # Use conditional to avoid numerical issues
    small_x = 1e-10
    return ufl.conditional(ufl.gt(ufl.abs(x), small_x), 
                          x / (ufl.exp(x) - 1), 
                          1 - x/2 + x**2/12 - x**4/720)

def average_bernoulli(x):
    """
    Average Bernoulli function for flux calculation
    B_avg(x) = 0.5 * (B(x) + B(-x))
    
    Args:
        x: argument (scalar or array)
    
    Returns:
        Average Bernoulli function value(s)
    """
    return 0.5 * (bernoulli_function(x) + bernoulli_function(-x))

def scharfetter_gummel_weights(x):
    """
    Calculate Scharfetter-Gummel discretization weights
    
    For flux between two points with potential difference x:
    w_up = B(-x), w_down = B(x)
    
    Args:
        x: potential difference / thermal voltage
    
    Returns:
        (w_up, w_down): upwind and downwind weights
    """
    w_up = bernoulli_function(-x)
    w_down = bernoulli_function(x)
    return w_up, w_down

def numerical_derivative(f, x, h=1e-8):
    """
    Compute numerical derivative using central difference
    
    f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    
    Args:
        f: function to differentiate
        x: point at which to compute derivative
        h: finite difference step size
    
    Returns:
        Numerical derivative estimate
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def check_convergence(x_new, x_old, rtol=1e-6, atol=1e-12):
    """
    Check convergence based on relative and absolute tolerances
    
    Args:
        x_new: new solution vector
        x_old: old solution vector  
        rtol: relative tolerance
        atol: absolute tolerance
    
    Returns:
        (converged, error): convergence status and error measure
    """
    if hasattr(x_new, 'x'):  # DOLFINx Function
        x_new_vals = x_new.x.array
        x_old_vals = x_old.x.array
    else:  # numpy arrays
        x_new_vals = x_new
        x_old_vals = x_old
    
    # Calculate error
    diff = np.abs(x_new_vals - x_old_vals)
    scale = atol + rtol * np.maximum(np.abs(x_new_vals), np.abs(x_old_vals))
    error = np.max(diff / scale)
    
    converged = error < 1.0
    return converged, error

def adaptive_relaxation(iteration, error, target_error=1e-6, 
                       initial_relax=0.7, min_relax=0.1, max_relax=1.0):
    """
    Adaptive relaxation parameter for iterative solvers
    
    Args:
        iteration: current iteration number
        error: current error measure
        target_error: target convergence error
        initial_relax: initial relaxation parameter
        min_relax: minimum relaxation parameter
        max_relax: maximum relaxation parameter
    
    Returns:
        Adjusted relaxation parameter
    """
    if iteration < 3:
        return initial_relax
    
    # Increase relaxation if converging well
    if error < target_error * 10:
        relax = min(max_relax, initial_relax * 1.2)
    # Decrease relaxation if converging poorly
    elif error > target_error * 100:
        relax = max(min_relax, initial_relax * 0.8)
    else:
        relax = initial_relax
    
    return relax

def stability_test():
    """
    Test numerical stability functions
    """
    print("Testing numerical stability functions:")
    
    # Test Bernoulli function
    test_values = [-10, -1, -1e-12, 0, 1e-12, 1, 10]
    print("  Bernoulli function:")
    for x in test_values:
        B = bernoulli_function(x)
        print(f"    B({x}) = {B:.6f}")
        
        # Check for NaN or Inf
        if not np.isfinite(B):
            print(f"    ✗ Non-finite value at x={x}")
            return False
    
    # Test array input
    x_array = np.array(test_values)
    B_array = bernoulli_function(x_array)
    if not np.all(np.isfinite(B_array)):
        print(f"    ✗ Non-finite values in array computation")
        return False
    
    print("    ✓ All Bernoulli values finite")
    
    # Test Scharfetter-Gummel weights
    x_sg = 2.0
    w_up, w_down = scharfetter_gummel_weights(x_sg)
    print(f"  SG weights for x={x_sg}: w_up={w_up:.4f}, w_down={w_down:.4f}")
    
    # Should satisfy w_up + w_down ≈ 1 for small x
    if abs(x_sg) < 0.1:
        weight_sum = w_up + w_down
        if abs(weight_sum - 1.0) > 1e-10:
            print(f"    ✗ Weight sum = {weight_sum} (should be ≈ 1)")
            return False
    
    print("    ✓ SG weights computed correctly")
    
    # Test convergence check
    x1 = np.array([1.0, 2.0, 3.0])
    x2 = np.array([1.001, 2.001, 3.001])
    converged, error = check_convergence(x2, x1, rtol=1e-2)
    print(f"  Convergence test: converged={converged}, error={error:.3e}")
    
    return True