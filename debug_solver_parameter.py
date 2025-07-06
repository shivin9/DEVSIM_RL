#!/usr/bin/env python3
"""
Debug Solver Parameter Issue
Find exactly where the 'unknown' solver parameter is being set
"""

import sys
import os
sys.path.append('/home/shivin/Research/DEVSIM_RL/rl_diode_framework')

def debug_solver_parameter():
    """Debug the solver parameter issue"""
    print("🔍 DEBUG: Solver Parameter Issue")
    print("=" * 50)
    
    # Test 1: Check DEVSIM startup state
    print("\n1. DEVSIM startup state:")
    try:
        from devsim import get_parameter, set_parameter, reset_devsim
        
        # Check what happens at startup
        print("   Checking initial solver parameter...")
        try:
            initial_solver = get_parameter(name="direct_solver")
            print(f"   Initial solver: '{initial_solver}'")
        except Exception as e:
            print(f"   No initial solver parameter: {e}")
        
        # Test reset_devsim
        print("   Testing reset_devsim...")
        reset_devsim()
        try:
            after_reset_solver = get_parameter(name="direct_solver")
            print(f"   After reset solver: '{after_reset_solver}'")
        except Exception as e:
            print(f"   No solver after reset: {e}")
            
    except ImportError:
        print("   DEVSIM not available for testing")
        return
    
    # Test 2: Check if our RL framework code sets it
    print("\n2. Checking DiodeSimulator initialization:")
    try:
        from diode_simulator import DiodeSimulator
        print("   Creating DiodeSimulator...")
        sim = DiodeSimulator(grid_size=4, physical_size=2e-6)
        
        # Check solver parameter after initialization
        try:
            sim_solver = get_parameter(name="direct_solver")
            print(f"   After DiodeSimulator creation: '{sim_solver}'")
        except Exception as e:
            print(f"   No solver parameter after DiodeSimulator: {e}")
            
    except Exception as e:
        print(f"   Error creating DiodeSimulator: {e}")
    
    # Test 3: Check if GMSH code sets it
    print("\n3. Checking GMSH converter:")
    try:
        from matrix_to_gmsh_simple import SimpleMatrixToGMSH
        print("   Creating SimpleMatrixToGMSH...")
        converter = SimpleMatrixToGMSH()
        
        # Check solver parameter after GMSH
        try:
            gmsh_solver = get_parameter(name="direct_solver")
            print(f"   After GMSH creation: '{gmsh_solver}'")
        except Exception as e:
            print(f"   No solver parameter after GMSH: {e}")
            
    except Exception as e:
        print(f"   Error creating GMSH converter: {e}")
    
    # Test 4: Try to recreate the exact error
    print("\n4. Attempting to recreate the exact error:")
    try:
        # Follow the same sequence as the RL framework
        print("   Testing material matrix simulation...")
        
        # Create a simple test matrix
        import numpy as np
        test_matrix = np.ones((4, 4), dtype=np.uint8)
        test_matrix[:, :2] = 2  # P-type left half
        test_matrix[:, 2:] = 1  # N-type right half
        
        print("   Running simulation...")
        result = sim.simulate_diode(test_matrix)
        
        if not result['success']:
            print(f"   ❌ SIMULATION FAILED: {result['error']}")
            if "unknown" in result['error'].lower():
                print("   🎯 FOUND THE 'unknown' SOLVER ERROR!")
        else:
            print("   ✅ Simulation succeeded unexpectedly")
            
    except Exception as e:
        print(f"   Error in simulation test: {e}")
    
    print("\n" + "=" * 50)
    print("Debug complete!")

if __name__ == "__main__":
    debug_solver_parameter()