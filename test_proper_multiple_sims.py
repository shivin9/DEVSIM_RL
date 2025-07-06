#!/usr/bin/env python3
"""
Test Proper Multiple Simulations
Quick test with smaller devices to verify state management
"""

import numpy as np
import sys
import os
sys.path.append('/home/shivin/Research/DEVSIM_RL/rl_diode_framework')

def test_minimal_multiple_sims():
    """Test multiple simulations with minimal complexity"""
    print("🔄 TESTING MINIMAL MULTIPLE SIMULATIONS")
    print("=" * 60)
    
    from fixed_diode_simulator import FixedDiodeSimulator
    
    # Create very small simulator for faster testing
    simulator = FixedDiodeSimulator(grid_size=3, physical_size=1e-6)
    
    # Create simple test matrices
    matrices = []
    
    # Matrix 1: Simple P-N junction
    matrix1 = np.array([[2, 2, 1],
                        [2, 2, 1], 
                        [2, 2, 1]], dtype=np.uint8)
    matrices.append(("P-N Simple", matrix1))
    
    # Matrix 2: Different P-N ratio  
    matrix2 = np.array([[2, 1, 1],
                        [2, 1, 1],
                        [2, 1, 1]], dtype=np.uint8)
    matrices.append(("P-N Different", matrix2))
    
    results = []
    
    for i, (name, matrix) in enumerate(matrices):
        print(f"\n{i+1}. Testing '{name}':")
        print(f"   Matrix: {matrix[0]}")
        print(f"   P-type: {np.sum(matrix == 2)}, N-type: {np.sum(matrix == 1)}")
        
        try:
            # Check DEVSIM state before simulation
            from devsim import get_device_list, get_mesh_list
            print(f"   Before sim - Devices: {len(get_device_list())}, Meshes: {len(get_mesh_list())}")
            
            result = simulator.simulate_diode(matrix)
            
            # Check DEVSIM state after simulation
            print(f"   After sim  - Devices: {len(get_device_list())}, Meshes: {len(get_mesh_list())}")
            
            if result['success']:
                print(f"   ✅ SUCCESS!")
                print(f"   Forward: {result['forward_current']:.2e} A")
                print(f"   Time: {result['simulation_time']:.1f}s")
                print(f"   Cache: {result.get('from_cache', False)}")
                results.append(True)
            else:
                print(f"   ❌ FAILED: {result['error']}")
                results.append(False)
                
        except Exception as e:
            print(f"   💥 EXCEPTION: {e}")
            results.append(False)
    
    # Test same matrix again for caching
    print(f"\n3. Re-testing first matrix (cache test):")
    try:
        result = simulator.simulate_diode(matrices[0][1])
        if result['success']:
            print(f"   ✅ Cache test: {result.get('from_cache', False)}")
        else:
            print(f"   ❌ Cache test failed: {result['error']}")
    except Exception as e:
        print(f"   💥 Cache test exception: {e}")
    
    # Check final state
    try:
        from devsim import get_device_list, get_mesh_list, get_parameter
        print(f"\n4. Final DEVSIM state:")
        print(f"   Devices: {get_device_list()}")
        print(f"   Meshes: {get_mesh_list()}")
        
        solver = get_parameter(name="direct_solver")
        print(f"   Solver: '{solver}'")
    except Exception as e:
        print(f"   State check error: {e}")
    
    # Cleanup
    simulator.cleanup_all()
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n5. Results: {sum(results)}/{len(results)} successful ({success_rate:.0f}%)")
    
    return success_rate == 100

if __name__ == "__main__":
    success = test_minimal_multiple_sims()
    if success:
        print(f"\n🎊 MULTIPLE SIMULATION SUCCESS!")
    else:
        print(f"\n❌ MULTIPLE SIMULATION ISSUES!")