#!/usr/bin/env python3
"""
Test Multiple Simulations - Check if fixed simulator handles multiple runs correctly
"""

import numpy as np
import sys
import os
sys.path.append('/home/shivin/Research/DEVSIM_RL/rl_diode_framework')

from fixed_diode_simulator import FixedDiodeSimulator

def test_multiple_simulations():
    """Test multiple consecutive simulations"""
    print("🔄 TESTING MULTIPLE SIMULATIONS")
    print("=" * 60)
    
    # Create different test matrices
    test_matrices = []
    
    # Test 1: Normal P-N junction
    matrix1 = np.ones((4, 4), dtype=np.uint8)
    matrix1[:, :2] = 2  # P-type (left half)
    matrix1[:, 2:] = 1  # N-type (right half)
    test_matrices.append(("Normal P-N", matrix1))
    
    # Test 2: Different P-N ratio
    matrix2 = np.ones((4, 4), dtype=np.uint8)
    matrix2[:, :1] = 2  # P-type (left quarter)
    matrix2[:, 1:] = 1  # N-type (rest)
    test_matrices.append(("Small P region", matrix2))
    
    # Test 3: Another different geometry
    matrix3 = np.ones((4, 4), dtype=np.uint8)
    matrix3[:, :3] = 2  # P-type (left 3/4)
    matrix3[:, 3:] = 1  # N-type (right quarter)
    test_matrices.append(("Large P region", matrix3))
    
    print(f"Testing {len(test_matrices)} different geometries...")
    
    # Create simulator
    simulator = FixedDiodeSimulator(grid_size=4, physical_size=2e-6)
    
    results = []
    
    for i, (name, matrix) in enumerate(test_matrices):
        print(f"\n{i+1}. Testing '{name}':")
        print(f"   P-type pixels: {np.sum(matrix == 2)}")
        print(f"   N-type pixels: {np.sum(matrix == 1)}")
        
        try:
            result = simulator.simulate_diode(matrix)
            
            if result['success']:
                print(f"   ✅ SUCCESS!")
                print(f"   Forward current: {result['forward_current']:.2e} A")
                print(f"   Reverse current: {result['reverse_current']:.2e} A")
                print(f"   Power: {result['power']:.2e} W")
                print(f"   Rectification: {result['rectification_ratio']:.1e}")
                print(f"   Time: {result['simulation_time']:.1f}s")
                print(f"   From cache: {result.get('from_cache', False)}")
                
                results.append({
                    'name': name,
                    'success': True,
                    'forward_current': result['forward_current'],
                    'reverse_current': result['reverse_current'],
                    'power': result['power']
                })
            else:
                print(f"   ❌ FAILED: {result['error']}")
                results.append({
                    'name': name,
                    'success': False,
                    'error': result['error']
                })
                
        except Exception as e:
            print(f"   💥 EXCEPTION: {e}")
            results.append({
                'name': name,
                'success': False,
                'error': str(e)
            })
    
    # Test the same matrix again to check caching
    print(f"\n4. Re-testing first matrix (should use cache):")
    try:
        result = simulator.simulate_diode(test_matrices[0][1])
        if result['success']:
            print(f"   ✅ SUCCESS! From cache: {result.get('from_cache', False)}")
        else:
            print(f"   ❌ FAILED: {result['error']}")
    except Exception as e:
        print(f"   💥 EXCEPTION: {e}")
    
    # Get simulator statistics
    stats = simulator.get_simulation_stats()
    print(f"\n5. Simulator Statistics:")
    print(f"   Total simulations: {stats['simulation_count']}")
    print(f"   Cache size: {stats['cache_size']}")
    print(f"   Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"   Active devices: {stats['active_devices']}")
    print(f"   Active meshes: {stats['active_meshes']}")
    
    # Summary
    successes = sum(1 for r in results if r['success'])
    print(f"\n6. Results Summary:")
    print(f"   Successful simulations: {successes}/{len(results)}")
    print(f"   Success rate: {successes/len(results)*100:.1f}%")
    
    if successes == len(results):
        print(f"   🎉 ALL SIMULATIONS SUCCESSFUL!")
        
        # Check if results are different (they should be for different geometries)
        forward_currents = [r['forward_current'] for r in results if r['success']]
        if len(set([f"{c:.2e}" for c in forward_currents])) > 1:
            print(f"   ✅ Results vary correctly for different geometries")
        else:
            print(f"   ⚠️  Warning: All geometries gave same results")
            
    else:
        print(f"   ❌ Some simulations failed")
        failed = [r for r in results if not r['success']]
        for f in failed:
            print(f"     - {f['name']}: {f['error']}")
    
    # Cleanup
    simulator.cleanup_all()
    
    return successes == len(results)

if __name__ == "__main__":
    success = test_multiple_simulations()
    if success:
        print(f"\n🎊 MULTIPLE SIMULATION TEST: SUCCESS!")
        print(f"Fixed simulator handles multiple runs correctly")
    else:
        print(f"\n❌ MULTIPLE SIMULATION TEST: ISSUES DETECTED")
        print(f"Fixed simulator may have problems with consecutive runs")