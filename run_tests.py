#!/usr/bin/env python3
"""
Simple test runner for the drift-diffusion solver

Usage: python run_tests.py [--quick]
"""

import sys
import os
import subprocess
import argparse

def run_tests(quick=False):
    """Run the test suite"""
    
    print("DEVSIM DRIFT-DIFFUSION SOLVER - TEST RUNNER")
    print("="*50)
    
    if quick:
        print("Running quick tests only...")
    else:
        print("Running full test suite...")
    
    # Check if test file exists
    test_file = "test_devsim_solver.py"
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return 1
    
    # Check if main solver file exists
    solver_file = "devsim_diode_working.py"
    if not os.path.exists(solver_file):
        print(f"Error: {solver_file} not found!")
        return 1
    
    try:
        # Run the tests
        if quick:
            # For quick tests, we could modify to run only essential tests
            # For now, run all tests but suppress verbose output
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True, timeout=120)
        else:
            # Run with full output
            result = subprocess.run([sys.executable, test_file], timeout=300)
        
        if result.returncode == 0:
            print("\n✓ ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n✗ TESTS FAILED (exit code: {result.returncode})")
            if quick and result.stderr:
                print("Error output:")
                print(result.stderr)
            return result.returncode
            
    except subprocess.TimeoutExpired:
        print("\n✗ TESTS TIMED OUT")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR RUNNING TESTS: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="Run drift-diffusion solver tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick tests with minimal output")
    
    args = parser.parse_args()
    
    exit_code = run_tests(quick=args.quick)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()