#!/usr/bin/env python3
"""
Structure Performance Analysis - Compare different diode geometries
Tests random, honeycomb, banded, and standard P-N junction structures
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from typing import Dict, List, Tuple
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from diode_simulator import DiodeSimulator
from reward_calculator import RewardCalculator

class StructureGenerator:
    """Generate different diode structure geometries for testing"""
    
    def __init__(self, grid_size: int = 12):
        self.grid_size = grid_size
        
    def create_standard_pn_junction(self) -> np.ndarray:
        """Create standard P-N junction (left P, right N)"""
        matrix = np.ones((self.grid_size, self.grid_size), dtype=np.uint8)
        junction_pos = self.grid_size // 2
        matrix[:, :junction_pos] = 2  # P-type (left)
        matrix[:, junction_pos:] = 1  # N-type (right)
        return matrix
    
    def create_random_structure(self, seed: int = 42) -> np.ndarray:
        """Create random structure with equal P/N distribution"""
        np.random.seed(seed)
        matrix = np.random.choice([1, 2], size=(self.grid_size, self.grid_size), p=[0.5, 0.5])
        return matrix.astype(np.uint8)
    
    def create_honeycomb_structure(self) -> np.ndarray:
        """Create honeycomb pattern with alternating P/N hexagonal cells"""
        matrix = np.ones((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Create honeycomb pattern - simplified version for grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Create hexagonal pattern based on position
                hex_x = j // 3
                hex_y = i // 3
                
                # Alternate P/N based on hexagonal cell position
                if (hex_x + hex_y) % 2 == 0:
                    matrix[i, j] = 2  # P-type
                else:
                    matrix[i, j] = 1  # N-type
                    
        return matrix
    
    def create_banded_structure(self, band_width: int = 2) -> np.ndarray:
        """Create alternating bands of P and N materials"""
        matrix = np.ones((self.grid_size, self.grid_size), dtype=np.uint8)
        
        # Create horizontal bands
        for i in range(self.grid_size):
            band_number = i // band_width
            if band_number % 2 == 0:
                matrix[i, :] = 2  # P-type band
            else:
                matrix[i, :] = 1  # N-type band
                
        return matrix
    
    def create_interdigitated_structure(self) -> np.ndarray:
        """Create interdigitated structure with vertical fingers"""
        matrix = np.ones((self.grid_size, self.grid_size), dtype=np.uint8)
        finger_width = 2
        
        # Create vertical fingers
        for j in range(self.grid_size):
            finger_number = j // finger_width
            if finger_number % 2 == 0:
                matrix[:, j] = 2  # P-type finger
            else:
                matrix[:, j] = 1  # N-type finger
                
        return matrix

class StructurePerformanceAnalyzer:
    """Analyze and compare performance of different diode structures"""
    
    def __init__(self, grid_size: int = 12, physical_size: float = 6e-6):
        self.grid_size = grid_size
        self.physical_size = physical_size
        self.generator = StructureGenerator(grid_size)
        self.simulator = DiodeSimulator(grid_size, physical_size)
        
        # Results storage
        self.results = {}
        self.structures = {}
        
        print(f"StructurePerformanceAnalyzer initialized:")
        print(f"  Grid size: {grid_size}×{grid_size}")
        print(f"  Physical size: {physical_size*1e6:.1f} μm")
    
    def test_all_structures(self) -> Dict:
        """Test all structure types and compare performance"""
        print("\n" + "="*60)
        print("DIODE STRUCTURE PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Define structures to test
        structure_tests = {
            'standard_pn': self.generator.create_standard_pn_junction,
            'random': self.generator.create_random_structure,
            'honeycomb': self.generator.create_honeycomb_structure,
            'banded': self.generator.create_banded_structure,
            'interdigitated': self.generator.create_interdigitated_structure
        }
        
        # Test each structure
        for name, generator_func in structure_tests.items():
            print(f"\nTesting {name.upper()} structure...")
            
            # Generate structure
            structure = generator_func()
            self.structures[name] = structure
            
            # Validate structure
            validation = self.simulator.validate_material_matrix(structure)
            print(f"  Validation: {validation['valid']} - {validation.get('error_message', 'OK')}")
            
            if validation['valid']:
                # Simulate performance
                start_time = time.time()
                result = self.simulator.simulate_diode(structure)
                sim_time = time.time() - start_time
                
                if result['success']:
                    self.results[name] = {
                        'forward_current': result['forward_current'],
                        'reverse_current': result['reverse_current'],
                        'rectification_ratio': result['rectification_ratio'],
                        'power': result['power'],
                        'area': self.physical_size ** 2,
                        'simulation_time': sim_time,
                        'validation': validation,
                        'success': True
                    }
                    print(f"  ✓ Simulation successful ({sim_time:.3f}s)")
                else:
                    self.results[name] = {
                        'success': False,
                        'error': result['error'],
                        'validation': validation
                    }
                    print(f"  ✗ Simulation failed: {result['error']}")
            else:
                self.results[name] = {
                    'success': False,
                    'error': f"Validation failed: {validation['error_message']}",
                    'validation': validation
                }
                print(f"  ✗ Structure invalid: {validation['error_message']}")
        
        return self.results
    
    def analyze_performance(self) -> Dict:
        """Analyze and compare performance metrics"""
        if not self.results:
            print("No results to analyze. Run test_all_structures() first.")
            return {}
        
        print(f"\n" + "="*60)
        print("PERFORMANCE COMPARISON ANALYSIS")
        print("="*60)
        
        # Get baseline (standard P-N junction)
        baseline = self.results.get('standard_pn', {})
        if not baseline.get('success', False):
            print("ERROR: Baseline standard P-N junction failed. Cannot compare.")
            return {}
        
        analysis = {
            'baseline': baseline,
            'comparisons': {},
            'summary': {}
        }
        
        # Compare each structure to baseline
        for name, result in self.results.items():
            if name == 'standard_pn' or not result.get('success', False):
                continue
                
            comparison = self._compare_to_baseline(result, baseline, name)
            analysis['comparisons'][name] = comparison
        
        # Generate summary
        analysis['summary'] = self._generate_summary(analysis['comparisons'])
        
        return analysis
    
    def _compare_to_baseline(self, result: Dict, baseline: Dict, name: str) -> Dict:
        """Compare a structure's performance to baseline"""
        comparison = {
            'structure_name': name,
            'absolute_values': {
                'forward_current': result['forward_current'],
                'reverse_current': result['reverse_current'],
                'rectification_ratio': result['rectification_ratio'],
                'power': result['power'],
                'area': result['area']
            },
            'improvements': {},
            'percentage_changes': {}
        }
        
        # Calculate improvements and percentage changes
        metrics = ['forward_current', 'rectification_ratio', 'power']
        for metric in metrics:
            baseline_val = baseline[metric]
            current_val = result[metric]
            
            if baseline_val != 0:
                improvement = current_val - baseline_val
                percentage = (improvement / baseline_val) * 100
                
                comparison['improvements'][metric] = improvement
                comparison['percentage_changes'][metric] = percentage
        
        # Special handling for reverse current (lower magnitude is better)
        baseline_rev = abs(baseline['reverse_current'])
        current_rev = abs(result['reverse_current'])
        rev_improvement = baseline_rev - current_rev  # Positive = better (lower leakage)
        rev_percentage = (rev_improvement / baseline_rev) * 100 if baseline_rev != 0 else 0
        
        comparison['improvements']['reverse_current'] = rev_improvement
        comparison['percentage_changes']['reverse_current'] = rev_percentage
        
        return comparison
    
    def _generate_summary(self, comparisons: Dict) -> Dict:
        """Generate summary of best and worst performing structures"""
        if not comparisons:
            return {}
        
        summary = {
            'best_performers': {},
            'worst_performers': {},
            'overall_ranking': []
        }
        
        metrics = ['forward_current', 'reverse_current', 'rectification_ratio', 'power']
        
        # Find best and worst for each metric
        for metric in metrics:
            best_name = max(comparisons.keys(), 
                          key=lambda x: comparisons[x]['percentage_changes'][metric])
            worst_name = min(comparisons.keys(), 
                           key=lambda x: comparisons[x]['percentage_changes'][metric])
            
            summary['best_performers'][metric] = {
                'structure': best_name,
                'improvement': comparisons[best_name]['percentage_changes'][metric]
            }
            summary['worst_performers'][metric] = {
                'structure': worst_name,
                'degradation': comparisons[worst_name]['percentage_changes'][metric]
            }
        
        # Calculate overall performance score
        structure_scores = {}
        for name, comp in comparisons.items():
            # Weighted score (prioritize rectification and forward current)
            score = (
                comp['percentage_changes']['forward_current'] * 0.3 +
                comp['percentage_changes']['rectification_ratio'] * 0.4 +
                comp['percentage_changes']['power'] * 0.2 +
                comp['percentage_changes']['reverse_current'] * 0.1
            )
            structure_scores[name] = score
        
        # Rank structures by overall score
        ranked = sorted(structure_scores.items(), key=lambda x: x[1], reverse=True)
        summary['overall_ranking'] = ranked
        
        return summary
    
    def print_detailed_results(self, analysis: Dict):
        """Print detailed analysis results"""
        if not analysis:
            print("No analysis results available.")
            return
        
        baseline = analysis['baseline']
        comparisons = analysis['comparisons']
        summary = analysis['summary']
        
        print(f"\nBASELINE PERFORMANCE (Standard P-N Junction):")
        print(f"  Forward Current: {baseline['forward_current']:.3e} A")
        print(f"  Reverse Current: {baseline['reverse_current']:.3e} A")
        print(f"  Rectification Ratio: {baseline['rectification_ratio']:.2e}")
        print(f"  Power: {baseline['power']:.3e} W")
        print(f"  Area: {baseline['area']*1e12:.1f} μm²")
        
        print(f"\nDETAILED COMPARISONS:")
        print("-" * 80)
        print(f"{'Structure':<15} {'Forward I':<12} {'Reverse I':<12} {'Rectification':<15} {'Power':<12}")
        print(f"{'Name':<15} {'(% change)':<12} {'(% change)':<12} {'(% change)':<15} {'(% change)':<12}")
        print("-" * 80)
        
        for name, comp in comparisons.items():
            pc = comp['percentage_changes']
            print(f"{name:<15} {pc['forward_current']:>+8.2f}%    {pc['reverse_current']:>+8.2f}%    "
                  f"{pc['rectification_ratio']:>+10.2f}%    {pc['power']:>+8.2f}%")
        
        print(f"\nOVERALL RANKING (Best to Worst):")
        for i, (name, score) in enumerate(summary['overall_ranking'], 1):
            print(f"  {i}. {name:<15} (Score: {score:+6.2f}%)")
        
        print(f"\nBEST PERFORMERS BY METRIC:")
        for metric, data in summary['best_performers'].items():
            print(f"  {metric:<20}: {data['structure']:<15} ({data['improvement']:+6.2f}%)")
    
    def save_results(self, filename: str = "structure_performance_results.txt"):
        """Save results to file"""
        if not self.results:
            print("No results to save.")
            return
        
        with open(filename, 'w') as f:
            f.write("DIODE STRUCTURE PERFORMANCE ANALYSIS RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            # Write detailed results for each structure
            for name, result in self.results.items():
                f.write(f"{name.upper()} STRUCTURE:\n")
                if result.get('success', False):
                    f.write(f"  Forward Current: {result['forward_current']:.6e} A\n")
                    f.write(f"  Reverse Current: {result['reverse_current']:.6e} A\n")
                    f.write(f"  Rectification Ratio: {result['rectification_ratio']:.6e}\n")
                    f.write(f"  Power: {result['power']:.6e} W\n")
                    f.write(f"  Area: {result['area']*1e12:.1f} μm²\n")
                    f.write(f"  Simulation Time: {result['simulation_time']:.3f} s\n")
                else:
                    f.write(f"  FAILED: {result['error']}\n")
                f.write("\n")
        
        print(f"Results saved to {filename}")

def main():
    """Main function to run structure performance analysis"""
    print("Starting Diode Structure Performance Analysis...")
    
    # Initialize analyzer
    analyzer = StructurePerformanceAnalyzer(grid_size=12, physical_size=6e-6)
    
    # Test all structures
    results = analyzer.test_all_structures()
    
    # Analyze performance
    analysis = analyzer.analyze_performance()
    
    # Print results
    analyzer.print_detailed_results(analysis)
    
    # Save results
    analyzer.save_results()
    
    print(f"\nAnalysis complete! Tested {len(results)} structures.")
    return analyzer, results, analysis

if __name__ == "__main__":
    analyzer, results, analysis = main()