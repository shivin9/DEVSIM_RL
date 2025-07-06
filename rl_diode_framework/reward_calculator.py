#!/usr/bin/env python3
"""
RewardCalculator - Multi-objective Reward Function for RL Diode Design
Combines current, rectification, power, and geometry complexity into RL rewards
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import ndimage

class RewardCalculator:
    """
    Multi-objective reward calculator for diode RL optimization
    
    Objectives:
    1. Maximize forward current (higher is better)
    2. Maximize rectification ratio (higher is better)  
    3. Minimize power consumption (lower is better)
    4. Penalize excessive complexity (balance performance vs manufacturability)
    """
    
    def __init__(self, baseline_performance: Dict, reward_type: str = "sparse"):
        """
        Initialize reward calculator
        
        Args:
            baseline_performance: Performance of baseline rectangular diode
            reward_type: "sparse" (only reward improvements) or "shaped" (continuous feedback)
        """
        self.baseline = baseline_performance
        self.reward_type = reward_type
        
        # Reward weights (relative importance) - Prioritize critical parameters
        self.weights = {
            'current': 10.0,       # Forward current improvement (high priority)
            'rectification': 5.0,  # Rectification ratio improvement (high priority)
            'power': 2.0,          # Power reduction (medium priority)
            'complexity': 0.5,     # Geometry complexity penalty (low priority)
            'area': 1.0            # Footprint area penalty (low priority)
        }
        
        # Reward scaling factors
        self.scale_factors = {
            'current_scale': 1.0,      # Scale current rewards
            'rectification_scale': 0.1, # Scale rectification rewards
            'power_scale': 10.0,       # Scale power rewards
            'complexity_scale': 1.0,   # Scale complexity penalties
            'area_scale': 100.0        # Scale area rewards
        }
        
        # Failure penalties
        self.failure_penalty = -100.0
        self.invalid_penalty = -50.0
        
        # Baseline normalization values
        self.baseline_current = baseline_performance.get('forward_current', 1.0)
        self.baseline_rectification = baseline_performance.get('rectification_ratio', 1e3)
        self.baseline_power = baseline_performance.get('power', 1e-3)
        self.baseline_area = baseline_performance.get('area', 1e-10)  # m²
        
        # Reward history for analysis
        self.reward_history = []
        self.component_history = []
        
        print(f"RewardCalculator initialized:")
        print(f"  Reward type: {reward_type}")
        print(f"  Baseline current: {self.baseline_current:.2e} A")
        print(f"  Baseline rectification: {self.baseline_rectification:.1e}")
        print(f"  Baseline power: {self.baseline_power:.2e} W")
        print(f"  Baseline area: {self.baseline_area*1e12:.1f} μm²")
    
    def calculate_reward(self, simulation_result: Dict, geometry_matrix: np.ndarray, 
                        footprint_area: float) -> Dict:
        """
        Calculate multi-objective reward for RL agent with mandatory improvement constraints
        
        Args:
            simulation_result: Results from DiodeSimulator
            geometry_matrix: Material matrix (for complexity analysis)
            footprint_area: Physical area of the device
            
        Returns:
            reward_dict: Total reward and component breakdown
        """
        # Handle simulation failures
        if not simulation_result['success']:
            return self._create_failure_reward(simulation_result['error'])
        
        # Validate basic geometry requirements
        validation_result = self._validate_geometry(geometry_matrix)
        if not validation_result['valid']:
            return self._create_invalid_reward(validation_result['error'])
        
        # CRITICAL CONSTRAINT CHECK: Forward current and rectification ratio must improve
        current_ratio = simulation_result['forward_current'] / self.baseline_current
        rectification_ratio = simulation_result['rectification_ratio'] / self.baseline_rectification
        
        # If either critical parameter decreases, apply heavy penalty
        if current_ratio < 1.0 or rectification_ratio < 1.0:
            penalty_reward = self._create_constraint_violation_reward(
                current_ratio, rectification_ratio, simulation_result, footprint_area
            )
            # Optional: Print constraint violations for debugging
            if hasattr(self, 'verbose') and self.verbose:
                print(f"⚠️  Constraint violation: Current={current_ratio:.3f}, Rect={rectification_ratio:.3f}")
            return penalty_reward
        
        # Calculate individual reward components (only if critical constraints met)
        current_reward = self._calculate_current_reward(simulation_result['forward_current'])
        rectification_reward = self._calculate_rectification_reward(simulation_result['rectification_ratio'])
        power_reward = self._calculate_power_reward(simulation_result['power'])
        complexity_penalty = self._calculate_complexity_penalty(geometry_matrix)
        area_reward = self._calculate_area_reward(footprint_area)
        
        # Combine components with weights
        total_reward = (
            self.weights['current'] * current_reward +
            self.weights['rectification'] * rectification_reward +
            self.weights['power'] * power_reward +
            self.weights['complexity'] * complexity_penalty +
            self.weights['area'] * area_reward
        )
        
        # Create reward breakdown
        reward_dict = {
            'total_reward': total_reward,
            'success': True,
            'components': {
                'current': current_reward,
                'rectification': rectification_reward,
                'power': power_reward,
                'complexity': complexity_penalty,
                'area': area_reward
            },
            'performance': {
                'forward_current': simulation_result['forward_current'],
                'rectification_ratio': simulation_result['rectification_ratio'],
                'power': simulation_result['power'],
                'area': footprint_area
            },
            'improvements': {
                'current_improvement': (simulation_result['forward_current'] - self.baseline_current) / self.baseline_current,
                'rectification_improvement': (simulation_result['rectification_ratio'] - self.baseline_rectification) / self.baseline_rectification,
                'power_improvement': (self.baseline_power - simulation_result['power']) / self.baseline_power,
                'area_improvement': (self.baseline_area - footprint_area) / self.baseline_area
            }
        }
        
        # Log for analysis
        self.reward_history.append(total_reward)
        self.component_history.append(reward_dict['components'].copy())
        
        return reward_dict
    
    def _create_constraint_violation_reward(self, current_ratio: float, rectification_ratio: float, 
                                          simulation_result: Dict, footprint_area: float) -> Dict:
        """
        Create penalty reward when critical constraints are violated
        
        Args:
            current_ratio: forward_current / baseline_current
            rectification_ratio: rectification_ratio / baseline_rectification
            simulation_result: Simulation results for analysis
            footprint_area: Device area
            
        Returns:
            reward_dict: Penalty reward with detailed breakdown
        """
        # Heavy penalty for constraint violations
        base_penalty = -50.0
        
        # Additional penalties for worse violations
        current_penalty = 0.0
        if current_ratio < 1.0:
            current_penalty = -20.0 * (1.0 - current_ratio)  # Penalty proportional to degradation
        
        rectification_penalty = 0.0
        if rectification_ratio < 1.0:
            rectification_penalty = -10.0 * (1.0 - rectification_ratio)
        
        total_penalty = base_penalty + current_penalty + rectification_penalty
        
        # Create detailed breakdown for analysis
        constraint_violations = []
        if current_ratio < 1.0:
            constraint_violations.append(f"Forward current reduced by {(1.0-current_ratio)*100:.1f}%")
        if rectification_ratio < 1.0:
            constraint_violations.append(f"Rectification ratio reduced by {(1.0-rectification_ratio)*100:.1f}%")
        
        return {
            'total_reward': total_penalty,
            'valid_design': False,
            'constraint_violation': True,
            'violation_details': constraint_violations,
            'components': {
                'constraint_penalty': total_penalty,
                'current_penalty': current_penalty,
                'rectification_penalty': rectification_penalty,
                'current': 0.0,
                'rectification': 0.0,
                'power': 0.0,
                'complexity': 0.0,
                'area': 0.0
            },
            'metrics': {
                'current_ratio': current_ratio,
                'rectification_ratio': rectification_ratio,
                'forward_current': simulation_result['forward_current'],
                'baseline_current': self.baseline_current,
                'rectification_value': simulation_result['rectification_ratio'],
                'baseline_rectification': self.baseline_rectification
            }
        }
    
    def _calculate_current_reward(self, forward_current: float) -> float:
        """
        Calculate reward for forward current performance
        
        CRITICAL: This function should only be called if current_ratio >= 1.0
        (constraint already checked in main calculate_reward function)
        """
        current_ratio = forward_current / self.baseline_current
        
        # Since we've already checked constraints, we know current_ratio >= 1.0
        if self.reward_type == "sparse":
            # Reward for improvement above baseline
            reward = self.scale_factors['current_scale'] * (current_ratio - 1.0)
        else:  # shaped
            # Logarithmic reward for improvements
            reward = self.scale_factors['current_scale'] * np.log(current_ratio)
        
        return max(0.0, reward)  # Ensure non-negative
    
    def _calculate_rectification_reward(self, rectification_ratio: float) -> float:
        """
        Calculate reward for rectification ratio
        
        CRITICAL: This function should only be called if rect_ratio >= 1.0
        (constraint already checked in main calculate_reward function)
        """
        rect_ratio = rectification_ratio / self.baseline_rectification
        
        # Since we've already checked constraints, we know rect_ratio >= 1.0
        if self.reward_type == "sparse":
            # Reward for improvement above baseline
            reward = self.scale_factors['rectification_scale'] * np.log(rect_ratio)
        else:  # shaped
            # Continuous reward for improvements
            reward = self.scale_factors['rectification_scale'] * np.log(rect_ratio)
        
        return max(0.0, reward)  # Ensure non-negative
    
    def _calculate_power_reward(self, power: float) -> float:
        """Calculate reward for power consumption (lower is better)"""
        power_ratio = power / self.baseline_power
        
        if self.reward_type == "sparse":
            # Only reward if better than baseline (lower power)
            if power_ratio < 1.0:
                reward = self.scale_factors['power_scale'] * (1.0 - power_ratio)
            else:
                reward = 0.0
        else:  # shaped
            # Continuous reward (negative log of power ratio)
            reward = -self.scale_factors['power_scale'] * np.log(power_ratio)
        
        return reward
    
    def _calculate_complexity_penalty(self, geometry_matrix: np.ndarray) -> float:
        """Calculate penalty for geometry complexity"""
        # Complexity metrics
        interface_complexity = self._calculate_interface_complexity(geometry_matrix)
        connectivity_penalty = self._calculate_connectivity_penalty(geometry_matrix)
        feature_size_penalty = self._calculate_feature_size_penalty(geometry_matrix)
        
        # Combine complexity measures
        total_complexity = interface_complexity + connectivity_penalty + feature_size_penalty
        
        # Apply penalty (negative reward for high complexity)
        penalty = -self.scale_factors['complexity_scale'] * total_complexity
        
        return penalty
    
    def _calculate_area_reward(self, area: float) -> float:
        """Calculate reward for smaller device area"""
        area_ratio = area / self.baseline_area
        
        if self.reward_type == "sparse":
            # Only reward if smaller than baseline
            if area_ratio < 1.0:
                reward = self.scale_factors['area_scale'] * (1.0 - area_ratio)
            else:
                reward = 0.0
        else:  # shaped
            # Continuous reward (negative log of area ratio)
            reward = -self.scale_factors['area_scale'] * np.log(area_ratio)
        
        return reward
    
    def _calculate_interface_complexity(self, geometry_matrix: np.ndarray) -> float:
        """Calculate P-N interface complexity (length and irregularity)"""
        p_mask = (geometry_matrix == 2)
        n_mask = (geometry_matrix == 1)
        
        # Find interface using morphological operations
        p_dilated = ndimage.binary_dilation(p_mask)
        interface_pixels = p_dilated & n_mask
        interface_length = np.sum(interface_pixels)
        
        # Normalize by device size
        max_interface = min(geometry_matrix.shape) * 2  # Maximum possible interface
        normalized_interface = interface_length / max_interface
        
        return normalized_interface
    
    def _calculate_connectivity_penalty(self, geometry_matrix: np.ndarray) -> float:
        """Penalize disconnected regions"""
        p_mask = (geometry_matrix == 2)
        n_mask = (geometry_matrix == 1)
        
        # Count connected components
        p_labeled, p_components = ndimage.label(p_mask)
        n_labeled, n_components = ndimage.label(n_mask)
        
        # Penalize multiple disconnected regions
        connectivity_penalty = 0.1 * (p_components - 1) + 0.1 * (n_components - 1)
        
        return max(0, connectivity_penalty)
    
    def _calculate_feature_size_penalty(self, geometry_matrix: np.ndarray) -> float:
        """Penalize very small features (manufacturability)"""
        min_feature_size = 2  # Minimum acceptable feature size in pixels
        penalty = 0.0
        
        for material_type in [1, 2]:  # N-type and P-type
            mask = (geometry_matrix == material_type)
            if not np.any(mask):
                continue
            
            # Distance transform to find feature widths
            distance = ndimage.distance_transform_edt(mask)
            if np.any(distance > 0):
                min_width = 2 * np.min(distance[distance > 0])
                if min_width < min_feature_size:
                    penalty += 0.2 * (min_feature_size - min_width) / min_feature_size
        
        return penalty
    
    def _validate_geometry(self, geometry_matrix: np.ndarray) -> Dict:
        """Basic geometry validation for reward calculation"""
        total_pixels = geometry_matrix.size
        
        p_count = np.sum(geometry_matrix == 2)
        n_count = np.sum(geometry_matrix == 1)
        
        p_fraction = p_count / total_pixels
        n_fraction = n_count / total_pixels
        
        # Basic requirements
        has_p_material = p_fraction >= 0.05  # At least 5% P-type
        has_n_material = n_fraction >= 0.05  # At least 5% N-type
        
        # Check P-N interface exists
        p_mask = (geometry_matrix == 2)
        n_mask = (geometry_matrix == 1)
        p_dilated = ndimage.binary_dilation(p_mask)
        has_interface = np.any(p_dilated & n_mask)
        
        valid = has_p_material and has_n_material and has_interface
        
        error_msg = ""
        if not has_p_material:
            error_msg = "Insufficient P-type material"
        elif not has_n_material:
            error_msg = "Insufficient N-type material"
        elif not has_interface:
            error_msg = "No P-N interface"
        
        return {
            'valid': valid,
            'error': error_msg,
            'p_fraction': p_fraction,
            'n_fraction': n_fraction,
            'has_interface': has_interface
        }
    
    def _create_failure_reward(self, error_msg: str) -> Dict:
        """Create reward for simulation failures"""
        return {
            'total_reward': self.failure_penalty,
            'success': False,
            'error': f"Simulation failed: {error_msg}",
            'components': {
                'current': 0.0,
                'rectification': 0.0,
                'power': 0.0,
                'complexity': self.failure_penalty,
                'area': 0.0
            }
        }
    
    def _create_invalid_reward(self, error_msg: str) -> Dict:
        """Create reward for invalid geometries"""
        return {
            'total_reward': self.invalid_penalty,
            'success': False,
            'error': f"Invalid geometry: {error_msg}",
            'components': {
                'current': 0.0,
                'rectification': 0.0,
                'power': 0.0,
                'complexity': self.invalid_penalty,
                'area': 0.0
            }
        }
    
    def update_baseline(self, new_baseline: Dict):
        """Update baseline performance for reward calculation"""
        self.baseline = new_baseline
        self.baseline_current = new_baseline.get('forward_current', self.baseline_current)
        self.baseline_rectification = new_baseline.get('rectification_ratio', self.baseline_rectification)
        self.baseline_power = new_baseline.get('power', self.baseline_power)
        self.baseline_area = new_baseline.get('area', self.baseline_area)
        
        print(f"Baseline updated:")
        print(f"  Current: {self.baseline_current:.2e} A")
        print(f"  Rectification: {self.baseline_rectification:.1e}")
        print(f"  Power: {self.baseline_power:.2e} W")
    
    def get_reward_statistics(self) -> Dict:
        """Get reward calculation statistics"""
        if not self.reward_history:
            return {'no_data': True}
        
        return {
            'total_rewards': len(self.reward_history),
            'mean_reward': np.mean(self.reward_history),
            'std_reward': np.std(self.reward_history),
            'max_reward': np.max(self.reward_history),
            'min_reward': np.min(self.reward_history),
            'recent_mean': np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else np.mean(self.reward_history),
            'improvement_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> float:
        """Calculate reward trend (slope of recent rewards)"""
        if len(self.reward_history) < 5:
            return 0.0
        
        recent_rewards = self.reward_history[-10:]
        x = np.arange(len(recent_rewards))
        
        # Linear regression slope
        slope = np.polyfit(x, recent_rewards, 1)[0]
        return slope
    
    def plot_reward_analysis(self, save_path: str = "reward_analysis.png"):
        """Plot reward evolution and component analysis"""
        if not self.reward_history:
            print("No reward history to plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total reward evolution
        ax1.plot(self.reward_history, 'b-', linewidth=2)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Total Reward')
        ax1.set_title('Reward Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(self.reward_history) >= 5:
            x = np.arange(len(self.reward_history))
            z = np.polyfit(x, self.reward_history, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), 'r--', alpha=0.7, label=f'Trend: {z[0]:.3f}')
            ax1.legend()
        
        # Component breakdown (recent episodes)
        if self.component_history:
            recent_components = self.component_history[-20:]  # Last 20 episodes
            components = ['current', 'rectification', 'power', 'complexity', 'area']
            
            component_means = []
            for comp in components:
                values = [ep[comp] for ep in recent_components if comp in ep]
                component_means.append(np.mean(values) if values else 0)
            
            bars = ax2.bar(components, component_means, alpha=0.7)
            ax2.set_ylabel('Average Reward Component')
            ax2.set_title('Recent Component Breakdown')
            ax2.tick_params(axis='x', rotation=45)
            
            # Color bars based on positive/negative
            for i, bar in enumerate(bars):
                if component_means[i] < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
        
        # Reward distribution
        ax3.hist(self.reward_history, bins=20, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Reward Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reward Distribution')
        ax3.axvline(np.mean(self.reward_history), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.reward_history):.2f}')
        ax3.legend()
        
        # Recent performance (moving average)
        if len(self.reward_history) >= 10:
            window_size = min(10, len(self.reward_history) // 4)
            moving_avg = []
            for i in range(window_size, len(self.reward_history) + 1):
                moving_avg.append(np.mean(self.reward_history[i-window_size:i]))
            
            ax4.plot(range(window_size, len(self.reward_history) + 1), moving_avg, 'g-', linewidth=2)
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Moving Average Reward')
            ax4.set_title(f'Moving Average (window={window_size})')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Reward analysis plot saved to {save_path}")


def test_reward_calculator():
    """Test RewardCalculator functionality"""
    print("Testing RewardCalculator")
    print("=" * 50)
    
    # Mock baseline performance
    baseline_performance = {
        'forward_current': 1.5e-3,      # 1.5 mA
        'rectification_ratio': 1e4,     # 10,000
        'power': 1.0e-3,                # 1 mW
        'area': 100e-12                 # 100 μm²
    }
    
    # Create reward calculator
    reward_calc = RewardCalculator(baseline_performance, reward_type="sparse")
    
    # Test 1: Successful simulation with improvements
    print("\nTest 1: Improved performance")
    improved_sim = {
        'success': True,
        'forward_current': 2.0e-3,      # 33% better current
        'rectification_ratio': 1.5e4,   # 50% better rectification
        'power': 0.8e-3,                # 20% less power
    }
    
    # Create simple geometry matrix
    geometry = np.ones((8, 8), dtype=np.uint8)
    geometry[:, :4] = 2  # P-type left half
    geometry[:, 4:] = 1  # N-type right half
    
    reward_result = reward_calc.calculate_reward(improved_sim, geometry, 80e-12)  # 20% smaller area
    
    print(f"  Total reward: {reward_result['total_reward']:.3f}")
    print(f"  Components: {reward_result['components']}")
    print(f"  Improvements: {reward_result['improvements']}")
    
    # Test 2: Failed simulation
    print("\nTest 2: Failed simulation")
    failed_sim = {
        'success': False,
        'error': 'Convergence failure'
    }
    
    failed_reward = reward_calc.calculate_reward(failed_sim, geometry, 80e-12)
    print(f"  Total reward: {failed_reward['total_reward']:.3f}")
    print(f"  Error: {failed_reward['error']}")
    
    # Test 3: Invalid geometry (no P-type material)
    print("\nTest 3: Invalid geometry")
    invalid_geometry = np.ones((8, 8), dtype=np.uint8)  # All N-type
    
    valid_sim = {
        'success': True,
        'forward_current': 1.0e-3,
        'rectification_ratio': 5e3,
        'power': 1.2e-3,
    }
    
    invalid_reward = reward_calc.calculate_reward(valid_sim, invalid_geometry, 100e-12)
    print(f"  Total reward: {invalid_reward['total_reward']:.3f}")
    print(f"  Error: {invalid_reward['error']}")
    
    # Test 4: Complex geometry penalty
    print("\nTest 4: Complex geometry")
    complex_geometry = np.zeros((8, 8), dtype=np.uint8)
    # Create checkerboard pattern (high complexity)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                complex_geometry[i, j] = 2  # P-type
            else:
                complex_geometry[i, j] = 1  # N-type
    
    # Add some valid regions to pass basic validation
    complex_geometry[:2, :] = 2  # Ensure sufficient P-type
    complex_geometry[-2:, :] = 1  # Ensure sufficient N-type
    
    complex_reward = reward_calc.calculate_reward(improved_sim, complex_geometry, 80e-12)
    print(f"  Total reward: {complex_reward['total_reward']:.3f}")
    print(f"  Complexity penalty: {complex_reward['components']['complexity']:.3f}")
    
    # Test 5: Statistics
    print("\nTest 5: Statistics")
    # Add more reward samples
    for _ in range(10):
        test_sim = {
            'success': True,
            'forward_current': np.random.normal(1.5e-3, 0.3e-3),
            'rectification_ratio': np.random.normal(1e4, 2e3),
            'power': np.random.normal(1.0e-3, 0.2e-3),
        }
        reward_calc.calculate_reward(test_sim, geometry, np.random.normal(100e-12, 20e-12))
    
    stats = reward_calc.get_reward_statistics()
    print(f"  Statistics: {stats}")
    
    # Test plot (if running interactively)
    reward_calc.plot_reward_analysis("test_reward_analysis.png")
    
    print(f"\nRewardCalculator test complete!")
    return reward_calc


if __name__ == "__main__":
    test_reward_calculator()