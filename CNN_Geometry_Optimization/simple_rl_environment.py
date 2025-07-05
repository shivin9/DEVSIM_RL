#!/usr/bin/env python3
"""
Simple RL Environment for 2D Diode Geometry Optimization
Standalone implementation without gym dependencies
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List
from geometry_optimization_framework import GeometryMatrix, GeometryGenerator

class SimpleGeometryEnv:
    """
    Simple RL Environment for optimizing 2D diode geometry
    """
    
    def __init__(self, 
                 width: int = 32, 
                 height: int = 32,
                 max_steps: int = 50):
        """Initialize environment"""
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.current_step = 0
        
        # Initialize geometry
        self.geometry = GeometryMatrix(width, height)
        
        # Performance tracking
        self.best_reward = -np.inf
        self.best_geometry = None
        self.episode_history = []
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment to initial state"""
        self.current_step = 0
        
        # Start with rectangular junction
        self.geometry = GeometryGenerator.create_baseline_rectangular(self.width, self.height)
        
        return self._get_observation()
    
    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """Execute one environment step"""
        self.current_step += 1
        
        # Apply action to geometry
        self._apply_action(action)
        
        # Calculate reward
        reward, reward_info = self._calculate_reward()
        
        # Check episode termination
        done = (self.current_step >= self.max_steps)
        
        # Track best performance
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_geometry = self.geometry.get_material_matrix().copy()
        
        # Episode info
        info = {
            'step': self.current_step,
            'reward_components': reward_info,
            'geometry_metrics': self.geometry.calculate_metrics(),
            'is_best': reward > self.best_reward
        }
        
        return self._get_observation(), reward, done, info
    
    def _apply_action(self, action: Dict[str, float]):
        """Apply action to modify geometry"""
        # Convert normalized coordinates to pixel coordinates
        x = int(action['x'] * self.width)
        y = int(action['y'] * self.height)
        radius = max(1, int(action['radius'] * min(self.width, self.height)))
        material_type = int(action['material'])
        
        # Ensure coordinates are within bounds
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        
        # Apply modification
        self.geometry.apply_modification(x, y, radius, material_type)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation"""
        material_matrix = self.geometry.get_material_matrix()
        features = self._extract_features()
        
        return {
            'material_matrix': material_matrix,
            'features': features
        }
    
    def _extract_features(self) -> np.ndarray:
        """Extract geometry features"""
        metrics = self.geometry.calculate_metrics()
        
        features = np.array([
            metrics['p_fraction'],
            metrics['n_fraction'], 
            metrics['void_fraction'],
            metrics['interface_length'] / 40e-6,  # Normalized
            metrics['connectivity_score'],
            metrics['min_feature_size'] / 1e-6,   # Normalized
            float(metrics['p_components']),
            float(metrics['n_components'])
        ], dtype=np.float32)
        
        return features
    
    def _calculate_reward(self) -> Tuple[float, Dict[str, float]]:
        """Calculate reward based on geometry properties"""
        metrics = self.geometry.calculate_metrics()
        
        # Reward components
        R_interface = self._interface_reward(metrics['interface_length'])
        R_connectivity = self._connectivity_reward(metrics)
        R_manufacturability = self._manufacturability_reward(metrics)
        R_balance = self._material_balance_reward(metrics)
        R_violations = self._constraint_violations(metrics)
        
        # Total reward
        total_reward = (
            0.3 * R_interface +
            0.3 * R_connectivity +
            0.2 * R_manufacturability +
            0.1 * R_balance +
            R_violations
        )
        
        reward_info = {
            'interface': R_interface,
            'connectivity': R_connectivity,
            'manufacturability': R_manufacturability,
            'balance': R_balance,
            'violations': R_violations,
            'total': total_reward
        }
        
        return total_reward, reward_info
    
    def _interface_reward(self, interface_length: float) -> float:
        """Reward for interface length (proxy for current capacity)"""
        # Longer interface = higher current capacity
        # Normalize by typical values
        max_reasonable = 100e-6  # 100 μm
        normalized = min(interface_length / max_reasonable, 1.0)
        return normalized
    
    def _connectivity_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for good connectivity"""
        connectivity_score = metrics['connectivity_score']
        
        # Bonus for single connected components
        if metrics['p_components'] == 1 and metrics['n_components'] == 1:
            return connectivity_score * 1.5
        elif metrics['p_components'] <= 2 and metrics['n_components'] <= 2:
            return connectivity_score * 1.0
        else:
            return connectivity_score * 0.5
    
    def _manufacturability_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for manufacturable geometries"""
        min_feature = metrics['min_feature_size']
        
        # Penalty for too small features
        if min_feature >= 0.5e-6:
            return 1.0
        elif min_feature >= 0.3e-6:
            return 0.7
        else:
            return 0.3
    
    def _material_balance_reward(self, metrics: Dict[str, float]) -> float:
        """Reward for material balance"""
        p_frac = metrics['p_fraction']
        n_frac = metrics['n_fraction']
        void_frac = metrics['void_fraction']
        
        # Prefer balanced P-N, some void is ok
        balance = min(p_frac, n_frac) / max(p_frac, n_frac) if max(p_frac, n_frac) > 0 else 0
        void_penalty = max(0, void_frac - 0.3)  # Penalty for >30% void
        
        return balance - void_penalty
    
    def _constraint_violations(self, metrics: Dict[str, float]) -> float:
        """Penalty for constraint violations"""
        violations = 0.0
        
        # Must have both materials
        if metrics['p_fraction'] < 0.05 or metrics['n_fraction'] < 0.05:
            violations -= 2.0
        
        # Feature size constraint
        if metrics['min_feature_size'] < 0.2e-6:
            violations -= 1.0
        
        return violations
    
    def render(self) -> None:
        """Visualize current geometry"""
        fig = self.geometry.visualize(f"Step {self.current_step}")
        plt.show()
    
    def sample_random_action(self) -> Dict[str, float]:
        """Sample random action for testing"""
        return {
            'x': np.random.uniform(0.1, 0.9),      # Avoid edges for contact preservation
            'y': np.random.uniform(0.0, 1.0),
            'radius': np.random.uniform(0.02, 0.15),
            'material': np.random.choice([0, 1, 2])  # void, N-type, P-type
        }

class RandomAgent:
    """Random agent for testing"""
    
    def __init__(self, env: SimpleGeometryEnv):
        self.env = env
    
    def act(self, observation: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Choose random action"""
        return self.env.sample_random_action()

def test_environment_and_agent():
    """Test environment with random agent"""
    print("Testing Simple RL Environment")
    print("=" * 35)
    
    # Create environment and agent
    env = SimpleGeometryEnv(width=32, height=32, max_steps=15)
    agent = RandomAgent(env)
    
    # Run episode
    obs = env.reset()
    total_reward = 0
    rewards = []
    
    print(f"Initial features: {obs['features']}")
    print(f"Material matrix shape: {obs['material_matrix'].shape}")
    
    for step in range(15):
        # Agent chooses action
        action = agent.act(obs)
        
        # Environment step
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        
        # Log progress
        metrics = info['geometry_metrics']
        reward_components = info['reward_components']
        
        print(f"\nStep {step+1}:")
        print(f"  Action: x={action['x']:.2f}, y={action['y']:.2f}, r={action['radius']:.2f}, mat={action['material']}")
        print(f"  Reward: {reward:.3f} (total: {total_reward:.3f})")
        print(f"  Interface: {metrics['interface_length']*1e6:.1f}μm")
        print(f"  P/N fractions: {metrics['p_fraction']:.2f}/{metrics['n_fraction']:.2f}")
        print(f"  Components: P={metrics['p_components']}, N={metrics['n_components']}")
        
        if info['is_best']:
            print("  *** NEW BEST GEOMETRY ***")
        
        if done:
            break
    
    # Final results
    print(f"\nEpisode completed!")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Best reward: {env.best_reward:.3f}")
    print(f"Average reward: {np.mean(rewards):.3f}")
    
    # Show best geometry
    if env.best_geometry is not None:
        print("\nBest geometry found:")
        temp_geom = GeometryMatrix(env.width, env.height)
        temp_geom.set_material_matrix(env.best_geometry)
        best_metrics = temp_geom.calculate_metrics()
        
        print(f"  Interface length: {best_metrics['interface_length']*1e6:.1f} μm")
        print(f"  P/N fractions: {best_metrics['p_fraction']:.3f}/{best_metrics['n_fraction']:.3f}")
        print(f"  Connectivity: {best_metrics['connectivity_score']:.3f}")
        
        # Visualize best geometry
        fig = temp_geom.visualize("Best Geometry Found")
        plt.savefig('best_geometry_random_agent.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved visualization: best_geometry_random_agent.png")
    
    # Plot reward evolution
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Evolution - Random Agent')
    plt.grid(True, alpha=0.3)
    plt.savefig('reward_evolution_random.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved reward plot: reward_evolution_random.png")

if __name__ == "__main__":
    test_environment_and_agent()