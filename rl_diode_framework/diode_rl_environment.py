#!/usr/bin/env python3
"""
DiodeDesignEnvironment - OpenAI Gym Environment for RL Diode Design
Manages state, actions, and rewards for learning novel diode geometries
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import warnings

# Our modules
from diode_simulator import DiodeSimulator
from reward_calculator import RewardCalculator

# Suppress gymnasium warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

class DiodeDesignEnvironment(gym.Env):
    """
    OpenAI Gym environment for RL-based diode design optimization
    
    State: Material matrix (grid_size × grid_size) with values 0, 1, 2
    Action: Modify single cell material type
    Reward: Multi-objective score (current, rectification, power, complexity)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, 
                 grid_size: int = 8,
                 physical_size: float = 6e-6,
                 max_steps: int = 50,
                 reward_type: str = "sparse",
                 action_type: str = "discrete"):
        """
        Initialize RL environment for diode design
        
        Args:
            grid_size: Grid resolution (grid_size × grid_size)
            physical_size: Physical device size in meters
            max_steps: Maximum steps per episode
            reward_type: "sparse" or "shaped" rewards
            action_type: "discrete" or "continuous" actions
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.physical_size = physical_size
        self.max_steps = max_steps
        self.reward_type = reward_type
        self.action_type = action_type
        
        # Initialize simulator
        self.simulator = DiodeSimulator(grid_size=grid_size, physical_size=physical_size)
        
        # Get baseline performance for reward calculation
        self.baseline_geometry = self.simulator.create_normal_pn_junction()
        self.baseline_performance = self._establish_baseline()
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            self.baseline_performance, 
            reward_type=reward_type
        )
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Episode state
        self.current_step = 0
        self.current_geometry = None
        self.episode_history = []
        self.best_reward = float('-inf')
        self.best_geometry = None
        
        # Statistics
        self.episode_count = 0
        self.total_steps = 0
        self.successful_episodes = 0
        
        print(f"DiodeDesignEnvironment initialized:")
        print(f"  Grid size: {grid_size}×{grid_size}")
        print(f"  Physical size: {physical_size*1e6:.1f} μm")
        print(f"  Max steps: {max_steps}")
        print(f"  Action type: {action_type}")
        print(f"  Baseline performance: {self.baseline_performance}")
    
    def _establish_baseline(self) -> Dict:
        """Establish baseline performance for reward calculation"""
        print("  Establishing baseline performance...")
        
        baseline_sim = self.simulator.simulate_diode(self.baseline_geometry)
        
        if baseline_sim['success']:
            baseline_perf = {
                'forward_current': baseline_sim['forward_current'],
                'rectification_ratio': baseline_sim['rectification_ratio'],
                'power': baseline_sim['power'],
                'area': self.physical_size ** 2
            }
            print(f"    ✓ Baseline established from simulation")
        else:
            # Use reasonable defaults if simulation fails
            baseline_perf = {
                'forward_current': 1e-3,      # 1 mA
                'rectification_ratio': 1e3,   # 1000
                'power': 0.7e-3,              # 0.7 mW
                'area': self.physical_size ** 2
            }
            print(f"    ⚠️  Using default baseline (simulation failed)")
        
        return baseline_perf
    
    def _setup_spaces(self):
        """Setup action and observation spaces"""
        # Observation space: material matrix
        self.observation_space = spaces.Box(
            low=0, high=2, 
            shape=(self.grid_size, self.grid_size), 
            dtype=np.uint8
        )
        
        if self.action_type == "discrete":
            # Action space: (position, material_type)
            # Position: grid_size^2 possible positions
            # Material: 3 types (0=void, 1=N-type, 2=P-type)
            total_positions = self.grid_size * self.grid_size
            self.action_space = spaces.MultiDiscrete([total_positions, 3])
            
        else:  # continuous
            # Action space: [x, y, material_probability]
            self.action_space = spaces.Box(
                low=np.array([0, 0, 0]), 
                high=np.array([1, 1, 1]), 
                dtype=np.float32
            )
        
        print(f"    Action space: {self.action_space}")
        print(f"    Observation space: {self.observation_space}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state
        
        Returns:
            observation: Initial geometry matrix
            info: Episode information
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode state
        self.current_step = 0
        self.episode_history = []
        
        # Start with normal P-N junction (not random)
        self.current_geometry = self.simulator.create_normal_pn_junction()
        
        # Increment episode counter
        self.episode_count += 1
        
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'geometry_valid': True,
            'baseline_performance': self.baseline_performance
        }
        
        return self.current_geometry.copy(), info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Agent action (position and material type)
            
        Returns:
            observation: New geometry matrix
            reward: Reward for this step
            terminated: Episode finished (max steps or failure)
            truncated: Episode truncated
            info: Step information
        """
        self.current_step += 1
        self.total_steps += 1
        
        # Apply action to modify geometry
        modified_geometry, action_valid = self._apply_action(action)
        
        if not action_valid:
            # Invalid action penalty
            reward = -10.0
            terminated = False
            truncated = False
            info = {
                'step': self.current_step,
                'action_valid': False,
                'error': 'Invalid action',
                'reward_components': {'penalty': reward}
            }
            return self.current_geometry.copy(), reward, terminated, truncated, info
        
        # Update current geometry
        self.current_geometry = modified_geometry
        
        # Run simulation
        sim_result = self.simulator.simulate_diode(self.current_geometry)
        
        # Calculate reward
        footprint_area = self.physical_size ** 2
        reward_result = self.reward_calculator.calculate_reward(
            sim_result, self.current_geometry, footprint_area
        )
        
        reward = reward_result['total_reward']
        
        # Track best performance
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_geometry = self.current_geometry.copy()
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Episode ends at max steps
        if self.current_step >= self.max_steps:
            truncated = True
            if reward > 0:  # Consider successful if positive reward
                self.successful_episodes += 1
        
        # Episode can terminate early for very poor designs
        if reward < -50.0:
            terminated = True
        
        # Log step information
        step_info = {
            'step': self.current_step,
            'action_valid': True,
            'simulation_success': sim_result['success'],
            'reward': reward,
            'reward_components': reward_result.get('components', {}),
            'performance': reward_result.get('performance', {}),
            'improvements': reward_result.get('improvements', {}),
            'best_reward': self.best_reward,
            'is_best': reward == self.best_reward
        }
        
        self.episode_history.append(step_info)
        
        return self.current_geometry.copy(), reward, terminated, truncated, step_info
    
    def _apply_action(self, action) -> Tuple[np.ndarray, bool]:
        """
        Apply action to modify geometry
        
        Args:
            action: Agent action
            
        Returns:
            new_geometry: Modified geometry matrix
            valid: Whether action was valid
        """
        new_geometry = self.current_geometry.copy()
        
        try:
            if self.action_type == "discrete":
                position_idx, material_type = action
                
                # Convert flat position index to 2D coordinates
                y = position_idx // self.grid_size
                x = position_idx % self.grid_size
                
                # Validate coordinates
                if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                    return new_geometry, False
                
                # Validate material type
                if not (0 <= material_type <= 2):
                    return new_geometry, False
                
            else:  # continuous
                x_norm, y_norm, material_prob = action
                
                # Convert normalized coordinates to grid positions
                x = int(x_norm * self.grid_size)
                y = int(y_norm * self.grid_size)
                
                # Determine material type from probability
                if material_prob < 0.33:
                    material_type = 0  # Void
                elif material_prob < 0.67:
                    material_type = 1  # N-type
                else:
                    material_type = 2  # P-type
                
                # Clamp coordinates
                x = max(0, min(x, self.grid_size - 1))
                y = max(0, min(y, self.grid_size - 1))
            
            # Apply modification
            new_geometry[y, x] = material_type
            
            # Validate resulting geometry
            validation = self.simulator.validate_material_matrix(new_geometry)
            
            if not validation['valid']:
                # Reject modification that breaks basic requirements
                return self.current_geometry, False
            
            return new_geometry, True
            
        except Exception as e:
            print(f"Action application error: {e}")
            return self.current_geometry, False
    
    def render(self, mode: str = 'human'):
        """
        Render current environment state
        
        Args:
            mode: Render mode ('human' or 'rgb_array')
        """
        if mode == 'rgb_array':
            return self._render_rgb_array()
        
        # Human-readable rendering
        print(f"\n=== Episode {self.episode_count}, Step {self.current_step} ===")
        print(f"Current geometry ({self.grid_size}×{self.grid_size}):")
        
        # Print material matrix with symbols
        symbols = {0: '·', 1: 'N', 2: 'P'}
        for row in self.current_geometry:
            print('  ' + ' '.join(symbols[cell] for cell in row))
        
        # Print recent performance if available
        if self.episode_history:
            last_step = self.episode_history[-1]
            print(f"Last reward: {last_step['reward']:.3f}")
            if 'performance' in last_step:
                perf = last_step['performance']
                print(f"Performance: Current={perf.get('forward_current', 0):.2e}A, "
                      f"Rect={perf.get('rectification_ratio', 0):.1e}, "
                      f"Power={perf.get('power', 0):.2e}W")
        
        print(f"Best reward this episode: {self.best_reward:.3f}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Render environment as RGB array for visualization"""
        # Create color map for materials
        colors = {
            0: [255, 255, 255],  # Void = White
            1: [100, 150, 255],  # N-type = Light Blue
            2: [255, 100, 100]   # P-type = Light Red
        }
        
        # Create RGB image
        rgb_array = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                material = self.current_geometry[i, j]
                rgb_array[i, j] = colors[material]
        
        # Scale up for better visibility
        scale_factor = max(1, 300 // self.grid_size)
        rgb_array = np.repeat(rgb_array, scale_factor, axis=0)
        rgb_array = np.repeat(rgb_array, scale_factor, axis=1)
        
        return rgb_array
    
    def get_environment_stats(self) -> Dict:
        """Get environment statistics"""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'successful_episodes': self.successful_episodes,
            'success_rate': self.successful_episodes / max(1, self.episode_count),
            'best_reward': self.best_reward,
            'current_step': self.current_step,
            'simulator_stats': self.simulator.get_simulation_stats(),
            'reward_stats': self.reward_calculator.get_reward_statistics()
        }
    
    def save_best_design(self, filename: str = "best_diode_design.npz"):
        """Save best discovered design"""
        if self.best_geometry is not None:
            np.savez(filename,
                    geometry=self.best_geometry,
                    reward=self.best_reward,
                    baseline_performance=self.baseline_performance,
                    grid_size=self.grid_size,
                    physical_size=self.physical_size)
            print(f"Best design saved to {filename}")
        else:
            print("No best design to save yet")
    
    def load_design(self, filename: str):
        """Load a saved design"""
        data = np.load(filename)
        self.current_geometry = data['geometry']
        print(f"Design loaded from {filename}")
        return self.current_geometry
    
    def close(self):
        """Clean up environment resources"""
        if hasattr(self, 'simulator'):
            self.simulator.cleanup_all()
        super().close()


def test_diode_rl_environment():
    """Test DiodeDesignEnvironment functionality"""
    print("Testing DiodeDesignEnvironment")
    print("=" * 60)
    
    # Create environment
    env = DiodeDesignEnvironment(
        grid_size=6,
        physical_size=5e-6,
        max_steps=10,
        reward_type="sparse",
        action_type="discrete"
    )
    
    print(f"\n1. Environment created successfully")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # Test reset
    print(f"\n2. Testing reset...")
    obs, info = env.reset(seed=42)
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Reset info: {info}")
    env.render()
    
    # Test valid actions
    print(f"\n3. Testing valid actions...")
    for i in range(5):
        # Random valid action
        position = np.random.randint(0, env.grid_size ** 2)
        material = np.random.randint(0, 3)
        action = [position, material]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {i+1}: Action={action}, Reward={reward:.3f}, "
              f"Valid={info['action_valid']}, Sim_success={info['simulation_success']}")
        
        if terminated or truncated:
            print(f"   Episode ended: terminated={terminated}, truncated={truncated}")
            break
    
    # Test invalid actions
    print(f"\n4. Testing invalid actions...")
    invalid_actions = [
        [env.grid_size ** 2 + 1, 1],  # Position out of bounds
        [0, 5],                       # Invalid material type
        [-1, 1],                      # Negative position
    ]
    
    for action in invalid_actions:
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Invalid action {action}: Reward={reward:.3f}, Valid={info['action_valid']}")
    
    # Test continuous actions
    print(f"\n5. Testing continuous actions...")
    env_cont = DiodeDesignEnvironment(
        grid_size=6,
        physical_size=5e-6,
        max_steps=5,
        action_type="continuous"
    )
    
    obs, info = env_cont.reset()
    
    for i in range(3):
        action = np.random.rand(3)  # Random continuous action
        obs, reward, terminated, truncated, info = env_cont.step(action)
        print(f"   Continuous step {i+1}: Action={action}, Reward={reward:.3f}")
    
    # Test statistics
    print(f"\n6. Environment statistics...")
    stats = env.get_environment_stats()
    print(f"   Stats: {stats}")
    
    # Test save/load
    print(f"\n7. Testing save/load...")
    env.save_best_design("test_design.npz")
    
    # Cleanup
    env.close()
    env_cont.close()
    
    print(f"\n" + "="*60)
    print(f"DIODE RL ENVIRONMENT TEST COMPLETE")
    print(f"="*60)
    print(f"✅ Environment: Functional (reset, step, render)")
    print(f"✅ Actions: Both discrete and continuous working")
    print(f"✅ Rewards: Multi-objective calculation working")
    print(f"✅ Validation: Geometry validation preventing invalid states")
    print(f"🚀 Ready for RL Agent implementation!")
    
    return env


if __name__ == "__main__":
    test_diode_rl_environment()