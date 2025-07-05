#!/usr/bin/env python3
"""
CNN-RL Integration with Geometry Environment
Connects CNN agent with the geometry optimization environment
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    from cnn_geometry_agent import CNNActorCritic, GeometryToTensor, CNNTrainer
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - showing conceptual integration")

from simple_rl_environment import SimpleGeometryEnv
from geometry_optimization_framework import GeometryMatrix

class CNNGeometryEnvironment(SimpleGeometryEnv):
    """Enhanced environment for CNN agent"""
    
    def __init__(self, width=32, height=32, max_steps=50):
        super().__init__(width, height, max_steps)
        self.action_history = []
        self.attention_history = []
        
    def reset(self):
        """Reset and return CNN-friendly observation"""
        obs = super().reset()
        self.action_history = []
        self.attention_history = []
        return obs
    
    def step(self, action_dict, attention_map=None):
        """Enhanced step with attention tracking"""
        obs, reward, done, info = super().step(action_dict)
        
        # Track action and attention
        self.action_history.append(action_dict.copy())
        if attention_map is not None:
            self.attention_history.append(attention_map.copy())
        
        # Enhanced info
        info['action_history'] = self.action_history
        info['attention_history'] = self.attention_history
        
        return obs, reward, done, info

class CNNAgent:
    """CNN-based RL agent for geometry optimization"""
    
    def __init__(self, input_size=32, device='cpu', lr=1e-4):
        self.device = device
        self.input_size = input_size
        
        if TORCH_AVAILABLE:
            self.network = CNNActorCritic().to(device)
            self.converter = GeometryToTensor(device)
            self.trainer = CNNTrainer(self.network, lr=lr, device=device)
            self.training_mode = True
        else:
            print("PyTorch not available - using random agent")
            self.network = None
            self.training_mode = False
        
        # Experience buffer
        self.episode_data = []
        self.global_step = 0
        
    def select_action(self, observation):
        """Select action using CNN policy"""
        if not TORCH_AVAILABLE or self.network is None:
            # Fallback to random action
            return self._random_action(), 0.0, 0.0, None
        
        material_matrix = observation['material_matrix']
        features = observation['features']
        
        # Convert to tensors
        geometry_tensor, features_tensor = self.converter.convert(material_matrix, features)
        
        # Get action from CNN
        if self.training_mode:
            action, log_prob, value, attention = self.network.select_action(
                geometry_tensor, features_tensor, deterministic=False
            )
        else:
            with torch.no_grad():
                action, log_prob, value, attention = self.network.select_action(
                    geometry_tensor, features_tensor, deterministic=True
                )
        
        # Convert to environment format
        env_action = {
            'x': float(action['x'][0]),
            'y': float(action['y'][0]),
            'radius': float(action['radius'][0]),
            'material': int(action['material'][0])
        }
        
        return env_action, float(log_prob[0]), float(value[0][0]), attention[0, 0]
    
    def _random_action(self):
        """Random action for fallback"""
        return {
            'x': np.random.uniform(0.1, 0.9),
            'y': np.random.uniform(0.0, 1.0),
            'radius': np.random.uniform(0.02, 0.15),
            'material': np.random.choice([0, 1, 2])
        }
    
    def store_experience(self, obs, action, reward, next_obs, done, log_prob, value):
        """Store experience for training"""
        experience = {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }
        self.episode_data.append(experience)
    
    def update_policy(self):
        """Update CNN policy using collected experience"""
        if not TORCH_AVAILABLE or len(self.episode_data) == 0:
            return {}
        
        # Extract experience components
        states = [exp['obs'] for exp in self.episode_data]
        actions = [exp['action'] for exp in self.episode_data]
        rewards = [exp['reward'] for exp in self.episode_data]
        dones = [exp['done'] for exp in self.episode_data]
        
        # Calculate next values (simplified)
        next_values = 0.0  # Assume terminal state
        
        # Compute loss and update
        loss_info = self.trainer.compute_loss(states, actions, rewards, dones, next_values)
        
        # Backward pass
        loss_info[0].backward()
        self.trainer.optimizer.step()
        self.trainer.optimizer.zero_grad()
        
        # Clear episode data
        self.episode_data = []
        self.global_step += 1
        
        return {
            'total_loss': loss_info[0].item(),
            'policy_loss': loss_info[1].item(),
            'value_loss': loss_info[2].item(),
            'entropy': loss_info[3].item()
        }

def train_cnn_agent(num_episodes=20, max_steps=30):
    """Train CNN agent on geometry optimization"""
    print("Training CNN Agent for Geometry Optimization")
    print("=" * 50)
    
    # Initialize environment and agent
    env = CNNGeometryEnvironment(width=32, height=32, max_steps=max_steps)
    device = 'cuda' if torch.cuda.is_available() and TORCH_AVAILABLE else 'cpu'
    agent = CNNAgent(device=device, lr=1e-4)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    best_reward = -np.inf
    best_geometry = None
    
    print(f"Device: {device}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"Episodes: {num_episodes}, Max steps: {max_steps}")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        episode_start_time = time.time()
        
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        for step in range(max_steps):
            # Agent selects action
            action, log_prob, value, attention = agent.select_action(obs)
            
            # Environment step
            next_obs, reward, done, info = env.step(action, attention)
            episode_reward += reward
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done, log_prob, value)
            
            # Log progress
            if step % 10 == 0 or done:
                interface_length = info['geometry_metrics']['interface_length'] * 1e6
                print(f"  Step {step:2d}: R={reward:6.3f}, Interface={interface_length:5.1f}μm")
            
            obs = next_obs
            if done:
                break
        
        # Update policy after episode
        loss_info = agent.update_policy()
        
        # Track metrics
        episode_rewards.append(episode_reward)
        if loss_info:
            episode_losses.append(loss_info['total_loss'])
        
        # Track best geometry
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_geometry = obs['material_matrix'].copy()
        
        episode_time = time.time() - episode_start_time
        
        # Episode summary
        print(f"  Episode reward: {episode_reward:.3f}")
        if loss_info:
            print(f"  Loss: {loss_info['total_loss']:.4f} (policy: {loss_info['policy_loss']:.4f})")
        print(f"  Episode time: {episode_time:.1f}s")
        print(f"  Best reward so far: {best_reward:.3f}")
    
    # Training summary
    print(f"\n" + "=" * 50)
    print("Training Complete!")
    print(f"Total episodes: {num_episodes}")
    print(f"Best reward: {best_reward:.3f}")
    print(f"Average reward (last 5): {np.mean(episode_rewards[-5:]):.3f}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'best_geometry': best_geometry,
        'best_reward': best_reward,
        'agent': agent,
        'env': env
    }

def visualize_cnn_learning(training_results):
    """Visualize CNN learning progress and attention"""
    print("Generating CNN learning visualizations...")
    
    episode_rewards = training_results['episode_rewards']
    best_geometry = training_results['best_geometry']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Learning curve
    axes[0, 0].plot(episode_rewards, 'b-', linewidth=2)
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Moving average
    window = min(5, len(episode_rewards) // 2)
    if window > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg, 'r-', linewidth=2)
        axes[0, 1].set_title(f'Moving Average (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Best geometry
    if best_geometry is not None:
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
        im = axes[0, 2].imshow(best_geometry, cmap=cmap, vmin=0, vmax=2)
        axes[0, 2].set_title('Best Geometry Found')
        axes[0, 2].set_xlabel('X')
        axes[0, 2].set_ylabel('Y')
        plt.colorbar(im, ax=axes[0, 2], ticks=[0, 1, 2], 
                    label='Material', shrink=0.6)
    
    # Loss curve (if available)
    if 'episode_losses' in training_results and training_results['episode_losses']:
        axes[1, 0].plot(training_results['episode_losses'], 'g-', linewidth=2)
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 1].hist(episode_rewards, bins=min(10, len(episode_rewards)//2+1), 
                   alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Episode Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Performance metrics
    if best_geometry is not None:
        # Calculate metrics for best geometry
        geom = GeometryMatrix(32, 32)
        geom.set_material_matrix(best_geometry)
        metrics = geom.calculate_metrics()
        
        metric_names = ['P-fraction', 'N-fraction', 'Void-fraction', 
                       'Interface(μm)', 'Connectivity', 'Min-size(μm)']
        metric_values = [
            metrics['p_fraction'],
            metrics['n_fraction'],
            metrics['void_fraction'],
            metrics['interface_length'] * 1e6,
            metrics['connectivity_score'],
            metrics['min_feature_size'] * 1e6
        ]
        
        bars = axes[1, 2].bar(range(len(metric_names)), metric_values, 
                             color=['red', 'blue', 'white', 'green', 'orange', 'purple'],
                             alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Best Geometry Metrics')
        axes[1, 2].set_xticks(range(len(metric_names)))
        axes[1, 2].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: cnn_training_results.png")

def demo_cnn_attention():
    """Demonstrate CNN attention mechanism"""
    print("\nDemonstrating CNN Attention Mechanism...")
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - skipping attention demo")
        return
    
    # Create agent and environment
    env = CNNGeometryEnvironment(width=32, height=32)
    agent = CNNAgent(device='cpu')
    
    # Create interesting geometry
    obs = env.reset()
    
    # Get action and attention
    action, log_prob, value, attention = agent.select_action(obs)
    
    # Visualize attention
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original geometry
    cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
    axes[0].imshow(obs['material_matrix'], cmap=cmap, vmin=0, vmax=2)
    axes[0].set_title('Input Geometry')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    
    # Attention map
    if attention is not None:
        im1 = axes[1].imshow(attention, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('CNN Attention Map')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[1], label='Attention Weight')
        
        # Overlay
        axes[2].imshow(obs['material_matrix'], cmap=cmap, vmin=0, vmax=2, alpha=0.7)
        axes[2].imshow(attention, cmap='hot', alpha=0.5, vmin=0, vmax=1)
        axes[2].set_title('Attention Overlay')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Y')
        
        # Add action indicator
        action_x = int(action['x'] * 32)
        action_y = int(action['y'] * 32)
        axes[2].scatter(action_x, action_y, c='yellow', s=100, marker='x', linewidth=3)
        axes[2].text(action_x+1, action_y+1, f"Action", color='yellow', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cnn_attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved: cnn_attention_analysis.png")
    print(f"Action selected: x={action['x']:.3f}, y={action['y']:.3f}, r={action['radius']:.3f}")
    print(f"State value: {value:.3f}")

def main():
    """Main demonstration of CNN-based geometry optimization"""
    print("CNN-BASED GEOMETRY OPTIMIZATION")
    print("🧠 Convolutional Neural Networks for Spatial Understanding")
    print("🎯 Multi-Channel Material Representation")
    print("⚡ Attention Mechanisms for Focused Learning")
    print()
    
    # Run training
    training_results = train_cnn_agent(num_episodes=15, max_steps=25)
    
    # Visualize results
    visualize_cnn_learning(training_results)
    
    # Demonstrate attention
    demo_cnn_attention()
    
    print("\n" + "=" * 60)
    print("CNN GEOMETRY OPTIMIZATION COMPLETE!")
    print("=" * 60)
    print(f"🏆 Best reward achieved: {training_results['best_reward']:.3f}")
    print(f"📈 Training episodes: {len(training_results['episode_rewards'])}")
    print(f"🧠 Network: {sum(p.numel() for p in training_results['agent'].network.parameters() if TORCH_AVAILABLE and training_results['agent'].network is not None):,} parameters")
    print()
    print("Generated files:")
    print("  📊 cnn_training_results.png - Training progress and metrics")
    print("  🎯 cnn_attention_analysis.png - Attention mechanism visualization")
    print()
    print("🚀 Ready for:")
    print("  • Advanced CNN architectures (ResNet, Attention)")
    print("  • Multi-objective optimization") 
    print("  • Transfer learning between geometries")
    print("  • Real DEVSIM physics integration")

if __name__ == "__main__":
    main()