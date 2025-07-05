#!/usr/bin/env python3
"""
CNN-based RL Agent for 2D Diode Geometry Optimization
Uses PyTorch CNNs to understand spatial patterns in material distributions
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Categorical, Beta
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for architecture demonstration
    class nn:
        class Module:
            def __init__(self): pass
        class Sequential:
            def __init__(self, *args): pass
        class Conv2d:
            def __init__(self, *args, **kwargs): pass
        class BatchNorm2d:
            def __init__(self, *args): pass
        class ReLU:
            def __init__(self, *args, **kwargs): pass
        class MaxPool2d:
            def __init__(self, *args): pass
        class AdaptiveAvgPool2d:
            def __init__(self, *args): pass
        class Linear:
            def __init__(self, *args): pass
        class Dropout:
            def __init__(self, *args): pass

import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random
from typing import Dict, Tuple, List
import time

class CNNGeometryEncoder(nn.Module):
    """CNN to extract spatial features from 2D material distributions"""
    
    def __init__(self, input_channels=3, input_size=32):
        super().__init__()
        
        # Multi-scale CNN architecture for geometry understanding
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism for important regions
        self.attention = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global pooling and feature extraction
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 3, 32, 32) - Multi-channel geometry
               Channel 0: Void regions (0/1)
               Channel 1: N-type regions (0/1) 
               Channel 2: P-type regions (0/1)
        Returns:
            features: (batch_size, 128) spatial features
            attention_map: (batch_size, 1, H, W) attention weights
        """
        # Hierarchical feature extraction
        x1 = self.conv1(x)  # (B, 32, 32, 32)
        x1_pool = F.max_pool2d(x1, 2)  # (B, 32, 16, 16)
        
        x2 = self.conv2(x1_pool)  # (B, 64, 16, 16)
        x2_pool = F.max_pool2d(x2, 2)  # (B, 64, 8, 8)
        
        x3 = self.conv3(x2_pool)  # (B, 128, 8, 8)
        
        # Attention mechanism
        attention_map = self.attention(x3)  # (B, 1, 8, 8)
        attended_features = x3 * attention_map  # Apply attention
        
        # Global feature extraction
        global_features = self.global_pool(attended_features)  # (B, 128, 1, 1)
        global_features = global_features.view(global_features.size(0), -1)  # (B, 128)
        
        # Final feature processing
        features = self.feature_extractor(global_features)  # (B, 128)
        
        return features, attention_map

class CNNPolicyNetwork(nn.Module):
    """Policy network with CNN spatial understanding"""
    
    def __init__(self, cnn_features=128, additional_features=8):
        super().__init__()
        
        # Combine CNN spatial features with numerical features
        combined_features = cnn_features + additional_features
        
        self.shared_layers = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True)
        )
        
        # Separate heads for different action components
        # Position head (where to modify)
        self.position_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)  # x_alpha, x_beta, y_alpha, y_beta for Beta distributions
        )
        
        # Size head (how big the modification)
        self.size_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2)  # radius_alpha, radius_beta for Beta distribution
        )
        
        # Material head (what material to place)
        self.material_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # void, N-type, P-type logits
        )
        
        # Action type head (add/remove/modify)
        self.action_type_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3)  # add, remove, modify logits
        )
        
    def forward(self, cnn_features, additional_features):
        """
        Args:
            cnn_features: (batch_size, 128) from CNN encoder
            additional_features: (batch_size, 8) numerical geometry features
        """
        # Combine features
        combined = torch.cat([cnn_features, additional_features], dim=1)
        shared = self.shared_layers(combined)
        
        # Generate action distributions
        position_params = self.position_head(shared)
        size_params = self.size_head(shared)
        material_logits = self.material_head(shared)
        action_type_logits = self.action_type_head(shared)
        
        # Create Beta distributions for continuous actions (0, 1)
        x_alpha, x_beta, y_alpha, y_beta = torch.chunk(position_params, 4, dim=1)
        x_alpha = F.softplus(x_alpha) + 1.0
        x_beta = F.softplus(x_beta) + 1.0
        y_alpha = F.softplus(y_alpha) + 1.0  
        y_beta = F.softplus(y_beta) + 1.0
        
        r_alpha, r_beta = torch.chunk(size_params, 2, dim=1)
        r_alpha = F.softplus(r_alpha) + 1.0
        r_beta = F.softplus(r_beta) + 1.0
        
        # Create distributions
        x_dist = Beta(x_alpha, x_beta)
        y_dist = Beta(y_alpha, y_beta)
        radius_dist = Beta(r_alpha, r_beta)
        material_dist = Categorical(logits=material_logits)
        action_type_dist = Categorical(logits=action_type_logits)
        
        return x_dist, y_dist, radius_dist, material_dist, action_type_dist

class CNNValueNetwork(nn.Module):
    """Value network for state evaluation"""
    
    def __init__(self, cnn_features=128, additional_features=8):
        super().__init__()
        
        combined_features = cnn_features + additional_features
        
        self.value_network = nn.Sequential(
            nn.Linear(combined_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)  # Single value output
        )
        
    def forward(self, cnn_features, additional_features):
        combined = torch.cat([cnn_features, additional_features], dim=1)
        value = self.value_network(combined)
        return value

class CNNActorCritic(nn.Module):
    """Complete CNN-based Actor-Critic architecture"""
    
    def __init__(self, input_channels=3, input_size=32, additional_features=8):
        super().__init__()
        
        self.encoder = CNNGeometryEncoder(input_channels, input_size)
        self.actor = CNNPolicyNetwork(128, additional_features)
        self.critic = CNNValueNetwork(128, additional_features)
        
    def forward(self, geometry_input, additional_features):
        """Complete forward pass"""
        # Extract spatial features
        cnn_features, attention_map = self.encoder(geometry_input)
        
        # Get policy distributions and value
        x_dist, y_dist, radius_dist, material_dist, action_type_dist = self.actor(
            cnn_features, additional_features
        )
        value = self.critic(cnn_features, additional_features)
        
        return (x_dist, y_dist, radius_dist, material_dist, action_type_dist), value, attention_map
    
    def select_action(self, geometry_input, additional_features, deterministic=False):
        """Select action using current policy"""
        with torch.no_grad():
            (x_dist, y_dist, radius_dist, material_dist, action_type_dist), value, attention = self.forward(
                geometry_input, additional_features
            )
            
            if deterministic:
                # Use mode for deterministic actions
                x = x_dist.mean
                y = y_dist.mean
                radius = radius_dist.mean
                material = material_dist.probs.argmax(dim=1)
                action_type = action_type_dist.probs.argmax(dim=1)
            else:
                # Sample from distributions
                x = x_dist.sample()
                y = y_dist.sample()
                radius = radius_dist.sample()
                material = material_dist.sample()
                action_type = action_type_dist.sample()
            
            # Calculate log probabilities
            log_prob_x = x_dist.log_prob(x)
            log_prob_y = y_dist.log_prob(y)
            log_prob_radius = radius_dist.log_prob(radius)
            log_prob_material = material_dist.log_prob(material)
            log_prob_action_type = action_type_dist.log_prob(action_type)
            
            total_log_prob = (log_prob_x + log_prob_y + log_prob_radius + 
                            log_prob_material + log_prob_action_type)
            
            # Convert to numpy for environment
            action = {
                'x': x.cpu().numpy(),
                'y': y.cpu().numpy(), 
                'radius': radius.cpu().numpy(),
                'material': material.cpu().numpy(),
                'action_type': action_type.cpu().numpy()
            }
            
            return action, total_log_prob.cpu().numpy(), value.cpu().numpy(), attention.cpu().numpy()

class GeometryToTensor:
    """Converts material matrix to CNN-ready tensor format"""
    
    def __init__(self, device='cpu'):
        self.device = device
        
    def convert(self, material_matrix, additional_features):
        """
        Convert material matrix to multi-channel tensor
        
        Args:
            material_matrix: (H, W) numpy array with values 0, 1, 2
            additional_features: (8,) numpy array with numerical features
            
        Returns:
            geometry_tensor: (1, 3, H, W) tensor
            features_tensor: (1, 8) tensor
        """
        H, W = material_matrix.shape
        
        # Create 3-channel representation
        channels = np.zeros((3, H, W), dtype=np.float32)
        
        # Channel 0: Void/Air (where material_matrix == 0)
        channels[0] = (material_matrix == 0).astype(np.float32)
        
        # Channel 1: N-type Silicon (where material_matrix == 1)
        channels[1] = (material_matrix == 1).astype(np.float32)
        
        # Channel 2: P-type Silicon (where material_matrix == 2)  
        channels[2] = (material_matrix == 2).astype(np.float32)
        
        # Convert to tensors and add batch dimension
        geometry_tensor = torch.FloatTensor(channels).unsqueeze(0).to(self.device)
        features_tensor = torch.FloatTensor(additional_features).unsqueeze(0).to(self.device)
        
        return geometry_tensor, features_tensor

class CNNTrainer:
    """Training loop for CNN-based geometry optimization"""
    
    def __init__(self, 
                 network,
                 lr=1e-4,
                 gamma=0.99,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 device='cpu'):
        
        self.network = network
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = device
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.attention_maps = []
        
    def compute_loss(self, states, actions, rewards, dones, next_values):
        """Compute Actor-Critic loss with entropy regularization"""
        
        # Convert states to tensors
        geometry_tensors = []
        feature_tensors = []
        converter = GeometryToTensor(self.device)
        
        for state in states:
            geom_tensor, feat_tensor = converter.convert(
                state['material_matrix'], state['features']
            )
            geometry_tensors.append(geom_tensor)
            feature_tensors.append(feat_tensor)
        
        geometry_batch = torch.cat(geometry_tensors, dim=0)
        features_batch = torch.cat(feature_tensors, dim=0)
        
        # Forward pass
        (x_dist, y_dist, radius_dist, material_dist, action_type_dist), values, attention = self.network(
            geometry_batch, features_batch
        )
        
        # Calculate advantages using GAE (Generalized Advantage Estimation)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        advantages = self.compute_advantages(rewards_tensor, values, next_values, dones)
        
        # Policy loss
        action_log_probs = []
        for i, action in enumerate(actions):
            log_prob_x = x_dist.log_prob(torch.FloatTensor([action['x']]).to(self.device))
            log_prob_y = y_dist.log_prob(torch.FloatTensor([action['y']]).to(self.device))
            log_prob_r = radius_dist.log_prob(torch.FloatTensor([action['radius']]).to(self.device))
            log_prob_m = material_dist.log_prob(torch.LongTensor([action['material']]).to(self.device))
            log_prob_a = action_type_dist.log_prob(torch.LongTensor([action['action_type']]).to(self.device))
            
            total_log_prob = log_prob_x[i] + log_prob_y[i] + log_prob_r[i] + log_prob_m[i] + log_prob_a[i]
            action_log_probs.append(total_log_prob)
        
        action_log_probs = torch.stack(action_log_probs)
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        
        # Value loss
        value_targets = advantages + values.squeeze()
        value_loss = F.mse_loss(values.squeeze(), value_targets.detach())
        
        # Entropy loss (exploration bonus)
        entropy = (x_dist.entropy() + y_dist.entropy() + radius_dist.entropy() + 
                  material_dist.entropy() + action_type_dist.entropy()).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return total_loss, policy_loss, value_loss, entropy, attention
        
    def compute_advantages(self, rewards, values, next_values, dones):
        """Compute advantages using GAE"""
        advantages = torch.zeros_like(rewards)
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values if not dones[t] else 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantage = delta + self.gamma * 0.95 * advantage  # GAE lambda = 0.95
            advantages[t] = advantage
            
        return advantages

def demonstrate_cnn_architecture():
    """Demonstrate the CNN architecture without training"""
    print("CNN-based RL Agent for Geometry Optimization")
    print("=" * 50)
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available - showing conceptual design")
        return
    
    # Create network
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    network = CNNActorCritic().to(device)
    converter = GeometryToTensor(device)
    
    # Create sample geometry
    material_matrix = np.random.randint(0, 3, size=(32, 32))
    additional_features = np.random.random(8)
    
    print(f"\nSample geometry matrix shape: {material_matrix.shape}")
    print(f"Material distribution: {np.bincount(material_matrix.flatten())}")
    
    # Convert to tensor
    geometry_tensor, features_tensor = converter.convert(material_matrix, additional_features)
    print(f"CNN input shape: {geometry_tensor.shape}")
    print(f"Features shape: {features_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        action, log_prob, value, attention = network.select_action(
            geometry_tensor, features_tensor
        )
    
    print(f"\nCNN Action Selection:")
    print(f"  Position: ({action['x'][0]:.3f}, {action['y'][0]:.3f})")
    print(f"  Radius: {action['radius'][0]:.3f}")
    print(f"  Material: {action['material'][0]}")
    print(f"  Action type: {action['action_type'][0]}")
    print(f"  Log probability: {log_prob[0]:.4f}")
    print(f"  State value: {value[0][0]:.4f}")
    print(f"  Attention map shape: {attention.shape}")
    
    # Network statistics
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"\nNetwork Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  CNN encoder parameters: {sum(p.numel() for p in network.encoder.parameters()):,}")
    print(f"  Actor parameters: {sum(p.numel() for p in network.actor.parameters()):,}")
    print(f"  Critic parameters: {sum(p.numel() for p in network.critic.parameters()):,}")
    
    # Visualize attention
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(material_matrix, cmap='viridis')
    plt.title('Original Geometry')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    attention_2d = attention[0, 0].cpu().numpy()
    plt.imshow(attention_2d, cmap='hot')
    plt.title('CNN Attention Map')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    # Overlay attention on geometry
    plt.imshow(material_matrix, cmap='viridis', alpha=0.7)
    plt.imshow(attention_2d, cmap='hot', alpha=0.5)
    plt.title('Attention Overlay')
    
    plt.tight_layout()
    plt.savefig('cnn_attention_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved attention visualization: cnn_attention_demo.png")
    print("✅ CNN architecture demonstration complete!")

if __name__ == "__main__":
    demonstrate_cnn_architecture()