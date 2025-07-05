# CNN-Based Geometry Optimization System

This folder contains a complete Convolutional Neural Network (CNN) based system for 2D diode geometry optimization using reinforcement learning.

## System Overview

The CNN system processes 2D material distributions through multiple channels and uses attention mechanisms to focus on important geometric features for optimization.

## Files Description

### Core CNN Components

1. **cnn_architecture_demo.py** - Demonstrates CNN processing without PyTorch requirement
   - Multi-channel material representation (void/N-type/P-type)
   - Hierarchical feature extraction
   - Attention mechanisms
   - Policy network for action prediction

2. **cnn_geometry_agent.py** - Full CNN-based RL agent using PyTorch
   - CNNGeometryEncoder: Spatial feature extraction
   - CNNPolicyNetwork: Action selection
   - CNNValueNetwork: State value estimation
   - Complete Actor-Critic architecture

3. **cnn_environment_integration.py** - Integration between CNN agent and environment
   - CNNGeometryEnvironment: Enhanced environment for CNN training
   - Training loops and visualization
   - Attention mechanism demonstrations

### Supporting Framework

4. **geometry_optimization_framework.py** - Core geometry representation
   - GeometryMatrix: Material matrix operations
   - GeometryGenerator: Various geometry patterns
   - Metrics calculation (interface length, connectivity, etc.)

5. **simple_rl_environment.py** - RL environment for geometry optimization
   - State/action spaces
   - Reward functions
   - Episode management

6. **matrix_to_gmsh_simple.py** - Conversion from numpy matrices to GMSH
   - Matrix to GMSH geometry conversion
   - Mesh generation
   - DEVSIM integration support

### Generated Visualizations

7. **cnn_processing_*.png** - CNN processing demonstrations
   - Shows multi-channel input processing
   - Hierarchical feature extraction visualization
   - Attention mechanism outputs
   - Action prediction results

## Step-by-Step Usage

### Step 1: Basic CNN Architecture Understanding
```bash
python cnn_architecture_demo.py
```
This demonstrates the CNN processing pipeline without requiring PyTorch installation.

### Step 2: Full CNN Agent (requires PyTorch)
```bash
python cnn_geometry_agent.py
```
Tests the complete CNN-based RL agent architecture.

### Step 3: Training Integration
```bash
python cnn_environment_integration.py
```
Runs CNN agent training on geometry optimization tasks.

## CNN Architecture Details

### Multi-Channel Input
- Channel 0: Void/Air regions (0/1 binary)
- Channel 1: N-type Silicon regions (0/1 binary)  
- Channel 2: P-type Silicon regions (0/1 binary)

### CNN Processing Pipeline
1. **Convolutional Layers**: Extract spatial patterns
   - Conv1: Basic edge detection (3→32 channels)
   - Conv2: Pattern combinations (32→64 channels)
   - Conv3: High-level features (64→128 channels)

2. **Attention Mechanism**: Focus on important regions
   - Spatial attention weights
   - Feature enhancement

3. **Global Pooling**: Fixed-size feature extraction
   - Spatial dimension reduction
   - Feature vector creation

4. **Policy Network**: Action prediction
   - Position selection (x, y coordinates)
   - Material choice (void/N-type/P-type)
   - Modification radius

### Key Advantages
- **Spatial Understanding**: CNN processes 2D geometry patterns
- **Multi-Material Awareness**: Separate channels for each material type
- **Attention Mechanism**: Focuses on geometrically important regions
- **End-to-End Learning**: Direct geometry-to-action mapping
- **Scalable**: Works with different geometry sizes

## Requirements

### Basic Demo (no installation needed)
- numpy
- matplotlib

### Full CNN System
- PyTorch
- numpy  
- matplotlib

## Next Steps

1. Install PyTorch for full CNN functionality
2. Integrate with DEVSIM physics simulation
3. Add more sophisticated reward functions
4. Implement advanced CNN architectures (ResNet, Attention)
5. Add transfer learning between different device types

## Generated Files

When running the demos, the system generates:
- Visualization PNG files showing CNN processing stages
- Attention map overlays on input geometries
- Action prediction visualizations
- Training progress plots (when using full system)