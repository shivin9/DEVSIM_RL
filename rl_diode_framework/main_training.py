#!/usr/bin/env python3
"""
Main Training Script - Complete RL-based Diode Design Optimization
Orchestrates full-scale training to discover novel diode geometries
"""

import argparse
import os
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from diode_rl_environment import DiodeDesignEnvironment
from diode_rl_agent import DQNAgent, DiodeRLTrainer

def create_experiment_config():
    """Create comprehensive experiment configuration"""
    config = {
        # Environment settings
        'environment': {
            'grid_size': 16,        # Reduced for faster training
            'physical_size': 6e-6,  # 6 micrometers
            'max_steps': 10,        # Reduced for faster episodes
            'reward_type': 'sparse',
            'action_type': 'discrete'
        },
        
        # Agent settings
        'agent': {
            'learning_rate': 1e-3,
            'epsilon_start': 1.0,
            'epsilon_end': 0.8,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'memory_size': 10000,
            'target_update': 10,
            'hidden_size': 128
        },
        
        # Training settings
        'training': {
            'num_episodes': 250,
            'save_frequency': 25,
            'render_frequency': 50,
            'evaluation_frequency': 50,
            'evaluation_episodes': 10
        },
        
        # Experiment settings
        'experiment': {
            'name': f"diode_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'description': "RL-based novel diode geometry optimization",
            'objective': "Discover compact, high-performance diode designs"
        }
    }
    return config

def setup_experiment_directory(config):
    """Setup experiment directory structure"""
    exp_name = config['experiment']['name']
    exp_dir = f"experiments/{exp_name}"
    
    # Create directories
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/models", exist_ok=True)
    os.makedirs(f"{exp_dir}/designs", exist_ok=True)
    os.makedirs(f"{exp_dir}/logs", exist_ok=True)
    os.makedirs(f"{exp_dir}/plots", exist_ok=True)
    
    # Save configuration
    with open(f"{exp_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir

def run_diode_optimization_experiment(config, exp_dir):
    """
    Run complete diode optimization experiment
    
    Args:
        config: Experiment configuration
        exp_dir: Experiment directory
    """
    print(f"\n{'='*80}")
    print(f"DIODE RL OPTIMIZATION EXPERIMENT")
    print(f"{'='*80}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    print(f"Directory: {exp_dir}")
    
    # Create environment
    print(f"\n1. Setting up environment...")
    env_config = config['environment']
    env = DiodeDesignEnvironment(
        grid_size=env_config['grid_size'],
        physical_size=env_config['physical_size'],
        max_steps=env_config['max_steps'],
        reward_type=env_config['reward_type'],
        action_type=env_config['action_type']
    )
    
    print(f"   Environment: {env_config['grid_size']}×{env_config['grid_size']} grid")
    print(f"   Physical size: {env_config['physical_size']*1e6:.1f} μm")
    print(f"   Action space: {env.action_space}")
    
    # Create agent
    print(f"\n2. Setting up RL agent...")
    agent_config = config['agent']
    action_space_size = env.grid_size**2 * 3  # positions × materials
    
    agent = DQNAgent(
        grid_size=env_config['grid_size'],
        action_space_size=action_space_size,
        learning_rate=agent_config['learning_rate'],
        epsilon_start=agent_config['epsilon_start'],
        epsilon_end=agent_config['epsilon_end'],
        epsilon_decay=agent_config['epsilon_decay'],
        batch_size=agent_config['batch_size'],
        memory_size=agent_config['memory_size'],
        target_update=agent_config['target_update']
    )
    
    print(f"   Agent: DQN with {action_space_size} actions")
    print(f"   Network: CNN + FC ({agent_config['hidden_size']} hidden)")
    print(f"   Memory: {agent_config['memory_size']} experiences")
    
    # Create trainer
    print(f"\n3. Setting up trainer...")
    trainer = DiodeRLTrainer(env, agent, save_dir=f"{exp_dir}/models")
    
    training_config = config['training']
    print(f"   Episodes: {training_config['num_episodes']}")
    print(f"   Save frequency: {training_config['save_frequency']}")
    print(f"   Evaluation frequency: {training_config['evaluation_frequency']}")
    
    # Run training
    print(f"\n4. Starting training...")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        trainer.train(
            num_episodes=training_config['num_episodes'],
            save_frequency=training_config['save_frequency'],
            render_frequency=training_config['render_frequency']
        )
        
        training_time = time.time() - start_time
        print(f"   Training completed in {training_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print(f"\n   Training interrupted by user")
        training_time = time.time() - start_time
        print(f"   Partial training time: {training_time/60:.1f} minutes")
    
    # Run evaluation
    print(f"\n5. Running final evaluation...")
    eval_results = trainer.evaluate(
        num_episodes=training_config['evaluation_episodes'],
        render=False
    )
    
    # Save evaluation results
    with open(f"{exp_dir}/logs/evaluation_results.json", 'w') as f:
        json.dump({
            'mean_reward': float(eval_results['mean_reward']),
            'std_reward': float(eval_results['std_reward']),
            'max_reward': float(eval_results['max_reward']),
            'all_rewards': [float(r) for r in eval_results['all_rewards']]
        }, f, indent=2)
    
    # Generate comprehensive analysis
    print(f"\n6. Generating analysis...")
    generate_experiment_analysis(trainer, config, exp_dir)
    
    # Save best designs
    print(f"\n7. Saving best designs...")
    save_best_designs(trainer, exp_dir)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Training episodes: {len(trainer.episode_rewards)}")
    print(f"Best training reward: {trainer.training_stats['best_reward']:.2f}")
    print(f"Evaluation mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Results saved to: {exp_dir}")
    
    return trainer, eval_results


def generate_experiment_analysis(trainer, config, exp_dir):
    """Generate comprehensive experiment analysis"""
    
    # Training progress analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Episode rewards
    rewards = trainer.episode_rewards
    episodes = range(len(rewards))
    
    ax1.plot(episodes, rewards, alpha=0.6, color='blue', label='Episode Rewards')
    if len(rewards) > 50:
        # Moving average
        window = min(50, len(rewards) // 10)
        moving_avg = []
        for i in range(window, len(rewards)):
            moving_avg.append(np.mean(rewards[i-window:i]))
        ax1.plot(range(window, len(rewards)), moving_avg, 'red', linewidth=2, label=f'Moving Average ({window})')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reward distribution
    ax2.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', label=f'Mean: {np.mean(rewards):.2f}')
    ax2.axvline(np.median(rewards), color='green', linestyle='--', label=f'Median: {np.median(rewards):.2f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    
    # Training losses
    if trainer.agent.losses:
        ax3.plot(trainer.agent.losses, alpha=0.7)
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss')
        ax3.grid(True, alpha=0.3)
    
    # Exploration decay
    episodes_range = range(len(rewards))
    epsilons = []
    epsilon = config['agent']['epsilon_start']
    epsilon_decay = config['agent']['epsilon_decay']
    epsilon_end = config['agent']['epsilon_end']
    
    for _ in episodes_range:
        epsilons.append(epsilon)
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
    
    ax4.plot(episodes_range, epsilons, 'orange', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Epsilon (Exploration Rate)')
    ax4.set_title('Exploration Decay')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{exp_dir}/plots/training_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance summary
    summary = {
        'experiment': config['experiment'],
        'training_episodes': len(rewards),
        'training_stats': {
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'final_10_mean': float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float(np.mean(rewards))
        },
        'agent_stats': {
            'total_training_steps': trainer.agent.steps_done,
            'final_epsilon': trainer.agent.epsilon,
            'memory_size': len(trainer.agent.memory),
            'total_losses': len(trainer.agent.losses)
        }
    }
    
    with open(f"{exp_dir}/logs/experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

def save_best_designs(trainer, exp_dir):
    """Save and analyze best discovered designs"""
    if trainer.training_stats['best_design'] is not None:
        best_design = trainer.training_stats['best_design']
        best_reward = trainer.training_stats['best_reward']
        
        # Save best design
        np.savez(f"{exp_dir}/designs/best_design.npz",
                geometry=best_design,
                reward=best_reward,
                description=f"Best design with reward {best_reward:.2f}")
        
        # Visualize best design
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Material visualization
        cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
        im1 = ax1.imshow(best_design, cmap=cmap, vmin=0, vmax=2, origin='lower')
        ax1.set_title(f'Best Design (Reward: {best_reward:.2f})')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # Add colorbar
        cbar = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2])
        cbar.set_ticklabels(['Void', 'N-type', 'P-type'])
        
        # Design analysis
        p_count = np.sum(best_design == 2)
        n_count = np.sum(best_design == 1)
        void_count = np.sum(best_design == 0)
        total_pixels = best_design.size
        
        analysis_text = f"""
        Design Analysis:
        
        Material Distribution:
        • P-type: {p_count} pixels ({p_count/total_pixels:.1%})
        • N-type: {n_count} pixels ({n_count/total_pixels:.1%})
        • Void: {void_count} pixels ({void_count/total_pixels:.1%})
        
        Performance:
        • Best Reward: {best_reward:.2f}
        
        Geometry:
        • Grid Size: {best_design.shape[0]}×{best_design.shape[1]}
        • Total Pixels: {total_pixels}
        """
        
        ax2.text(0.1, 0.9, analysis_text, transform=ax2.transAxes, 
                verticalalignment='top', fontsize=11, fontfamily='monospace')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        ax2.set_title('Design Analysis')
        
        plt.tight_layout()
        plt.savefig(f"{exp_dir}/plots/best_design.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Best design saved (reward: {best_reward:.2f})")
        print(f"   Design: P={p_count}, N={n_count}, Void={void_count}")

def find_existing_experiments():
    """Find all existing experiments in the experiments directory"""
    experiments = []
    if os.path.exists("experiments"):
        for exp_name in os.listdir("experiments"):
            exp_path = os.path.join("experiments", exp_name)
            if os.path.isdir(exp_path):
                # Check if it has the required structure
                config_path = os.path.join(exp_path, "config.json")
                models_path = os.path.join(exp_path, "models")
                if os.path.exists(config_path) and os.path.exists(models_path):
                    experiments.append(exp_name)
    return sorted(experiments)

def load_experiment_config(exp_name):
    """Load configuration from existing experiment"""
    config_path = os.path.join("experiments", exp_name, "config.json")
    with open(config_path, 'r') as f:
        return json.load(f)

def find_latest_checkpoint(exp_dir):
    """Find the latest model checkpoint in an experiment"""
    models_dir = os.path.join(exp_dir, "models")
    if not os.path.exists(models_dir):
        return None
    
    # Look for episode checkpoints
    episode_models = []
    for filename in os.listdir(models_dir):
        if filename.startswith("model_episode_") and filename.endswith(".pth"):
            try:
                episode_num = int(filename.split("_")[2].split(".")[0])
                episode_models.append((episode_num, filename))
            except (ValueError, IndexError):
                continue
    
    if episode_models:
        # Return the highest episode number checkpoint
        latest_episode, latest_file = max(episode_models)
        return os.path.join(models_dir, latest_file), latest_episode
    
    # Fallback to final_model.pth if it exists
    final_model_path = os.path.join(models_dir, "final_model.pth")
    if os.path.exists(final_model_path):
        # Try to determine episode count from training stats
        try:
            import pickle
            stats_path = os.path.join(models_dir, "training_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                episode_count = len(stats.get('episode_rewards', []))
                return final_model_path, episode_count
        except:
            pass
        return final_model_path, 0
    
    return None

def main():
    """Main training script entry point"""
    parser = argparse.ArgumentParser(description='RL-based Diode Design Optimization')
    parser.add_argument('--episodes', type=int, default=500, 
                       help='For new experiments: total episodes to train. For restart: additional episodes to run from checkpoint')
    parser.add_argument('--grid-size', type=int, default=8, help='Grid size for geometry')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--restart', type=str, default=None, help='Restart experiment by name')
    parser.add_argument('--list-experiments', action='store_true', help='List all available experiments')
    parser.add_argument('--continue-latest', action='store_true', help='Continue the most recent experiment')
    
    args = parser.parse_args()
    
    # Handle listing experiments
    if args.list_experiments:
        experiments = find_existing_experiments()
        if experiments:
            print("Available experiments to restart:")
            for i, exp in enumerate(experiments, 1):
                exp_dir = os.path.join("experiments", exp)
                config = load_experiment_config(exp)
                checkpoint_info = find_latest_checkpoint(exp_dir)
                
                episodes_completed = 0
                total_episodes = config.get('training', {}).get('num_episodes', 'Unknown')
                
                if checkpoint_info:
                    _, episodes_completed = checkpoint_info
                
                print(f"  {i}. {exp}")
                print(f"     Episodes: {episodes_completed}/{total_episodes}")
                print(f"     Grid size: {config.get('environment', {}).get('grid_size', 'Unknown')}")
                print(f"     Description: {config.get('experiment', {}).get('description', 'No description')}")
                print()
        else:
            print("No experiments found to restart.")
        return
    
    # Handle restart functionality
    restart_exp_name = None
    if args.restart:
        restart_exp_name = args.restart
    elif args.continue_latest:
        experiments = find_existing_experiments()
        if experiments:
            restart_exp_name = experiments[-1]  # Most recent
        else:
            print("No experiments found to continue.")
            return
    
    if restart_exp_name:
        # Restart existing experiment
        experiments = find_existing_experiments()
        if restart_exp_name not in experiments:
            print(f"Experiment '{restart_exp_name}' not found.")
            print("Available experiments:", experiments)
            return
        
        print(f"Restarting experiment: {restart_exp_name}")
        exp_dir = os.path.join("experiments", restart_exp_name)
        config = load_experiment_config(restart_exp_name)
        
        # Check for checkpoint
        checkpoint_info = find_latest_checkpoint(exp_dir)
        if checkpoint_info:
            checkpoint_path, episodes_completed = checkpoint_info
            print(f"Found checkpoint at episode {episodes_completed}: {checkpoint_path}")
        else:
            print("No checkpoint found, starting from beginning")
            checkpoint_path, episodes_completed = None, 0
        
        # For restart, --episodes means ADDITIONAL episodes to run
        additional_episodes = args.episodes if args.episodes != 500 else 50  # Default 50 additional episodes
        
        # Add input validation
        if additional_episodes <= 0:
            print(f"Error: Additional episodes must be positive, got {additional_episodes}")
            return
        
        print(f"Will run {additional_episodes} additional episodes from episode {episodes_completed}")
        
        # Run restart experiment
        trainer, results = run_restart_experiment(config, exp_dir, checkpoint_path, episodes_completed, additional_episodes)
        
    else:
        # Create new experiment (original behavior)
        config = create_experiment_config()
        
        # Update with command line arguments
        if args.episodes:
            config['training']['num_episodes'] = args.episodes
        if args.grid_size:
            config['environment']['grid_size'] = args.grid_size
        if args.name:
            config['experiment']['name'] = args.name
        
        # Setup experiment
        exp_dir = setup_experiment_directory(config)
        
        # Run experiment
        trainer, results = run_diode_optimization_experiment(config, exp_dir)
    
    print(f"\n🎉 Experiment completed successfully!")
    if restart_exp_name:
        print(f"📁 Continued experiment: {exp_dir}")
    else:
        print(f"📁 Results: {exp_dir}")


def run_restart_experiment(config, exp_dir, checkpoint_path=None, start_episode=0, additional_episodes=50):
    """
    Run experiment from checkpoint (restart functionality)
    
    Args:
        config: Experiment configuration
        exp_dir: Experiment directory
        checkpoint_path: Path to model checkpoint file
        start_episode: Episode number to start from
        additional_episodes: Number of additional episodes to run beyond start_episode
    """
    print(f"\n{'='*80}")
    print(f"RESTARTING DIODE RL OPTIMIZATION EXPERIMENT")
    print(f"{'='*80}")
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Directory: {exp_dir}")
    print(f"Starting from episode: {start_episode}")
    print(f"Additional episodes to run: {additional_episodes}")
    target_episode = start_episode + additional_episodes
    print(f"Target final episode: {target_episode}")
    if checkpoint_path:
        print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create environment (same as original)
    print(f"\n1. Setting up environment...")
    env_config = config['environment']
    env = DiodeDesignEnvironment(
        grid_size=env_config['grid_size'],
        physical_size=env_config['physical_size'],
        max_steps=env_config['max_steps'],
        reward_type=env_config['reward_type'],
        action_type=env_config['action_type']
    )
    
    print(f"   Environment: {env_config['grid_size']}×{env_config['grid_size']} grid")
    print(f"   Physical size: {env_config['physical_size']*1e6:.1f} μm")
    
    # Create agent (same as original)
    print(f"\n2. Setting up RL agent...")
    agent_config = config['agent']
    action_space_size = env.grid_size**2 * 3  # positions × materials
    
    agent = DQNAgent(
        grid_size=env_config['grid_size'],
        action_space_size=action_space_size,
        learning_rate=agent_config['learning_rate'],
        epsilon_start=agent_config['epsilon_start'],
        epsilon_end=agent_config['epsilon_end'],
        epsilon_decay=agent_config['epsilon_decay'],
        batch_size=agent_config['batch_size'],
        memory_size=agent_config['memory_size'],
        target_update=agent_config['target_update']
    )
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n3. Loading checkpoint...")
        try:
            agent.load_model(checkpoint_path)
            print(f"   ✅ Checkpoint loaded successfully")
            print(f"   Current epsilon: {agent.epsilon:.3f}")
            print(f"   Training steps: {agent.steps_done}")
        except Exception as e:
            print(f"   ⚠️  Failed to load checkpoint: {e}")
            print(f"   ERROR: Cannot restart without valid checkpoint!")
            print(f"   Please check the checkpoint file or start a new experiment")
            return None, None
    else:
        print(f"\n3. No checkpoint to load, starting fresh")
        print(f"   ERROR: Cannot restart without checkpoint!")
        return None, None
    
    # Create trainer with restart capability
    print(f"\n4. Setting up trainer...")
    trainer = DiodeRLTrainer(env, agent, save_dir=f"{exp_dir}/models")
    
    # Load existing training statistics if available
    try:
        import pickle
        stats_path = os.path.join(exp_dir, "models", "training_stats.pkl")
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
            
            # Restore training history up to start_episode
            if 'episode_rewards' in stats and len(stats['episode_rewards']) > start_episode:
                trainer.episode_rewards = stats['episode_rewards'][:start_episode]
                trainer.episode_lengths = stats.get('episode_lengths', [])[:start_episode]
                
                # Update best design tracking
                if trainer.episode_rewards:
                    best_idx = np.argmax(trainer.episode_rewards)
                    trainer.training_stats['best_reward'] = trainer.episode_rewards[best_idx]
                
                print(f"   ✅ Loaded {len(trainer.episode_rewards)} episodes of training history")
            
    except Exception as e:
        print(f"   ⚠️  Could not load training statistics: {e}")
    
    training_config = config['training']
    original_total = training_config['num_episodes']
    
    print(f"   Original total episodes: {original_total}")
    print(f"   Episodes completed: {start_episode}")
    print(f"   Additional episodes: {additional_episodes}")
    print(f"   New target episodes: {target_episode}")
    
    # CRITICAL: Validate the target_episode calculation
    expected_target = start_episode + additional_episodes
    if target_episode != expected_target:
        print(f"❌ ERROR: target_episode calculation is wrong!")
        print(f"   Expected: {expected_target}, Got: {target_episode}")
        return None, None
    
    if additional_episodes <= 0:
        print(f"\n✅ No additional episodes requested!")
        
        # Run evaluation
        print(f"\n5. Running final evaluation...")
        eval_results = trainer.evaluate(
            num_episodes=training_config['evaluation_episodes'],
            render=False
        )
        return trainer, eval_results
    
    # Continue training
    print(f"\n5. Resuming training...")
    print(f"   Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   🎯 WILL RUN EPISODES: {start_episode} to {target_episode-1} (inclusive)")
    print(f"   📊 TOTAL EPISODES TO RUN: {target_episode - start_episode}")
    
    start_time = time.time()
    
    try:
        # CRITICAL: Use our calculated target_episode, NOT the original config
        trainer.train_from_episode(
            start_episode=start_episode,
            target_episodes=target_episode,  # This should be start_episode + additional_episodes
            save_frequency=training_config['save_frequency'],
            render_frequency=training_config['render_frequency']
        )
        
        training_time = time.time() - start_time
        print(f"   Training resumed and completed in {training_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print(f"\n   Training interrupted by user")
        training_time = time.time() - start_time
        print(f"   Partial training time: {training_time/60:.1f} minutes")
    
    # Run evaluation
    print(f"\n6. Running final evaluation...")
    eval_results = trainer.evaluate(
        num_episodes=training_config['evaluation_episodes'],
        render=False
    )
    
    # Save evaluation results
    with open(f"{exp_dir}/logs/evaluation_results.json", 'w') as f:
        json.dump({
            'mean_reward': float(eval_results['mean_reward']),
            'std_reward': float(eval_results['std_reward']),
            'max_reward': float(eval_results['max_reward']),
            'all_rewards': [float(r) for r in eval_results['all_rewards']],
            'restarted_from_episode': start_episode
        }, f, indent=2)
    
    # Generate analysis
    print(f"\n7. Generating analysis...")
    generate_experiment_analysis(trainer, config, exp_dir)
    
    # Save best designs
    print(f"\n8. Saving best designs...")
    save_best_designs(trainer, exp_dir)
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESTART COMPLETE")
    print(f"{'='*80}")
    print(f"Episodes resumed from: {start_episode}")
    print(f"Additional episodes run: {additional_episodes}")
    print(f"Final episode reached: {target_episode}")
    print(f"Total episodes completed: {len(trainer.episode_rewards)}")
    print(f"Best training reward: {trainer.training_stats['best_reward']:.2f}")
    print(f"Evaluation mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    
    return trainer, eval_results

if __name__ == "__main__":
    main()