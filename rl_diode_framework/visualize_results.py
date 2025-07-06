#!/usr/bin/env python3
"""
Result Visualization - View and analyze discovered diode designs
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from pathlib import Path

def load_experiment_results(exp_dir):
    """Load experiment results from directory"""
    results = {}
    
    # Load configuration
    config_path = os.path.join(exp_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            results['config'] = json.load(f)
    
    # Load experiment summary
    summary_path = os.path.join(exp_dir, 'logs', 'experiment_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
    
    # Load evaluation results
    eval_path = os.path.join(exp_dir, 'logs', 'evaluation_results.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            results['evaluation'] = json.load(f)
    
    # Load best design
    design_path = os.path.join(exp_dir, 'designs', 'best_design.npz')
    if os.path.exists(design_path):
        design_data = np.load(design_path)
        results['best_design'] = {
            'geometry': design_data['geometry'],
            'reward': float(design_data['reward']),
            'description': str(design_data['description'])
        }
    
    return results

def visualize_diode_geometry(geometry, title="Diode Geometry", figsize=(10, 8)):
    """
    Create detailed visualization of diode geometry
    
    Args:
        geometry: numpy array of material matrix
        title: Plot title
        figsize: Figure size
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    # Define colors and symbols
    colors = {0: 'white', 1: 'lightblue', 2: 'lightcoral'}
    symbols = {0: '·', 1: 'N', 2: 'P'}
    material_names = {0: 'Void', 1: 'N-type', 2: 'P-type'}
    
    # 1. Material distribution plot
    cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
    im1 = ax1.imshow(geometry, cmap=cmap, vmin=0, vmax=2, origin='lower')
    ax1.set_title('Material Distribution')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Add grid
    ax1.set_xticks(np.arange(-0.5, geometry.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, geometry.shape[0], 1), minor=True)
    ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Add text annotations
    for i in range(geometry.shape[0]):
        for j in range(geometry.shape[1]):
            ax1.text(j, i, symbols[geometry[i, j]], 
                    ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2], shrink=0.8)
    cbar1.set_ticklabels(['Void', 'N-type', 'P-type'])
    
    # 2. P-N Junction visualization
    p_mask = (geometry == 2)
    n_mask = (geometry == 1)
    
    # Create junction visualization
    junction_map = np.zeros_like(geometry)
    junction_map[p_mask] = 2  # P regions
    junction_map[n_mask] = 1  # N regions
    
    # Find P-N interfaces
    interface_map = np.zeros_like(geometry)
    for i in range(geometry.shape[0]):
        for j in range(geometry.shape[1]-1):
            if (geometry[i,j] == 2 and geometry[i,j+1] == 1) or (geometry[i,j] == 1 and geometry[i,j+1] == 2):
                interface_map[i,j] = 1
                interface_map[i,j+1] = 1
    for i in range(geometry.shape[0]-1):
        for j in range(geometry.shape[1]):
            if (geometry[i,j] == 2 and geometry[i+1,j] == 1) or (geometry[i,j] == 1 and geometry[i+1,j] == 2):
                interface_map[i,j] = 1
                interface_map[i+1,j] = 1
    
    # Plot P-N junction with interface highlighting
    im2 = ax2.imshow(geometry, cmap=cmap, vmin=0, vmax=2, origin='lower', alpha=0.7)
    
    # Highlight interfaces
    interface_y, interface_x = np.where(interface_map == 1)
    ax2.scatter(interface_x, interface_y, c='red', s=50, marker='s', alpha=0.8, label='P-N Interface')
    
    ax2.set_title('P-N Junction Analysis')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Material statistics
    total_pixels = geometry.size
    material_counts = {}
    material_fractions = {}
    
    for material in [0, 1, 2]:
        count = np.sum(geometry == material)
        material_counts[material] = count
        material_fractions[material] = count / total_pixels
    
    # Bar plot of material distribution
    materials = list(material_names.values())
    counts = [material_counts[i] for i in range(3)]
    colors_bar = ['white', 'lightblue', 'lightcoral']
    
    bars = ax3.bar(materials, counts, color=colors_bar, edgecolor='black', alpha=0.8)
    ax3.set_title('Material Distribution')
    ax3.set_ylabel('Pixel Count')
    
    # Add percentage labels on bars
    for bar, count, fraction in zip(bars, counts, material_fractions.values()):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}\n({fraction:.1%})',
                ha='center', va='bottom')
    
    ax3.set_ylim(0, max(counts) * 1.2)
    
    # 4. Geometry analysis text
    # Calculate additional metrics
    interface_pixels = np.sum(interface_map)
    
    # Connectivity analysis
    from scipy import ndimage
    p_components = ndimage.label(p_mask)[1]
    n_components = ndimage.label(n_mask)[1]
    
    analysis_text = f"""
    GEOMETRY ANALYSIS
    
    Grid Size: {geometry.shape[0]} × {geometry.shape[1]}
    Total Pixels: {total_pixels}
    
    Material Distribution:
    • P-type: {material_counts[2]} pixels ({material_fractions[2]:.1%})
    • N-type: {material_counts[1]} pixels ({material_fractions[1]:.1%})
    • Void: {material_counts[0]} pixels ({material_fractions[0]:.1%})
    
    Junction Properties:
    • Interface pixels: {interface_pixels}
    • P-type components: {p_components}
    • N-type components: {n_components}
    
    Design Characteristics:
    • Material balance: {1 - abs(material_fractions[2] - material_fractions[1]):.2f}
    • Active area: {(material_fractions[1] + material_fractions[2]):.1%}
    • Interface density: {interface_pixels/total_pixels:.1%}
    """
    
    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes,
            verticalalignment='top', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Analysis Summary')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def compare_designs(baseline_geometry, best_geometry, baseline_reward=0, best_reward=0):
    """Compare baseline and best discovered designs"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Define colormap
    cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
    symbols = {0: '·', 1: 'N', 2: 'P'}
    
    # 1. Baseline design
    im1 = ax1.imshow(baseline_geometry, cmap=cmap, vmin=0, vmax=2, origin='lower')
    ax1.set_title(f'Baseline Design\n(Reward: {baseline_reward:.1f})')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    
    # Add text annotations for baseline
    for i in range(baseline_geometry.shape[0]):
        for j in range(baseline_geometry.shape[1]):
            ax1.text(j, i, symbols[baseline_geometry[i, j]], 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.colorbar(im1, ax=ax1, ticks=[0, 1, 2], shrink=0.8).set_ticklabels(['Void', 'N-type', 'P-type'])
    
    # 2. Best discovered design
    im2 = ax2.imshow(best_geometry, cmap=cmap, vmin=0, vmax=2, origin='lower')
    ax2.set_title(f'Best Discovered Design\n(Reward: {best_reward:.1f})')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    # Add text annotations for best
    for i in range(best_geometry.shape[0]):
        for j in range(best_geometry.shape[1]):
            ax2.text(j, i, symbols[best_geometry[i, j]], 
                    ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2], shrink=0.8).set_ticklabels(['Void', 'N-type', 'P-type'])
    
    # 3. Material comparison
    materials = ['Void', 'N-type', 'P-type']
    baseline_counts = [np.sum(baseline_geometry == i) for i in range(3)]
    best_counts = [np.sum(best_geometry == i) for i in range(3)]
    
    x = np.arange(len(materials))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, baseline_counts, width, label='Baseline', alpha=0.8)
    bars2 = ax3.bar(x + width/2, best_counts, width, label='Best Design', alpha=0.8)
    
    ax3.set_xlabel('Material Type')
    ax3.set_ylabel('Pixel Count')
    ax3.set_title('Material Distribution Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(materials)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
    
    # 4. Improvement analysis
    improvement = best_reward - baseline_reward
    improvement_pct = (improvement / abs(baseline_reward)) * 100 if baseline_reward != 0 else float('inf')
    
    # Calculate design differences
    diff_matrix = best_geometry.astype(int) - baseline_geometry.astype(int)
    changes = np.sum(diff_matrix != 0)
    
    analysis_text = f"""
    DESIGN COMPARISON ANALYSIS
    
    Performance Improvement:
    • Baseline reward: {baseline_reward:.2f}
    • Best reward: {best_reward:.2f}
    • Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)
    
    Design Changes:
    • Modified pixels: {changes}/{baseline_geometry.size}
    • Change percentage: {changes/baseline_geometry.size:.1%}
    
    Material Changes:
    • Void: {best_counts[0] - baseline_counts[0]:+d} pixels
    • N-type: {best_counts[1] - baseline_counts[1]:+d} pixels  
    • P-type: {best_counts[2] - baseline_counts[2]:+d} pixels
    
    Key Insights:
    • The RL agent discovered a design with
      {improvement_pct:+.1f}% better performance
    • Modified {changes/baseline_geometry.size:.1%} of the geometry
    • {"Increased" if best_counts[2] > baseline_counts[2] else "Decreased"} P-type material
    • {"Increased" if best_counts[0] > baseline_counts[0] else "Decreased"} void regions
    """
    
    ax4.text(0.05, 0.95, analysis_text, transform=ax4.transAxes,
            verticalalignment='top', fontsize=11, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Improvement Analysis')
    
    plt.suptitle('Baseline vs Best Discovered Design', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def visualize_experiment_results(exp_dir, save_plots=True):
    """Visualize complete experiment results"""
    
    print(f"Loading experiment results from: {exp_dir}")
    results = load_experiment_results(exp_dir)
    
    if 'best_design' not in results:
        print("❌ No best design found in experiment results")
        return
    
    # Extract data
    best_geometry = results['best_design']['geometry']
    best_reward = results['best_design']['reward']
    
    print(f"✅ Loaded best design with reward: {best_reward:.2f}")
    print(f"   Geometry shape: {best_geometry.shape}")
    
    # Create baseline geometry for comparison
    grid_size = best_geometry.shape[0]
    baseline_geometry = np.ones((grid_size, grid_size), dtype=np.uint8)
    baseline_geometry[:, :grid_size//2] = 2  # P-type left
    baseline_geometry[:, grid_size//2:] = 1  # N-type right
    
    # 1. Detailed visualization of best design
    print("\n1. Creating detailed visualization of best design...")
    fig1 = visualize_diode_geometry(
        best_geometry, 
        f"Best Discovered Diode Design (Reward: {best_reward:.2f})"
    )
    
    if save_plots:
        plot_path = os.path.join(exp_dir, 'plots', 'detailed_best_design.png')
        fig1.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
    
    plt.show()
    
    # 2. Comparison with baseline
    print("\n2. Creating baseline vs best comparison...")
    fig2 = compare_designs(
        baseline_geometry, best_geometry,
        baseline_reward=0, best_reward=best_reward
    )
    
    if save_plots:
        plot_path = os.path.join(exp_dir, 'plots', 'design_comparison.png')
        fig2.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {plot_path}")
    
    plt.show()
    
    # 3. Print detailed analysis
    print("\n3. Detailed Analysis:")
    print("="*50)
    
    # Material analysis
    total_pixels = best_geometry.size
    p_count = np.sum(best_geometry == 2)
    n_count = np.sum(best_geometry == 1)
    void_count = np.sum(best_geometry == 0)
    
    print(f"Grid size: {best_geometry.shape[0]}×{best_geometry.shape[1]}")
    print(f"Total pixels: {total_pixels}")
    print(f"")
    print(f"Material distribution:")
    print(f"  P-type:  {p_count:2d} pixels ({p_count/total_pixels:5.1%})")
    print(f"  N-type:  {n_count:2d} pixels ({n_count/total_pixels:5.1%})")
    print(f"  Void:    {void_count:2d} pixels ({void_count/total_pixels:5.1%})")
    print(f"")
    
    # Geometry pattern
    print(f"Design pattern:")
    symbols = {0: '·', 1: 'N', 2: 'P'}
    for i, row in enumerate(best_geometry):
        row_str = '  ' + ' '.join(symbols[cell] for cell in row)
        print(f"  {row_str}")
    
    print(f"")
    print(f"Performance:")
    print(f"  Best reward: {best_reward:.2f}")
    
    if 'summary' in results:
        summary = results['summary']
        print(f"  Training episodes: {summary['training_episodes']}")
        print(f"  Mean reward: {summary['training_stats']['mean_reward']:.2f}")
        print(f"  Max reward: {summary['training_stats']['max_reward']:.2f}")
    
    if 'evaluation' in results:
        eval_data = results['evaluation']
        print(f"  Evaluation mean: {eval_data['mean_reward']:.2f} ± {eval_data['std_reward']:.2f}")
    
    print("="*50)
    print("✅ Visualization complete!")
    
    return results

def main():
    """Main visualization script"""
    parser = argparse.ArgumentParser(description='Visualize RL diode design results')
    parser.add_argument('exp_dir', help='Experiment directory path')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.exp_dir):
        print(f"❌ Experiment directory not found: {args.exp_dir}")
        return
    
    # Visualize results
    results = visualize_experiment_results(args.exp_dir, save_plots=not args.no_save)
    
    print(f"\n🎨 Visualization complete for: {args.exp_dir}")

if __name__ == "__main__":
    main()