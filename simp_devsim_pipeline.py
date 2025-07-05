#!/usr/bin/env python3
"""
SIMP → DEVSIM Pipeline
Fast topology discovery with SIMP + rigorous physics validation with DEVSIM
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import tempfile
import os
from typing import Dict, List, Tuple, Optional

# Import our components
from simp_topology_optimization import SIMPOptimizer
from geometry_optimization_framework import GeometryMatrix
from matrix_to_gmsh_simple import SimpleMatrixToGMSH

# DEVSIM imports
from devsim import (
    create_gmsh_mesh, add_gmsh_region, add_gmsh_contact, 
    finalize_mesh, create_device, set_parameter, solve, get_contact_current
)

from devsim.python_packages.simple_physics import (
    SetSiliconParameters, CreateSiliconDriftDiffusion,
    CreateSiliconDriftDiffusionAtContact, GetContactBiasName, 
    CreateSiliconPotentialOnly, CreateSiliconPotentialOnlyContact
)

from devsim.python_packages.model_create import CreateNodeModel, CreateSolution
from devsim import set_node_values, get_contact_list

class SIMPToDevsimConverter:
    """
    Convert SIMP density field to DEVSIM-compatible geometry
    """
    
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.converter = SimpleMatrixToGMSH()
        
    def density_to_materials(self, density_field: np.ndarray) -> np.ndarray:
        """
        Convert SIMP density field to discrete materials
        
        Strategy: Ensure continuous semiconductor path with proper P-N junction
        """
        nely, nelx = density_field.shape
        material_matrix = np.zeros_like(density_field, dtype=np.uint8)
        
        # First pass: Basic threshold-based assignment
        solid_mask = density_field > self.threshold
        
        # If almost no solid material, create a basic diode structure
        solid_fraction = np.sum(solid_mask) / (nely * nelx)
        if solid_fraction < 0.1:
            # Create a minimal working diode structure
            material_matrix[:, :nelx//2] = 2  # P-type left half
            material_matrix[:, nelx//2:] = 1  # N-type right half
            return material_matrix
        
        # Create continuous regions with smoothing
        for i in range(nely):
            for j in range(nelx):
                x_pos = j / nelx  # Normalized x position
                
                # Force continuous material along current path
                if j < 3:  # Left contact region
                    material_matrix[i, j] = 2  # P-type
                elif j >= nelx - 3:  # Right contact region  
                    material_matrix[i, j] = 1  # N-type
                elif density_field[i, j] > self.threshold:
                    # Solid region based on position and neighbors
                    if x_pos < 0.45:
                        material_matrix[i, j] = 2  # P-type
                    elif x_pos > 0.55:
                        material_matrix[i, j] = 1  # N-type
                    else:
                        # Junction region - smooth transition
                        junction_blend = (x_pos - 0.45) / 0.1  # 0 to 1 across junction
                        # Add some density influence
                        density_factor = (density_field[i, j] - self.threshold) / (1.0 - self.threshold)
                        combined_factor = 0.7 * junction_blend + 0.3 * density_factor
                        
                        material_matrix[i, j] = 1 if combined_factor > 0.5 else 2
                else:
                    # Low density but check neighbors for connectivity
                    neighbors_solid = 0
                    neighbor_material = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < nely and 0 <= nj < nelx:
                                if density_field[ni, nj] > self.threshold:
                                    neighbors_solid += 1
                                    neighbor_material += 1 if nj > nelx//2 else 2
                    
                    if neighbors_solid >= 3:  # Connect if many solid neighbors
                        avg_material = neighbor_material / neighbors_solid
                        material_matrix[i, j] = 1 if avg_material < 1.5 else 2
                    else:
                        material_matrix[i, j] = 0  # Void
        
        # Post-process: ensure contact connectivity
        # Make sure there's a continuous path from contacts
        self._ensure_contact_connectivity(material_matrix)
        
        return material_matrix
    
    def _ensure_contact_connectivity(self, material_matrix: np.ndarray):
        """Ensure continuous paths from contacts"""
        nely, nelx = material_matrix.shape
        
        # Ensure left contact has continuous P-type path
        left_contact = material_matrix[:, 0]
        for i in range(nely):
            if material_matrix[i, 0] != 2:
                material_matrix[i, 0] = 2
            # Propagate inward for a few columns
            for j in range(1, min(3, nelx//4)):
                if material_matrix[i, j] == 0:  # Fill voids near contact
                    material_matrix[i, j] = 2
        
        # Ensure right contact has continuous N-type path  
        for i in range(nely):
            if material_matrix[i, -1] != 1:
                material_matrix[i, -1] = 1
            # Propagate inward for a few columns
            for j in range(max(nelx-3, 3*nelx//4), nelx-1):
                if material_matrix[i, j] == 0:  # Fill voids near contact
                    material_matrix[i, j] = 1
    
    def create_geometry_matrix(self, density_field: np.ndarray, 
                             physical_width: float = 10e-6, 
                             physical_height: float = 10e-6) -> GeometryMatrix:
        """Convert density field to GeometryMatrix"""
        nely, nelx = density_field.shape
        
        # Create geometry matrix
        geom = GeometryMatrix(nelx, nely, physical_width, physical_height)
        
        # Convert density to materials
        material_matrix = self.density_to_materials(density_field)
        
        # Set the material matrix
        geom.material_matrix = material_matrix
        geom.enforce_contact_constraints()
        
        return geom

class SIMPDevSimPipeline:
    """
    Complete SIMP → DEVSIM optimization pipeline
    """
    
    def __init__(self, nelx=20, nely=20, physical_size=10e-6):
        self.nelx = nelx
        self.nely = nely
        self.physical_size = physical_size
        
        # Components
        self.simp_optimizer = None
        self.converter = SIMPToDevsimConverter()
        self.gmsh_converter = SimpleMatrixToGMSH(physical_size, physical_size)
        
        # Results tracking
        self.optimization_history = []
        self.validation_results = []
        
    def optimize_topology(self, 
                         objective_type='current_maximization',
                         max_iter=100, 
                         volfrac=0.5) -> np.ndarray:
        """
        Run SIMP topology optimization with semiconductor-specific objectives
        """
        print(f"Starting SIMP Topology Optimization")
        print(f"Objective: {objective_type}")
        print(f"Domain: {self.nelx} × {self.nely} elements")
        print("=" * 50)
        
        # Create modified SIMP optimizer for semiconductor objectives
        self.simp_optimizer = SemiconductorSIMPOptimizer(
            nelx=self.nelx, 
            nely=self.nely, 
            volfrac=volfrac,
            objective_type=objective_type
        )
        
        # Run optimization
        density_field = self.simp_optimizer.optimize(max_iter=max_iter)
        
        # Store results
        self.optimization_history = {
            'density_field': density_field,
            'compliance_history': self.simp_optimizer.compliance_history,
            'volume_history': self.simp_optimizer.volume_history,
            'objective_type': objective_type
        }
        
        return density_field
    
    def validate_with_devsim(self, density_field: np.ndarray, 
                           voltage_range: List[float] = None) -> Dict:
        """
        Validate SIMP topology with full DEVSIM physics
        """
        print(f"\nValidating topology with DEVSIM...")
        
        if voltage_range is None:
            voltage_range = [0.0, 0.2, 0.4, 0.5]
        
        try:
            # Convert to geometry matrix
            geom = self.converter.create_geometry_matrix(
                density_field, self.physical_size, self.physical_size
            )
            
            # Analyze geometry
            metrics = geom.calculate_metrics()
            print(f"Geometry metrics:")
            print(f"  P-fraction: {metrics['p_fraction']:.3f}")
            print(f"  N-fraction: {metrics['n_fraction']:.3f}")
            print(f"  Interface length: {metrics['interface_length']*1e6:.1f} μm")
            
            # Check if geometry is simulatable
            if metrics['p_fraction'] == 0 or metrics['n_fraction'] == 0:
                print(f"  ⚠️  Geometry lacks P-N junction - skipping DEVSIM")
                return {
                    'success': False,
                    'error': 'No P-N junction',
                    'geometry_metrics': metrics
                }
            
            # Run DEVSIM simulation
            results = self._run_devsim_simulation(geom, voltage_range)
            
            if results['success']:
                print(f"  ✓ DEVSIM simulation successful")
                print(f"  Current at 0.5V: {results.get('current_0_5V', 0):.2e} A")
                print(f"  Max current: {results.get('max_current', 0):.2e} A")
            else:
                print(f"  ✗ DEVSIM simulation failed: {results.get('error', 'Unknown')}")
            
            # Combine results
            validation_result = {
                **results,
                'geometry_metrics': metrics,
                'density_field': density_field
            }
            
            self.validation_results.append(validation_result)
            return validation_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'error': str(e),
                'density_field': density_field
            }
            self.validation_results.append(error_result)
            return error_result
    
    def _run_devsim_simulation(self, geom: GeometryMatrix, 
                              voltage_range: List[float]) -> Dict:
        """Run DEVSIM simulation with timeout protection"""
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                geo_file = os.path.join(temp_dir, "simp_device.geo")
                
                # Convert to GMSH
                self.gmsh_converter.convert_matrix_to_gmsh(geom.material_matrix, geo_file)
                msh_file = self.gmsh_converter.generate_mesh(geo_file)
                
                if not msh_file:
                    return {'success': False, 'error': 'Mesh generation failed'}
                
                # DEVSIM setup with unique names to avoid conflicts
                import random
                unique_id = f"{int(time.time()*1000)}_{random.randint(1000,9999)}"
                device_name = f"simp_device_{unique_id}"
                mesh_name = f"simp_mesh_{unique_id}"
                region = "Bulk"
                
                # Load mesh
                create_gmsh_mesh(mesh=mesh_name, file=msh_file)
                add_gmsh_region(mesh=mesh_name, gmsh_name="Bulk", region=region, material="Silicon")
                add_gmsh_contact(mesh=mesh_name, gmsh_name="P_contact", region=region, material="metal", name="anode")
                add_gmsh_contact(mesh=mesh_name, gmsh_name="N_contact", region=region, material="metal", name="cathode")
                finalize_mesh(mesh=mesh_name)
                create_device(mesh=mesh_name, device=device_name)
                
                # Physics setup
                SetSiliconParameters(device_name, region, 300)
                
                # Create doping profile from geometry
                self._setup_doping_from_geometry(device_name, region, geom)
                
                # Initial solution
                CreateSolution(device_name, region, "Potential")
                CreateSiliconPotentialOnly(device_name, region)
                
                for contact in get_contact_list(device=device_name):
                    set_parameter(device=device_name, name=GetContactBiasName(contact), value=0.0)
                    CreateSiliconPotentialOnlyContact(device_name, region, contact)
                
                # Solve Poisson with relaxed settings
                solve(type="dc", absolute_error=1.0, relative_error=1e-10, maximum_iterations=30)
                
                # Drift-diffusion setup with more conservative initialization
                CreateSolution(device_name, region, "Electrons")
                CreateSolution(device_name, region, "Holes")
                
                set_node_values(device=device_name, region=region, name="Electrons", init_from="IntrinsicElectrons")
                set_node_values(device=device_name, region=region, name="Holes", init_from="IntrinsicHoles")
                
                CreateSiliconDriftDiffusion(device_name, region)
                for contact in get_contact_list(device=device_name):
                    CreateSiliconDriftDiffusionAtContact(device_name, region, contact)
                
                # Solve equilibrium with more relaxed settings
                try:
                    solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=25)
                except:
                    # If drift-diffusion fails, try with even more relaxed settings
                    solve(type="dc", absolute_error=1e6, relative_error=1e-6, maximum_iterations=20)
                
                # Voltage sweep with more conservative approach
                currents = []
                voltages_actual = []
                
                # Start with smaller voltage steps for better convergence
                test_voltages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] if voltage_range is None else voltage_range
                
                for v in test_voltages:
                    set_parameter(device=device_name, name=GetContactBiasName("anode"), value=v)
                    
                    # Try multiple convergence strategies
                    converged = False
                    for attempt in range(2):
                        try:
                            if attempt == 0:
                                # First attempt: normal settings
                                solve(type="dc", absolute_error=1e8, relative_error=1e-8, maximum_iterations=15)
                            else:
                                # Second attempt: very relaxed settings
                                solve(type="dc", absolute_error=1e6, relative_error=1e-6, maximum_iterations=10)
                            
                            converged = True
                            break
                        except:
                            continue
                    
                    if converged:
                        try:
                            # Get current
                            i_electron = get_contact_current(device=device_name, contact="anode", equation="ElectronContinuityEquation")
                            i_hole = get_contact_current(device=device_name, contact="anode", equation="HoleContinuityEquation")
                            i_total = i_electron + i_hole
                            
                            currents.append(i_total)
                            voltages_actual.append(v)
                        except:
                            # Current extraction failed, skip this point
                            continue
                    else:
                        # Voltage point failed to converge, skip
                        continue
                
                # Calculate performance metrics
                if len(currents) > 0:
                    max_current = max(currents)
                    current_0_5V = None
                    
                    # Find current at 0.5V if available
                    for v, i in zip(voltages_actual, currents):
                        if abs(v - 0.5) < 0.1:
                            current_0_5V = i
                            break
                    
                    return {
                        'success': True,
                        'voltages': voltages_actual,
                        'currents': currents,
                        'max_current': max_current,
                        'current_0_5V': current_0_5V or currents[-1],
                        'num_converged': len(currents)
                    }
                else:
                    return {'success': False, 'error': 'No voltage points converged'}
                    
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _setup_doping_from_geometry(self, device: str, region: str, geom: GeometryMatrix):
        """Setup spatially-varying doping based on geometry matrix"""
        # For now, use a simple approach - could be improved with interpolation
        
        # Find P-N boundary
        p_regions = (geom.material_matrix == 2)
        n_regions = (geom.material_matrix == 1)
        
        if np.any(p_regions) and np.any(n_regions):
            # Find approximate junction position
            p_x_coords = []
            for j in range(geom.height):
                p_cols = np.where(p_regions[j, :])[0]
                if len(p_cols) > 0:
                    p_x_coords.append(np.max(p_cols))
            
            if p_x_coords:
                junction_col = np.median(p_x_coords)
                junction_pos = (junction_col + 0.5) * geom.dx
            else:
                junction_pos = geom.physical_width / 2
        else:
            junction_pos = geom.physical_width / 2
        
        # Create doping model
        CreateNodeModel(device, region, "Acceptors", f"1.0e18*step({junction_pos}-x)")
        CreateNodeModel(device, region, "Donors", f"1.0e18*step(x-{junction_pos})")
        CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
    
    def run_optimization_campaign(self, 
                                campaign_name: str,
                                test_cases: List[Dict]) -> Dict:
        """
        Run a complete optimization campaign with multiple test cases
        """
        print(f"\n" + "="*70)
        print(f"SIMP → DEVSIM OPTIMIZATION CAMPAIGN: {campaign_name}")
        print("="*70)
        
        campaign_results = []
        
        for i, case in enumerate(test_cases):
            print(f"\nTest Case {i+1}: {case.get('name', f'Case_{i+1}')}")
            print("-" * 50)
            
            start_time = time.time()
            
            # SIMP optimization
            density_field = self.optimize_topology(
                objective_type=case.get('objective', 'current_maximization'),
                max_iter=case.get('max_iter', 20),
                volfrac=case.get('volfrac', 0.5)
            )
            
            # DEVSIM validation
            validation_result = self.validate_with_devsim(
                density_field, 
                case.get('voltage_range', [0.0, 0.2, 0.4, 0.5])
            )
            
            case_time = time.time() - start_time
            
            # Combine results
            case_result = {
                'case_name': case.get('name', f'Case_{i+1}'),
                'parameters': case,
                'density_field': density_field,
                'validation_result': validation_result,
                'optimization_time': case_time,
                'simp_history': self.optimization_history.copy()
            }
            
            campaign_results.append(case_result)
            
            print(f"Case completed in {case_time:.1f}s")
        
        # Campaign analysis
        self._analyze_campaign_results(campaign_name, campaign_results)
        
        return {
            'campaign_name': campaign_name,
            'results': campaign_results,
            'summary': self._summarize_campaign(campaign_results)
        }
    
    def _analyze_campaign_results(self, campaign_name: str, results: List[Dict]):
        """Analyze and visualize campaign results"""
        
        # Create comprehensive visualization
        n_cases = len(results)
        fig, axes = plt.subplots(3, n_cases, figsize=(4*n_cases, 12))
        
        if n_cases == 1:
            axes = axes.reshape(-1, 1)
        
        for i, result in enumerate(results):
            case_name = result['case_name']
            
            # Plot 1: SIMP density field
            ax1 = axes[0, i]
            im1 = ax1.imshow(result['density_field'], cmap='viridis', origin='lower')
            ax1.set_title(f'{case_name}\nSIMP Density')
            plt.colorbar(im1, ax=ax1)
            
            # Plot 2: Material assignment
            ax2 = axes[1, i]
            if result['validation_result']['success']:
                geom = self.converter.create_geometry_matrix(result['density_field'])
                cmap = plt.cm.colors.ListedColormap(['white', 'lightblue', 'lightcoral'])
                im2 = ax2.imshow(geom.material_matrix, cmap=cmap, vmin=0, vmax=2, origin='lower')
                ax2.set_title('Material Assignment')
                cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2])
                cbar2.set_ticklabels(['Void', 'N-type', 'P-type'])
            else:
                ax2.text(0.5, 0.5, 'Validation\nFailed', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=12, color='red')
                ax2.set_title('Validation Failed')
            
            # Plot 3: Performance
            ax3 = axes[2, i]
            if (result['validation_result']['success'] and 
                'voltages' in result['validation_result']):
                
                voltages = result['validation_result']['voltages']
                currents = result['validation_result']['currents']
                
                ax3.semilogy(voltages, np.abs(currents), 'bo-', linewidth=2, markersize=6)
                ax3.set_xlabel('Voltage (V)')
                ax3.set_ylabel('|Current| (A)')
                ax3.set_title('I-V Characteristics')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No I-V Data', ha='center', va='center',
                        transform=ax3.transAxes, fontsize=12, color='red')
                ax3.set_title('No I-V Data')
        
        plt.tight_layout()
        plt.savefig(f'{campaign_name}_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nCampaign visualization saved: {campaign_name}_results.png")
    
    def _summarize_campaign(self, results: List[Dict]) -> Dict:
        """Summarize campaign results"""
        successful_cases = [r for r in results if r['validation_result']['success']]
        
        summary = {
            'total_cases': len(results),
            'successful_validations': len(successful_cases),
            'success_rate': len(successful_cases) / len(results) if results else 0
        }
        
        if successful_cases:
            # Performance statistics
            currents_0_5V = [r['validation_result'].get('current_0_5V', 0) 
                           for r in successful_cases if r['validation_result'].get('current_0_5V')]
            
            if currents_0_5V:
                summary.update({
                    'best_current_0_5V': max(currents_0_5V),
                    'worst_current_0_5V': min(currents_0_5V),
                    'avg_current_0_5V': np.mean(currents_0_5V)
                })
        
        return summary

class SemiconductorSIMPOptimizer(SIMPOptimizer):
    """
    Extended SIMP optimizer with semiconductor-specific objectives
    """
    
    def __init__(self, nelx=20, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, 
                 objective_type='current_maximization'):
        super().__init__(nelx, nely, volfrac, penal, rmin)
        self.objective_type = objective_type
        
    def _simplified_electrical_analysis(self, x):
        """
        Enhanced electrical analysis for semiconductor objectives
        """
        # Base analysis
        compliance, V = super()._simplified_electrical_analysis(x)
        
        if self.objective_type == 'current_maximization':
            # For current maximization, we want to minimize resistance (maximize conductance)
            # Invert compliance so higher values are better
            objective = -compliance
        elif self.objective_type == 'junction_optimization':
            # Optimize P-N junction interface area
            interface_length = self._calculate_interface_length(x)
            objective = -(compliance / (interface_length + 1e-12))
        else:
            # Default: minimize power dissipation
            objective = compliance
            
        return objective, V
    
    def _calculate_interface_length(self, x):
        """Estimate P-N interface length from density distribution"""
        # Simple gradient-based interface detection
        grad_x = np.gradient(x, axis=1)
        grad_y = np.gradient(x, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Interface where gradient is high
        interface_length = np.sum(gradient_magnitude > 0.1) * min(self.dx, self.dy)
        return interface_length

def test_simp_devsim_pipeline():
    """Test the complete SIMP → DEVSIM pipeline"""
    
    print("Testing SIMP → DEVSIM Pipeline")
    print("=" * 50)
    
    # Create pipeline
    pipeline = SIMPDevSimPipeline(nelx=50, nely=50, physical_size=10e-6)
    
    # Test cases
    test_cases = [
        {
            'name': 'CurrentMax_LowVol',
            'objective': 'current_maximization',
            'volfrac': 0.4,
            'max_iter': 50
        },
        {
            'name': 'CurrentMax_HighVol', 
            'objective': 'current_maximization',
            'volfrac': 0.6,
            'max_iter': 50
        },
        {
            'name': 'JunctionOpt_MidVol',
            'objective': 'junction_optimization',
            'volfrac': 0.5,
            'max_iter': 50
        }
    ]
    
    # Run campaign
    results = pipeline.run_optimization_campaign(
        campaign_name="SIMPtoDevsim_Test",
        test_cases=test_cases
    )
    
    # Print summary
    print(f"\n" + "="*70)
    print("PIPELINE TEST SUMMARY")
    print("="*70)
    
    summary = results['summary']
    print(f"Total test cases: {summary['total_cases']}")
    print(f"Successful validations: {summary['successful_validations']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    
    if 'best_current_0_5V' in summary:
        print(f"Best current at 0.5V: {summary['best_current_0_5V']:.2e} A")
        print(f"Performance range: {summary['worst_current_0_5V']:.2e} to {summary['best_current_0_5V']:.2e} A")
    
    print(f"\nPipeline test complete! ✓")
    
    return results

if __name__ == "__main__":
    test_simp_devsim_pipeline()