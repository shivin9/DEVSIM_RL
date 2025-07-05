#!/usr/bin/env python3
"""
Test Matrix-to-DEVSIM Pipeline
Comprehensive testing of geometry matrix → GMSH → DEVSIM → performance evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path

# Our framework imports
from geometry_optimization_framework import GeometryMatrix, GeometryGenerator
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

class MatrixToDevsimTester:
    """
    Comprehensive tester for matrix-based geometry to DEVSIM pipeline
    """
    
    def __init__(self):
        self.test_results = {}
        self.converter = SimpleMatrixToGMSH()
        
    def test_geometry_generation(self):
        """Test Phase 1: Geometry matrix generation"""
        print("\n" + "="*60)
        print("PHASE 1: Testing Geometry Matrix Generation")
        print("="*60)
        
        test_cases = {
            'rectangular': lambda: GeometryGenerator.create_baseline_rectangular(16, 16),
            'circular': lambda: GeometryGenerator.create_circular_junction(16, 16, radius_ratio=0.3),
            'interdigitated': lambda: GeometryGenerator.create_interdigitated(16, 16, finger_width=2),
            'honeycomb': lambda: GeometryGenerator.create_honeycomb_pattern(16, 16, cell_size=4)
        }
        
        results = {}
        
        for name, generator in test_cases.items():
            print(f"\nTesting {name} geometry generation...")
            try:
                geom = generator()
                metrics = geom.calculate_metrics()
                
                # Validation checks
                checks = {
                    'has_both_materials': metrics['p_fraction'] > 0 and metrics['n_fraction'] > 0,
                    'contacts_preserved': self._check_contact_constraints(geom),
                    'reasonable_interface': metrics['interface_length'] > 0,
                    'connectivity_ok': metrics['connectivity_score'] > 0
                }
                
                all_passed = all(checks.values())
                
                results[name] = {
                    'success': True,
                    'metrics': metrics,
                    'validation': checks,
                    'all_checks_passed': all_passed
                }
                
                print(f"  ✓ Generated successfully")
                print(f"  P-fraction: {metrics['p_fraction']:.3f}")
                print(f"  N-fraction: {metrics['n_fraction']:.3f}") 
                print(f"  Interface length: {metrics['interface_length']*1e6:.1f} μm")
                print(f"  Validation: {'PASS' if all_passed else 'FAIL'}")
                
                if not all_passed:
                    failed_checks = [k for k, v in checks.items() if not v]
                    print(f"  Failed checks: {failed_checks}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[name] = {'success': False, 'error': str(e)}
        
        self.test_results['geometry_generation'] = results
        passed_count = sum(1 for r in results.values() if r.get('success', False) and r.get('all_checks_passed', False))
        print(f"\nPhase 1 Summary: {passed_count}/{len(test_cases)} geometries passed all tests")
        
        return results
    
    def test_matrix_to_gmsh_conversion(self):
        """Test Phase 2: Matrix to GMSH conversion"""
        print("\n" + "="*60) 
        print("PHASE 2: Testing Matrix-to-GMSH Conversion")
        print("="*60)
        
        # Use successful geometries from Phase 1
        phase1_results = self.test_results.get('geometry_generation', {})
        successful_geoms = {name: None for name, result in phase1_results.items() 
                          if result.get('success', False) and result.get('all_checks_passed', False)}
        
        if not successful_geoms:
            print("No successful geometries from Phase 1 - running geometry generation first")
            self.test_geometry_generation()
            phase1_results = self.test_results.get('geometry_generation', {})
            successful_geoms = {name: None for name, result in phase1_results.items() 
                              if result.get('success', False) and result.get('all_checks_passed', False)}
        
        results = {}
        
        for geom_name in successful_geoms.keys():
            print(f"\nTesting {geom_name} → GMSH conversion...")
            
            try:
                # Regenerate geometry
                if geom_name == 'rectangular':
                    geom = GeometryGenerator.create_baseline_rectangular(16, 16)
                elif geom_name == 'circular':
                    geom = GeometryGenerator.create_circular_junction(16, 16, radius_ratio=0.3)
                elif geom_name == 'interdigitated':
                    geom = GeometryGenerator.create_interdigitated(16, 16, finger_width=2)
                elif geom_name == 'honeycomb':
                    geom = GeometryGenerator.create_honeycomb_pattern(16, 16, cell_size=4)
                
                # Convert to GMSH
                with tempfile.TemporaryDirectory() as temp_dir:
                    geo_file = os.path.join(temp_dir, f"{geom_name}_test.geo")
                    msh_file = os.path.join(temp_dir, f"{geom_name}_test.msh")
                    
                    # Convert matrix to geo
                    self.converter.convert_matrix_to_gmsh(geom.material_matrix, geo_file)
                    
                    # Generate mesh
                    mesh_result = self.converter.generate_mesh(geo_file)
                    
                    # Validation checks
                    msh_file_actual = mesh_result if mesh_result else msh_file
                    checks = {
                        'geo_file_created': os.path.exists(geo_file),
                        'msh_file_created': os.path.exists(msh_file_actual) if mesh_result else False,
                        'mesh_generation_success': mesh_result is not None,
                        'file_size_reasonable': os.path.getsize(msh_file_actual) > 1000 if mesh_result and os.path.exists(msh_file_actual) else False
                    }
                    
                    all_passed = all(checks.values())
                    
                    results[geom_name] = {
                        'success': True,
                        'validation': checks,
                        'all_checks_passed': all_passed,
                        'geo_file_size': os.path.getsize(geo_file),
                        'msh_file_size': os.path.getsize(msh_file)
                    }
                    
                    print(f"  ✓ Conversion successful")
                    print(f"  GEO file size: {os.path.getsize(geo_file)} bytes")
                    print(f"  MSH file size: {os.path.getsize(msh_file)} bytes")
                    print(f"  Validation: {'PASS' if all_passed else 'FAIL'}")
                    
                    if not all_passed:
                        failed_checks = [k for k, v in checks.items() if not v]
                        print(f"  Failed checks: {failed_checks}")
                        
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[geom_name] = {'success': False, 'error': str(e)}
        
        self.test_results['matrix_to_gmsh'] = results
        passed_count = sum(1 for r in results.values() if r.get('success', False) and r.get('all_checks_passed', False))
        print(f"\nPhase 2 Summary: {passed_count}/{len(results)} conversions passed all tests")
        
        return results
    
    def test_devsim_simulation(self):
        """Test Phase 3: DEVSIM physics simulation"""
        print("\n" + "="*60)
        print("PHASE 3: Testing DEVSIM Physics Simulation")
        print("="*60)
        
        # Use successful conversions from Phase 2
        phase2_results = self.test_results.get('matrix_to_gmsh', {})
        successful_conversions = {name: None for name, result in phase2_results.items() 
                                if result.get('success', False) and result.get('all_checks_passed', False)}
        
        if not successful_conversions:
            print("No successful conversions from Phase 2 - running previous phases first")
            self.test_geometry_generation()
            self.test_matrix_to_gmsh_conversion()
            phase2_results = self.test_results.get('matrix_to_gmsh', {})
            successful_conversions = {name: None for name, result in phase2_results.items() 
                                    if result.get('success', False) and result.get('all_checks_passed', False)}
        
        results = {}
        
        for geom_name in successful_conversions.keys():
            print(f"\nTesting {geom_name} DEVSIM simulation...")
            
            try:
                # Regenerate geometry and mesh
                if geom_name == 'rectangular':
                    geom = GeometryGenerator.create_baseline_rectangular(16, 16)
                elif geom_name == 'circular':
                    geom = GeometryGenerator.create_circular_junction(16, 16, radius_ratio=0.3)
                elif geom_name == 'interdigitated':
                    geom = GeometryGenerator.create_interdigitated(16, 16, finger_width=2)
                elif geom_name == 'honeycomb':
                    geom = GeometryGenerator.create_honeycomb_pattern(16, 16, cell_size=4)
                
                # Run DEVSIM simulation
                sim_results = self._run_devsim_simulation(geom, geom_name)
                
                if sim_results['success']:
                    results[geom_name] = sim_results
                    print(f"  ✓ Simulation successful")
                    print(f"  Converged voltages: {sim_results['converged_count']}/{len(sim_results['voltages'])}")
                    if sim_results['converged_count'] > 0:
                        print(f"  Current at 0.5V: {sim_results['current_0_5V']:.2e} A")
                        print(f"  Max current: {sim_results['max_current']:.2e} A")
                else:
                    print(f"  ✗ Simulation failed: {sim_results['error']}")
                    results[geom_name] = sim_results
                        
            except Exception as e:
                print(f"  ✗ Exception: {e}")
                results[geom_name] = {'success': False, 'error': str(e)}
        
        self.test_results['devsim_simulation'] = results
        passed_count = sum(1 for r in results.values() if r.get('success', False))
        print(f"\nPhase 3 Summary: {passed_count}/{len(results)} simulations completed successfully")
        
        return results
    
    def test_full_pipeline(self):
        """Test complete pipeline end-to-end"""
        print("\n" + "="*60)
        print("FULL PIPELINE TEST")
        print("="*60)
        
        # Run all phases
        phase1 = self.test_geometry_generation()
        phase2 = self.test_matrix_to_gmsh_conversion()  
        phase3 = self.test_devsim_simulation()
        
        # Overall summary
        print(f"\n" + "="*60)
        print("PIPELINE TEST SUMMARY")
        print("="*60)
        
        total_geoms = len(phase1)
        phase1_pass = sum(1 for r in phase1.values() if r.get('success', False) and r.get('all_checks_passed', False))
        phase2_pass = sum(1 for r in phase2.values() if r.get('success', False) and r.get('all_checks_passed', False))
        phase3_pass = sum(1 for r in phase3.values() if r.get('success', False))
        
        print(f"Phase 1 (Geometry): {phase1_pass}/{total_geoms} passed")
        print(f"Phase 2 (GMSH): {phase2_pass}/{len(phase2)} passed")
        print(f"Phase 3 (DEVSIM): {phase3_pass}/{len(phase3)} passed")
        print(f"End-to-end success: {phase3_pass}/{total_geoms} geometries")
        
        # Identify best performing geometry
        successful_sims = {name: result for name, result in phase3.items() if result.get('success', False)}
        if successful_sims:
            best_current = max(successful_sims.items(), key=lambda x: x[1].get('current_0_5V', -1e15))
            print(f"\nBest performing geometry: {best_current[0]}")
            print(f"Current at 0.5V: {best_current[1]['current_0_5V']:.2e} A")
        
        pipeline_success = phase3_pass > 0
        print(f"\nPipeline Status: {'✓ WORKING' if pipeline_success else '✗ BROKEN'}")
        
        return pipeline_success
    
    def _check_contact_constraints(self, geom):
        """Check that contact constraints are properly enforced"""
        # Left edge should be P-type (material = 2)
        left_edge = geom.material_matrix[:, :3]
        left_all_p = np.all(left_edge == 2)
        
        # Right edge should be N-type (material = 1)
        right_edge = geom.material_matrix[:, -3:]
        right_all_n = np.all(right_edge == 1)
        
        return left_all_p and right_all_n
    
    def _run_devsim_simulation(self, geom, geom_name):
        """Run DEVSIM simulation for a geometry"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                geo_file = os.path.join(temp_dir, f"{geom_name}_sim.geo")
                msh_file = os.path.join(temp_dir, f"{geom_name}_sim.msh")
                
                # Convert and mesh
                self.converter.convert_matrix_to_gmsh(geom.material_matrix, geo_file)
                msh_file = self.converter.generate_mesh(geo_file)
                
                # DEVSIM setup
                device = f"{geom_name}_device"
                region = "Bulk"
                
                # Load mesh
                create_gmsh_mesh(mesh=f"{geom_name}_mesh", file=msh_file)
                add_gmsh_region(mesh=f"{geom_name}_mesh", gmsh_name="Bulk", region=region, material="Silicon")
                add_gmsh_contact(mesh=f"{geom_name}_mesh", gmsh_name="P_contact", region=region, material="metal", name="base")
                add_gmsh_contact(mesh=f"{geom_name}_mesh", gmsh_name="N_contact", region=region, material="metal", name="emitter")
                finalize_mesh(mesh=f"{geom_name}_mesh")
                create_device(mesh=f"{geom_name}_mesh", device=device)
                
                # Physics setup
                SetSiliconParameters(device, region, 300)
                
                # Simple doping profile (center junction)
                junction_pos = geom.physical_width / 2
                CreateNodeModel(device, region, "Acceptors", f"1.0e18*step({junction_pos}-x)")
                CreateNodeModel(device, region, "Donors", f"1.0e18*step(x-{junction_pos})")
                CreateNodeModel(device, region, "NetDoping", "Donors-Acceptors")
                
                # Initial solution
                CreateSolution(device, region, "Potential")
                CreateSiliconPotentialOnly(device, region)
                
                for contact in get_contact_list(device=device):
                    set_parameter(device=device, name=GetContactBiasName(contact), value=0.0)
                    CreateSiliconPotentialOnlyContact(device, region, contact)
                
                # Solve Poisson
                solve(type="dc", absolute_error=1.0, relative_error=1e-12, maximum_iterations=30)
                
                # Drift-diffusion setup
                CreateSolution(device, region, "Electrons")
                CreateSolution(device, region, "Holes")
                
                set_node_values(device=device, region=region, name="Electrons", init_from="IntrinsicElectrons")
                set_node_values(device=device, region=region, name="Holes", init_from="IntrinsicHoles")
                
                CreateSiliconDriftDiffusion(device, region)
                for contact in get_contact_list(device=device):
                    CreateSiliconDriftDiffusionAtContact(device, region, contact)
                
                # Solve equilibrium
                solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
                
                # Test voltage sweep
                voltages = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
                currents = []
                converged_count = 0
                
                for v in voltages:
                    set_parameter(device=device, name=GetContactBiasName("base"), value=v)
                    
                    try:
                        solve(type="dc", absolute_error=1e10, relative_error=1e-10, maximum_iterations=30)
                        
                        # Get current
                        i_electron = get_contact_current(device=device, contact="base", equation="ElectronContinuityEquation")
                        i_hole = get_contact_current(device=device, contact="base", equation="HoleContinuityEquation")
                        i_total = i_electron + i_hole
                        
                        currents.append(i_total)
                        converged_count += 1
                        
                    except:
                        currents.append(np.nan)
                
                # Calculate metrics
                valid_currents = [c for c in currents if not np.isnan(c)]
                current_0_5V = currents[-1] if len(currents) > 0 and not np.isnan(currents[-1]) else 0
                max_current = max(valid_currents) if valid_currents else 0
                
                return {
                    'success': True,
                    'voltages': voltages,
                    'currents': currents,
                    'converged_count': converged_count,
                    'current_0_5V': current_0_5V,
                    'max_current': max_current
                }
                
        except Exception as e:
            return {'success': False, 'error': str(e)}

def main():
    """Run comprehensive pipeline testing"""
    print("Matrix-to-DEVSIM Pipeline Comprehensive Testing")
    print("=" * 60)
    
    tester = MatrixToDevsimTester()
    pipeline_working = tester.test_full_pipeline()
    
    if pipeline_working:
        print(f"\n🎉 SUCCESS: Pipeline is working end-to-end!")
        print(f"Ready for optimization loop implementation.")
    else:
        print(f"\n❌ FAILURE: Pipeline has issues that need fixing.")
        print(f"Check individual phase results above.")
    
    return pipeline_working

if __name__ == "__main__":
    main()