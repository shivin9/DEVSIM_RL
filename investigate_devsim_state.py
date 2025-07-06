#!/usr/bin/env python3
"""
Investigate DEVSIM State Management
Check what happens to DEVSIM state between simulations
"""

import numpy as np

def investigate_devsim_state():
    """Investigate how DEVSIM handles state between simulations"""
    print("🔍 INVESTIGATING DEVSIM STATE MANAGEMENT")
    print("=" * 60)
    
    try:
        from devsim import (
            get_parameter, set_parameter, reset_devsim,
            create_2d_mesh, delete_device, delete_mesh,
            get_device_list, get_mesh_list
        )
        
        print("\n1. Initial DEVSIM state:")
        try:
            solver = get_parameter(name="direct_solver")
            print(f"   Solver: '{solver}'")
        except Exception as e:
            print(f"   No solver parameter: {e}")
        
        print(f"   Devices: {get_device_list()}")
        print(f"   Meshes: {get_mesh_list()}")
        
        print("\n2. Creating first simulation objects...")
        create_2d_mesh(mesh="test_mesh_1")
        print(f"   After mesh creation:")
        print(f"   Devices: {get_device_list()}")
        print(f"   Meshes: {get_mesh_list()}")
        
        print("\n3. Deleting first objects...")
        try:
            delete_mesh(mesh="test_mesh_1")
            print(f"   After deletion:")
            print(f"   Devices: {get_device_list()}")
            print(f"   Meshes: {get_mesh_list()}")
        except Exception as e:
            print(f"   Delete error: {e}")
        
        print("\n4. Creating second simulation objects...")
        try:
            create_2d_mesh(mesh="test_mesh_2")
            print(f"   After second mesh creation:")
            print(f"   Devices: {get_device_list()}")
            print(f"   Meshes: {get_mesh_list()}")
        except Exception as e:
            print(f"   Second mesh creation error: {e}")
        
        print("\n5. What happens with reset_devsim()...")
        try:
            solver_before = get_parameter(name="direct_solver")
            print(f"   Solver before reset: '{solver_before}'")
            
            reset_devsim()
            
            solver_after = get_parameter(name="direct_solver")
            print(f"   Solver after reset: '{solver_after}'")
            
            print(f"   Devices after reset: {get_device_list()}")
            print(f"   Meshes after reset: {get_mesh_list()}")
            
        except Exception as e:
            print(f"   Reset investigation error: {e}")
        
        print("\n6. Alternative: Using unique names without reset...")
        try:
            # Test if we can create multiple meshes with unique names
            for i in range(3):
                mesh_name = f"unique_mesh_{i}"
                print(f"   Creating {mesh_name}...")
                create_2d_mesh(mesh=mesh_name)
                print(f"     Meshes: {get_mesh_list()}")
        except Exception as e:
            print(f"   Unique names error: {e}")
        
        print("\n7. Cleanup test...")
        try:
            current_meshes = get_mesh_list()
            for mesh in current_meshes:
                print(f"   Deleting mesh: {mesh}")
                delete_mesh(mesh=mesh)
            
            print(f"   Final meshes: {get_mesh_list()}")
            print(f"   Final devices: {get_device_list()}")
        except Exception as e:
            print(f"   Cleanup error: {e}")
            
    except ImportError:
        print("   DEVSIM not available")
        return
    
    print("\n" + "=" * 60)
    print("DEVSIM State Investigation Complete!")

if __name__ == "__main__":
    investigate_devsim_state()