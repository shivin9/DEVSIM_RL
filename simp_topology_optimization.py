#!/usr/bin/env python3
"""
SIMP Topology Optimization for Semiconductor Devices
Implements Solid Isotropic Material with Penalization method
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time

class SIMPOptimizer:
    """
    SIMP topology optimization for semiconductor devices
    """
    
    def __init__(self, nelx=40, nely=40, volfrac=0.5, penal=3.0, rmin=1.5):
        """
        Initialize SIMP optimizer
        
        Args:
            nelx, nely: Number of elements in x and y directions
            volfrac: Volume fraction constraint
            penal: Penalization parameter
            rmin: Filter radius
        """
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        
        # Physical dimensions (10 μm device)
        self.lx = 10e-6  # Length
        self.ly = 10e-6  # Width
        self.dx = self.lx / nelx
        self.dy = self.ly / nely
        
        # Material properties for semiconductors
        self.E0 = 1e-12  # Base electrical conductivity (void)
        self.E1 = 1e-3   # Semiconductor conductivity
        self.nu = 0.3    # Not used for electrical problems but kept for structure
        
        # Initialize density field
        self.x = np.ones((nely, nelx)) * volfrac
        
        # Prepare filter
        self.H, self.Hs = self._prepare_filter()
        
        # Setup gradient operators for analytical sensitivities
        self.Dx, self.Dy = self._setup_gradient_operators()
        
        # History tracking
        self.compliance_history = []
        self.volume_history = []
        self.density_history = []
        
    def _prepare_filter(self):
        """Prepare density filter matrix"""
        iH = np.ones((self.nelx * self.nely * int((2 * (np.ceil(self.rmin) - 1) + 1) ** 2),), dtype=int)
        jH = np.ones_like(iH)
        sH = np.zeros_like(iH, dtype=float)
        cc = 0
        
        for i in range(self.nelx):
            for j in range(self.nely):
                row = i * self.nely + j
                kk1 = int(np.maximum(i - (np.ceil(self.rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(self.rmin), self.nelx))
                ll1 = int(np.maximum(j - (np.ceil(self.rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(self.rmin), self.nely))
                
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * self.nely + l
                        fac = self.rmin - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                        if fac > 0:
                            iH[cc] = row
                            jH[cc] = col
                            sH[cc] = fac
                            cc += 1
        
        # Finalize filter matrix
        H = csc_matrix((sH[:cc], (iH[:cc], jH[:cc])), 
                       shape=(self.nelx * self.nely, self.nelx * self.nely))
        Hs = H.sum(axis=1).A1
        
        return H, Hs
    
    def _setup_gradient_operators(self):
        """
        Setup sparse matrices for fast voltage gradient computation
        Returns Dx, Dy matrices such that:
        Ex = (Dx @ V).reshape(nely, nelx)  # ∂V/∂x 
        Ey = (Dy @ V).reshape(nely, nelx)  # ∂V/∂y
        """
        n_nodes = (self.nely + 1) * (self.nelx + 1)
        n_elements = self.nely * self.nelx
        
        Dx = self._build_gradient_matrix_x(n_elements, n_nodes)
        Dy = self._build_gradient_matrix_y(n_elements, n_nodes)
        
        return Dx, Dy
    
    def _build_gradient_matrix_x(self, n_elements, n_nodes):
        """Build sparse matrix for ∂V/∂x computation"""
        from scipy.sparse import coo_matrix
        
        row_indices = []
        col_indices = []
        values = []
        
        elem_idx = 0
        for i in range(self.nely):
            for j in range(self.nelx):
                # Element nodes (counter-clockwise from bottom-left)
                n1 = i * (self.nelx + 1) + j        # bottom-left
                n2 = i * (self.nelx + 1) + (j + 1)  # bottom-right  
                n3 = (i + 1) * (self.nelx + 1) + (j + 1)  # top-right
                n4 = (i + 1) * (self.nelx + 1) + j  # top-left
                
                # ∂V/∂x using bilinear shape functions at element center
                # dN/dx = [-1, 1, 1, -1] / (2*dx) for bilinear quad
                coeff = 1.0 / (2.0 * self.dx)
                
                row_indices.extend([elem_idx] * 4)
                col_indices.extend([n1, n2, n3, n4])
                values.extend([-coeff, coeff, coeff, -coeff])
                
                elem_idx += 1
        
        return coo_matrix((values, (row_indices, col_indices)), 
                          shape=(n_elements, n_nodes)).tocsr()
    
    def _build_gradient_matrix_y(self, n_elements, n_nodes):
        """Build sparse matrix for ∂V/∂y computation"""
        from scipy.sparse import coo_matrix
        
        row_indices = []
        col_indices = []
        values = []
        
        elem_idx = 0
        for i in range(self.nely):
            for j in range(self.nelx):
                # Element nodes (counter-clockwise from bottom-left)
                n1 = i * (self.nelx + 1) + j        # bottom-left
                n2 = i * (self.nelx + 1) + (j + 1)  # bottom-right  
                n3 = (i + 1) * (self.nelx + 1) + (j + 1)  # top-right
                n4 = (i + 1) * (self.nelx + 1) + j  # top-left
                
                # ∂V/∂y using bilinear shape functions at element center
                # dN/dy = [-1, -1, 1, 1] / (2*dy) for bilinear quad
                coeff = 1.0 / (2.0 * self.dy)
                
                row_indices.extend([elem_idx] * 4)
                col_indices.extend([n1, n2, n3, n4])
                values.extend([-coeff, -coeff, coeff, coeff])
                
                elem_idx += 1
        
        return coo_matrix((values, (row_indices, col_indices)), 
                          shape=(n_elements, n_nodes)).tocsr()
    
    
    def _material_interpolation(self, x):
        """
        SIMP material interpolation for electrical conductivity
        σ(ρ) = σ_min + (σ_max - σ_min) * ρ^p
        """
        return self.E0 + (self.E1 - self.E0) * x ** self.penal
    
    def _simplified_electrical_analysis(self, x):
        """
        Simplified electrical analysis using finite differences
        Solves: ∇ · (σ(ρ) ∇V) = 0
        """
        # Material conductivity based on density
        sigma = self._material_interpolation(x)
        
        # Set up finite difference grid
        # Simple Laplacian with conductivity weighting
        # This is a simplified version - real implementation would use FEM
        
        # Boundary conditions: V=1 on left, V=0 on right
        V = np.zeros((self.nely + 1, self.nelx + 1))
        V[:, 0] = 1.0  # Left boundary
        V[:, -1] = 0.0  # Right boundary
        
        # Simple iterative solver (Gauss-Seidel)
        for iteration in range(100):
            V_old = V.copy()
            
            for i in range(1, self.nely):
                for j in range(1, self.nelx):
                    # Update potential (simple averaging for now)
                    V[i, j] = 0.25 * (V[i-1, j] + V[i+1, j] + V[i, j-1] + V[i, j+1])
            
            # Check convergence
            if np.max(np.abs(V - V_old)) < 1e-6:
                break
        
        # Calculate current density and compliance
        # J = -σ ∇V, compliance = ∫ J²/σ dΩ (power dissipation)
        compliance = 0.0
        
        for i in range(self.nely):
            for j in range(self.nelx):
                # Gradient of potential (electric field) - center differences
                if j < self.nelx:
                    Ex = -(V[i, min(j+1, self.nelx)] - V[i, j]) / self.dx
                else:
                    Ex = 0
                    
                if i < self.nely:
                    Ey = -(V[min(i+1, self.nely), j] - V[i, j]) / self.dy
                else:
                    Ey = 0
                
                # Power dissipation: minimize ∫ σ|∇V|² dΩ (favors conductive paths)
                compliance += sigma[i, j] * (Ex**2 + Ey**2) * self.dx * self.dy
        
        return compliance, V
    
    def _sensitivity_analysis(self, x):
        """
        Compute sensitivity of compliance w.r.t. density
        Using finite differences for simplicity
        """
        # Get baseline compliance
        c0, _ = self._simplified_electrical_analysis(x)
        
        # Sensitivity array
        dc = np.zeros_like(x)
        
        # Finite difference step
        delta = 1e-6
        
        for i in range(self.nely):
            for j in range(self.nelx):
                # Perturb density
                x_pert = x.copy()
                x_pert[i, j] += delta
                
                # Clamp to [0,1]
                x_pert[i, j] = np.clip(x_pert[i, j], 0, 1)
                
                # Compute perturbed compliance
                c_pert, _ = self._simplified_electrical_analysis(x_pert)
                
                # Sensitivity
                dc[i, j] = (c_pert - c0) / delta
        
        return dc, c0
    
    def _analytical_sensitivity_analysis(self, x):
        """
        Compute analytical sensitivities
        Much faster than finite differences: O(N) vs O(N²)
        """
        # Get baseline compliance and voltage solution
        c0, V = self._simplified_electrical_analysis(x)
        
        # Material conductivity and its derivative
        sigma = self._material_interpolation(x)
        dE_drho = self.penal * (self.E1 - self.E0) * x**(self.penal - 1)
        
        # Use the SAME gradient calculation as electrical analysis (forward differences)
        dc = np.zeros_like(x)
        
        for i in range(self.nely):
            for j in range(self.nelx):
                # Same gradient calculation as in _simplified_electrical_analysis
                if j < self.nelx:
                    Ex = -(V[i, min(j+1, self.nelx)] - V[i, j]) / self.dx
                else:
                    Ex = 0
                    
                if i < self.nely:
                    Ey = -(V[min(i+1, self.nely), j] - V[i, j]) / self.dy
                else:
                    Ey = 0
                
                # Base power dissipation sensitivity
                E_field_sq = Ex**2 + Ey**2
                dc[i, j] = dE_drho[i, j] * E_field_sq * self.dx * self.dy
        
        return dc, c0
    
    def _compute_voltage_gradient_x(self, V):
        """Fast computation of ∂V/∂x using pre-built sparse matrix"""
        # V comes as (nely+1, nelx+1) array, need to flatten for matrix multiplication
        V_flat = V.flatten()
        Ex_flat = self.Dx @ V_flat
        return Ex_flat.reshape(self.nely, self.nelx)
    
    def _compute_voltage_gradient_y(self, V):
        """Fast computation of ∂V/∂y using pre-built sparse matrix"""
        # V comes as (nely+1, nelx+1) array, need to flatten for matrix multiplication
        V_flat = V.flatten()
        Ey_flat = self.Dy @ V_flat
        return Ey_flat.reshape(self.nely, self.nelx)
    
    def _optimality_criteria_update(self, x, dc):
        """
        Update densities using Optimality Criteria method
        """
        # Filter sensitivities
        dc_flat = np.reshape(dc, (-1, 1))
        dc_filtered_flat = self.H @ dc_flat
        if hasattr(dc_filtered_flat, 'A1'):
            dc_filtered_flat = dc_filtered_flat.A1
        else:
            dc_filtered_flat = dc_filtered_flat.flatten()
        dc_filtered = np.reshape(dc_filtered_flat / self.Hs, x.shape)
        
        # Bisection algorithm for Lagrange multiplier
        l1, l2 = 1e-9, 1e9
        move = 0.2
        
        # Ensure we have some negative sensitivities for the sqrt
        dc_filtered = np.minimum(dc_filtered, -1e-12)
        
        max_iterations = 20
        for iteration in range(max_iterations):
            if (l2 - l1) / (l1 + l2) < 1e-3:
                break
                
            lmid = 0.5 * (l2 + l1)
            
            # Update rule with safeguards
            sqrt_term = np.sqrt(np.maximum(-dc_filtered / lmid, 1e-12))
            x_cnew = x * sqrt_term
            x_cnew = np.minimum(1.0, np.minimum(x + move, x_cnew))
            x_cnew = np.maximum(0.0, np.maximum(x - move, x_cnew))
            
            # Check volume constraint
            if np.sum(x_cnew) > self.volfrac * self.nelx * self.nely:
                l1 = lmid
            else:
                l2 = lmid
        
        return x_cnew
    
    def optimize(self, max_iter=100, tol=1e-3, use_analytical_sensitivities=True):
        """
        Run SIMP optimization
        
        Args:
            max_iter: Maximum iterations
            tol: Convergence tolerance  
            use_analytical_sensitivities: Use fast analytical sensitivities (default True)
        """
        print(f"Starting SIMP Optimization")
        print(f"Design domain: {self.nelx} × {self.nely} elements")
        print(f"Volume fraction: {self.volfrac}")
        print(f"Penalty parameter: {self.penal}")
        print(f"Sensitivity method: {'Analytical' if use_analytical_sensitivities else 'Finite Differences'}")
        print("=" * 50)
        
        x = self.x.copy()
        
        for iteration in range(max_iter):
            start_time = time.time()
            
            # Sensitivity analysis - choose method
            if use_analytical_sensitivities:
                dc, compliance = self._analytical_sensitivity_analysis(x)
            else:
                dc, compliance = self._sensitivity_analysis(x)
            
            # Save history
            self.compliance_history.append(compliance)
            self.volume_history.append(np.sum(x) / (self.nelx * self.nely))
            self.density_history.append(x.copy())
            
            # Update densities
            x_new = self._optimality_criteria_update(x, dc)
            
            # Check convergence
            change = np.max(np.abs(x_new - x))
            
            # Update
            x = x_new
            
            iteration_time = time.time() - start_time
            
            print(f"Iter {iteration+1:3d}: Compliance = {compliance:.6e}, "
                  f"Volume = {np.sum(x)/(self.nelx*self.nely):.3f}, "
                  f"Change = {change:.6e}, Time = {iteration_time:.2f}s")
            
            # Check convergence
            if change < tol:
                print(f"Converged after {iteration+1} iterations")
                break
        
        self.x = x
        return x
    
    def plot_results(self):
        """Plot optimization results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Final design
        im1 = ax1.imshow(self.x, cmap='gray', origin='lower')
        ax1.set_title('Final Topology')
        ax1.set_xlabel('X elements')
        ax1.set_ylabel('Y elements')
        plt.colorbar(im1, ax=ax1)
        
        # Convergence history
        ax2.semilogy(self.compliance_history, 'b-', linewidth=2)
        ax2.set_title('Compliance History')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Compliance')
        ax2.grid(True, alpha=0.3)
        
        # Volume fraction history
        ax3.plot(self.volume_history, 'r-', linewidth=2)
        ax3.axhline(y=self.volfrac, color='k', linestyle='--', alpha=0.7, label=f'Target: {self.volfrac}')
        ax3.set_title('Volume Fraction History')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Volume Fraction')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Thresholded design (0/1)
        x_thresh = (self.x > 0.5).astype(float)
        im4 = ax4.imshow(x_thresh, cmap='gray', origin='lower')
        ax4.set_title('Thresholded Design (>0.5)')
        ax4.set_xlabel('X elements')
        ax4.set_ylabel('Y elements')
        plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('simp_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def to_devsim_geometry(self, threshold=0.5):
        """
        Convert SIMP result to DEVSIM-compatible geometry matrix
        """
        from geometry_optimization_framework import GeometryMatrix
        
        # Create geometry matrix
        geom = GeometryMatrix(self.nelx, self.nely, self.lx, self.ly)
        
        # Convert density to discrete materials
        # Threshold approach: density > threshold = semiconductor (N-type default)
        material_matrix = np.zeros_like(self.x, dtype=np.uint8)
        
        # Set material based on density and position
        for i in range(self.nely):
            for j in range(self.nelx):
                if self.x[i, j] > threshold:
                    # Determine P or N type based on position
                    x_pos = j / self.nelx
                    if x_pos < 0.5:
                        material_matrix[i, j] = 2  # P-type (left side)
                    else:
                        material_matrix[i, j] = 1  # N-type (right side)
                else:
                    material_matrix[i, j] = 0  # Void
        
        geom.material_matrix = material_matrix
        geom.enforce_contact_constraints()
        
        return geom
    
    def analyze_design(self):
        """Analyze the optimized design"""
        print("\n" + "="*50)
        print("SIMP DESIGN ANALYSIS")
        print("="*50)
        
        # Material statistics
        total_elements = self.nelx * self.nely
        volume_fraction = np.sum(self.x) / total_elements
        
        print(f"Total elements: {total_elements}")
        print(f"Final volume fraction: {volume_fraction:.3f}")
        print(f"Target volume fraction: {self.volfrac}")
        print(f"Volume constraint satisfied: {abs(volume_fraction - self.volfrac) < 0.01}")
        
        # Thresholded analysis
        x_thresh = (self.x > 0.5).astype(float)
        solid_elements = np.sum(x_thresh)
        void_elements = total_elements - solid_elements
        
        print(f"\nThresholded design (>0.5):")
        print(f"Solid elements: {solid_elements} ({solid_elements/total_elements:.3f})")
        print(f"Void elements: {void_elements} ({void_elements/total_elements:.3f})")
        
        # Connectivity analysis
        labeled, num_features = ndimage.label(x_thresh)
        print(f"Connected solid regions: {num_features}")
        
        # Gray elements (intermediate densities)
        gray_elements = np.sum((self.x > 0.1) & (self.x < 0.9))
        print(f"Gray elements (0.1 < ρ < 0.9): {gray_elements} ({gray_elements/total_elements:.3f})")
        
        # Performance
        final_compliance = self.compliance_history[-1] if self.compliance_history else 0
        print(f"\nFinal compliance: {final_compliance:.6e}")
        
        if len(self.compliance_history) > 1:
            improvement = (self.compliance_history[0] - final_compliance) / self.compliance_history[0] * 100
            print(f"Compliance improvement: {improvement:.1f}%")

def test_simp_optimization():
    """Test SIMP optimization for semiconductor device"""
    print("Testing SIMP Topology Optimization for Semiconductor Devices")
    print("=" * 70)
    
    # Test different configurations
    test_cases = [
        {"nelx": 20, "nely": 20, "volfrac": 0.4, "name": "Small_LowVol"},
        {"nelx": 30, "nely": 30, "volfrac": 0.5, "name": "Medium_MidVol"},
        {"nelx": 40, "nely": 40, "volfrac": 0.6, "name": "Large_HighVol"}
    ]
    
    results = {}
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['name']}")
        print("-" * 40)
        
        # Create optimizer
        simp = SIMPOptimizer(
            nelx=case["nelx"], 
            nely=case["nely"], 
            volfrac=case["volfrac"],
            penal=3.0,
            rmin=1.5
        )
        
        # Run optimization
        start_time = time.time()
        x_opt = simp.optimize(max_iter=50, tol=1e-3)
        optimization_time = time.time() - start_time
        
        # Analyze results
        simp.analyze_design()
        
        # Save results
        results[case["name"]] = {
            "optimizer": simp,
            "final_design": x_opt,
            "optimization_time": optimization_time,
            "compliance_history": simp.compliance_history.copy()
        }
        
        # Plot individual results
        simp.plot_results()
        plt.savefig(f'simp_{case["name"]}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Convert to DEVSIM geometry for validation
        try:
            devsim_geom = simp.to_devsim_geometry()
            print(f"✓ Successfully converted to DEVSIM geometry")
            
            # Visualize DEVSIM geometry
            fig = devsim_geom.visualize(f"DEVSIM Geometry - {case['name']}")
            plt.savefig(f'devsim_geom_{case["name"]}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"✗ DEVSIM conversion failed: {e}")
    
    # Comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot convergence comparison
    plt.subplot(2, 2, 1)
    for name, result in results.items():
        plt.semilogy(result["compliance_history"], label=name, linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Compliance')
    plt.title('Convergence Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot final designs
    for i, (name, result) in enumerate(results.items()):
        plt.subplot(2, 2, i+2)
        plt.imshow(result["final_design"], cmap='gray', origin='lower')
        plt.title(f'Final Design: {name}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('simp_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n" + "="*70)
    print("SIMP OPTIMIZATION TEST COMPLETE")
    print("="*70)
    print(f"Generated {len(results)} optimized topologies")
    print(f"All results saved as PNG files")
    
    return results

if __name__ == "__main__":
    test_simp_optimization()