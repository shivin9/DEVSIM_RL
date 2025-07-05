// Simple 2D rectangular diode geometry
// Replicates the working gmsh_diode2d.msh structure

// Parameters
L = 1e-5;      // Length: 10 μm
W = 1e-5;      // Width: 10 μm
mesh_size = 5e-7;  // Mesh size: 0.5 μm

// Define corner points
Point(1) = {0, 0, 0, mesh_size};
Point(2) = {L, 0, 0, mesh_size};
Point(3) = {L, W, 0, mesh_size};
Point(4) = {0, W, 0, mesh_size};

// Define boundary lines
Line(1) = {1, 2};  // Bottom edge
Line(2) = {2, 3};  // Right edge
Line(3) = {3, 4};  // Top edge
Line(4) = {4, 1};  // Left edge

// Define surface
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Physical definitions (matching working example)
Physical Curve("Base") = {4};      // Left edge contact (P+ contact)
Physical Curve("Emitter") = {2};   // Right edge contact (N+ contact)
Physical Surface("Bulk") = {1};    // Bulk region

// Mesh options
Mesh.Algorithm = 6;        // Frontal-Delaunay
Mesh.ElementOrder = 1;     // Linear elements