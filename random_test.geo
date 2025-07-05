// Auto-generated GMSH geometry from RL optimization
// Dimensions: 20.0 x 20.0 μm

mesh_size = 1e-6;

// Background domain
Point(1) = {0, 0, 0, mesh_size};
Point(2) = {2e-05, 0, 0, mesh_size};
Point(3) = {2e-05, 2e-05, 0, mesh_size};
Point(4) = {0, 2e-05, 0, mesh_size};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Contacts
Physical Curve("P_contact") = {4};  // Left edge
Physical Curve("N_contact") = {2};  // Right edge

// Physical region (single bulk)
Physical Surface("Bulk") = {1};

// Mesh options
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;