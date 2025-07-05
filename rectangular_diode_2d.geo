// GMSH geometry file for 2D rectangular diode
// Replicates the exact geometry from devsim_2d_diode_corrected.py
// 10μm × 10μm rectangle with P-N junction at x = 5μm

// Geometry parameters (in meters)
device_length = 1e-5;      // 10 μm
device_width = 1e-5;       // 10 μm
junction_pos = 0.5e-5;     // 5 μm (P-N junction position)

// Mesh size parameters
mesh_coarse = 1e-6;        // 1 μm (coarse mesh)
mesh_fine = 1e-8;          // 0.01 μm (fine mesh near junction)
mesh_contact = 1e-8;       // 0.01 μm (fine mesh at contacts)

// Contact parameters
contact_width = 0.2e-5;    // 2 μm contact width (matches original yl=0.8e-5 to yh=1e-5)

// Define points for the rectangle
Point(1) = {0, 0, 0, mesh_contact};                    // Bottom-left corner
Point(2) = {device_length, 0, 0, mesh_contact};       // Bottom-right corner
Point(3) = {device_length, device_width, 0, mesh_coarse}; // Top-right corner
Point(4) = {0, device_width, 0, mesh_coarse};         // Top-left corner

// Define points for P-N junction line (vertical line at x = junction_pos)
Point(5) = {junction_pos, 0, 0, mesh_fine};           // Junction bottom
Point(6) = {junction_pos, device_width, 0, mesh_fine}; // Junction top

// Define points for contact regions
// Top contact (P+ side) - line contact at x=0, y from 0.8e-5 to 1e-5
Point(7) = {0, 0.8e-5, 0, mesh_contact};              // Top contact start
Point(8) = {0, device_width, 0, mesh_contact};        // Top contact end (Point 4)

// Bottom contact (N+ side) - line contact at x=device_length
Point(9) = {device_length, 0, 0, mesh_contact};       // Bottom contact start (Point 2)
Point(10) = {device_length, device_width, 0, mesh_contact}; // Bottom contact end (Point 3)

// Define lines for the rectangle boundary
Line(1) = {1, 5};          // Bottom edge: left to junction
Line(2) = {5, 2};          // Bottom edge: junction to right
Line(3) = {2, 10};         // Right edge: bottom to top
Line(4) = {10, 6};         // Top edge: right to junction
Line(5) = {6, 4};          // Top edge: junction to left
Line(6) = {4, 7};          // Left edge: top to contact start
Line(7) = {7, 1};          // Left edge: contact start to bottom

// Define the P-N junction line
Line(8) = {5, 6};          // P-N junction (vertical line)

// Define contact lines
Line(9) = {7, 4};          // Top contact line (P+ contact)
Line(10) = {2, 10};        // Bottom contact line (N+ contact) - same as Line(3)

// Define surfaces (regions)
// Single bulk region covering the entire diode
Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Plane Surface(1) = {1};

// Set mesh refinement near junction
Field[1] = Distance;
Field[1].CurvesList = {8}; // P-N junction line
Field[1].Sampling = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = mesh_fine;
Field[2].SizeMax = mesh_coarse;
Field[2].DistMin = 1e-6;   // 1 μm from junction
Field[2].DistMax = 3e-6;   // 3 μm from junction

// Set mesh refinement near contacts
Field[3] = Distance;
Field[3].CurvesList = {9, 10}; // Contact lines
Field[3].Sampling = 100;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = mesh_contact;
Field[4].SizeMax = mesh_coarse;
Field[4].DistMin = 0.5e-6; // 0.5 μm from contacts
Field[4].DistMax = 2e-6;   // 2 μm from contacts

// Combine mesh fields
Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

// Physical regions (for DEVSIM)
Physical Surface("Bulk") = {1};   // Single bulk region with P-N junction

// Physical boundaries/contacts (for DEVSIM)
Physical Curve("top_contact") = {9};   // P+ contact (top)
Physical Curve("bot_contact") = {10};  // N+ contact (bottom)

// Physical boundary lines (for boundary conditions)
Physical Curve("p_n_junction") = {8};  // P-N junction
Physical Curve("boundary") = {1, 2, 3, 4, 5, 6, 7}; // Outer boundary

// Mesh generation options
Mesh.Algorithm = 6;        // Frontal-Delaunay
Mesh.ElementOrder = 1;     // Linear elements
Mesh.Optimize = 1;         // Optimize mesh quality