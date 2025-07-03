// Gmsh geometry file for a 2D P-N diode with refined mesh at the junction

// Physical and geometric parameters
total_length_m = 10.0e-6;
width_m = 5.0e-6;
contact_length_m = 2.0e-6;
junction_length_m = 6.0e-6;
junction_center = contact_length_m + junction_length_m / 2;

// Define mesh sizes
mesh_size_coarse = 1.0e-6;   // Coarse mesh size in contact regions
mesh_size_fine = 0.02e-6;    // Very fine mesh size at the junction (20 nm)
mesh_transition_dist = 2.0e-6; // Distance over which the mesh transitions

// 1. Define Geometry
// Point at the center of the junction for refinement
Point(1) = {junction_center, width_m/2, 0};

// Points of the rectangle
Point(2) = {0, 0, 0};
Point(3) = {total_length_m, 0, 0};
Point(4) = {total_length_m, width_m, 0};
Point(5) = {0, width_m, 0};

// Lines forming the rectangle
Line(1) = {2, 3}; // Bottom
Line(2) = {3, 4}; // N-contact (cathode)
Line(3) = {4, 5}; // Top
Line(4) = {5, 2}; // P-contact (anode)

// Surface
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// Embed the refinement point in the surface
Point{1} In Surface{1};

// 2. Define Mesh Refinement
// Field 1: Distance from the central refinement point
Field[1] = Distance;
Field[1].PointsList = {1};

// Field 2: Threshold field to define mesh size based on distance from Field 1
Field[2] = Threshold;
Field[2].IField = 1; // Use the distance field (Field 1) as input
Field[2].LcMin = mesh_size_fine;
Field[2].LcMax = mesh_size_coarse;
Field[2].DistMin = mesh_transition_dist / 2;
Field[2].DistMax = mesh_transition_dist;

// Use the Threshold field as the background field for meshing
Background Field = 2;

// 3. Define Physical Groups for FEniCSx
Physical Curve("p_contact", 1) = {4};
Physical Curve("n_contact", 2) = {2};
Physical Curve("insulating", 3) = {1, 3};
Physical Surface("domain", 4) = {1};