# File: setup_mosfet.py
# Description: Defines a simple 2D MOSFET structure in DEVSIM.
# This should be run once before calling run_devsim_simulation repeatedly.

import devsim
import numpy as np
import logging
import sys

logger = logging.getLogger(__name__)

# --- Device Dimensions and Mesh Parameters ---
# Dimensions in Microns (converted to cm later for DEVSIM)
UM_TO_CM = 1e-4
NM_TO_CM = 1e-7

# Geometry
L_GATE = 0.1   # Gate Length (um)
L_SD = 0.1     # Source/Drain Length (um)
W_DEVICE = 1.0 # Device width (into the 2D plane, used for current normalization - typically 1 um)

H_OXIDE_NM = 1.5 # Initial default oxide thickness (nm) - WILL BE OVERWRITTEN by RL agent via set_parameter
H_SILICON = 0.2  # Silicon substrate thickness (um)
H_GATE = 0.05    # Gate contact height (arbitrary for simulation setup)

# Mesh Spacing (crucial for convergence and accuracy, refine significantly!)
spacing_ox_y = 0.1 * NM_TO_CM  # Fine spacing in oxide
spacing_si_y = 0.5 * NM_TO_CM  # Fine spacing near interface in Si
spacing_bulk_y = 10 * NM_TO_CM # Coarser spacing in bulk Si
spacing_lat_x = 5 * NM_TO_CM   # Lateral spacing under gate/S-D junctions
spacing_sd_x = 10 * NM_TO_CM   # Coarser spacing in S/D contacts

# --- Device Name and Regions/Contacts ---
DEVICE_NAME = "my_mosfet"
REGION_SILICON = "substrate"
REGION_OXIDE = "oxide"
CONTACT_GATE = "gate"
CONTACT_SOURCE = "source"
CONTACT_DRAIN = "drain"
CONTACT_BODY = "body"
INTERFACE_SI_OX = "si_oxide_interface"

# --- Physical Constants ---
q = 1.602176634e-19 # Elementary Charge (C)
eps_0 = 8.854187817e-14 # Permittivity of vacuum (F/cm)
k_boltzmann = 8.617333262e-5 # Boltzmann constant (eV/K)
T = 300.0              # Temperature (K)

# Material Permittivities (relative)
eps_si = 11.7
eps_ox = 3.9

# Silicon Intrinsic Carrier Concentration (cm^-3) at 300K
n_i_si = 1.0e10

# Doping Concentrations (cm^-3)
N_SOURCE_DRAIN = 1.0e20 # N+ doping for Source/Drain
N_SUBSTRATE_DEFAULT = 5e17 # P-type substrate doping - WILL BE OVERWRITTEN by RL agent

def create_mosfet_structure(default_tox_nm=H_OXIDE_NM, default_nsub_cm3=N_SUBSTRATE_DEFAULT):
    """
    Creates the basic 2D MOSFET mesh, regions, contacts, materials,
    doping, and physics models in DEVSIM.
    """
    logger.info("--- Starting DEVSIM MOSFET Structure Creation ---")

    # Convert dimensions to cm
    h_oxide = default_tox_nm * NM_TO_CM
    h_silicon = H_SILICON * UM_TO_CM
    h_gate = H_GATE * UM_TO_CM
    l_gate = L_GATE * UM_TO_CM
    l_sd = L_SD * UM_TO_CM

    # Calculate coordinates
    x_src_left = 0.0
    x_src_right = l_sd
    x_gate_left = x_src_right
    x_gate_right = x_gate_left + l_gate
    x_drain_left = x_gate_right
    x_drain_right = x_drain_left + l_sd
    x_max = x_drain_right

    y_gate_top = h_oxide + h_gate
    y_oxide_top = h_oxide
    y_si_top = 0.0
    y_si_bottom = -h_silicon

    # --- 1. Create Mesh ---
    logger.info("Creating 2D mesh...")
    mesh_name = DEVICE_NAME # Use device name for mesh name for simplicity
    devsim.create_2d_mesh(mesh=mesh_name)

    # Define horizontal lines (X-coordinates)
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=x_src_left, ps=spacing_sd_x)
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=x_src_right, ps=spacing_lat_x)
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=x_gate_right, ps=spacing_lat_x)
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="x", pos=x_drain_right, ps=spacing_sd_x)

    # Define vertical lines (Y-coordinates)
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=y_gate_top, ps=h_gate)       # Top of gate contact
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=y_oxide_top, ps=spacing_ox_y) # Top of oxide / Gate bottom
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=y_si_top, ps=spacing_si_y)    # Si/Oxide interface
    # Add some refinement near the surface in Si
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=-0.05*h_silicon, ps=spacing_si_y*2)
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=-0.2*h_silicon, ps=spacing_bulk_y)
    devsim.add_2d_mesh_line(mesh=mesh_name, dir="y", pos=y_si_bottom, ps=spacing_bulk_y) # Bottom of Si / Body contact

    # --- 2. Define Regions ---
    logger.info("Defining regions...")
    devsim.add_2d_region(mesh=mesh_name, material="Oxide", region=REGION_OXIDE, xl=x_gate_left, xh=x_gate_right, yl=y_si_top, yh=y_oxide_top)
    devsim.add_2d_region(mesh=mesh_name, material="Silicon", region=REGION_SILICON, xl=x_src_left, xh=x_drain_right, yl=y_si_bottom, yh=y_si_top)
    # Note: Implicit assumption is only these two rectangular regions for simplicity.
    # A real mesh might have more complex region definitions.

    # --- 3. Define Contacts ---
    logger.info("Defining contacts...")
    # Source Contact (left side surface)
    devsim.add_2d_contact(mesh=mesh_name, name=CONTACT_SOURCE, material="metal", region=REGION_SILICON, xl=x_src_left, xh=x_src_right, yl=y_si_top, yh=y_si_top)
    # Drain Contact (right side surface)
    devsim.add_2d_contact(mesh=mesh_name, name=CONTACT_DRAIN, material="metal", region=REGION_SILICON, xl=x_drain_left, xh=x_drain_right, yl=y_si_top, yh=y_si_top)
    # Gate Contact (top of oxide)
    devsim.add_2d_contact(mesh=mesh_name, name=CONTACT_GATE, material="metal", region=REGION_OXIDE, xl=x_gate_left, xh=x_gate_right, yl=y_oxide_top, yh=y_oxide_top)
    # Body Contact (bottom of substrate)
    devsim.add_2d_contact(mesh=mesh_name, name=CONTACT_BODY, material="metal", region=REGION_SILICON, xl=x_src_left, xh=x_drain_right, yl=y_si_bottom, yh=y_si_bottom)

    # --- 4. Define Interface ---
    logger.info("Defining interface...")
    devsim.add_2d_interface(mesh=mesh_name, name=INTERFACE_SI_OX, region0=REGION_SILICON, region1=REGION_OXIDE, xl=x_gate_left, xh=x_gate_right, yl=y_si_top, yh=y_si_top)

    # --- 5. Finalize Mesh and Create Device ---
    logger.info("Finalizing mesh and creating device...")
    devsim.finalize_mesh(mesh=mesh_name)
    devsim.create_device(mesh=mesh_name, device=DEVICE_NAME)

    # --- 6. Set Material Parameters ---
    logger.info("Setting material parameters...")
    # Silicon Parameters
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="Permittivity", value=eps_si * eps_0)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="ElectronCharge", value=q)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="n_i", value=n_i_si)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="T", value=T)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="kT", value=k_boltzmann * T)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="Vt", value=k_boltzmann * T) # Thermal voltage in Volts
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="mu_n", value=400.0) # Simple constant electron mobility (cm^2/V*s)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="mu_p", value=200.0) # Simple constant hole mobility (cm^2/V*s)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="taun", value=1e-7)  # Electron lifetime for SRH (s)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_SILICON, name="taup", value=1e-7)  # Hole lifetime for SRH (s)

    # Oxide Parameters
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_OXIDE, name="Permittivity", value=eps_ox * eps_0)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_OXIDE, name="ElectronCharge", value=q)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_OXIDE, name="T", value=T)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_OXIDE, name="kT", value=k_boltzmann * T)
    devsim.set_parameter(device=DEVICE_NAME, region=REGION_OXIDE, name="Vt", value=k_boltzmann * T)

    # Parameters controlled by RL agent (set default values here, overwritten later)
    devsim.set_parameter(device=DEVICE_NAME, name="oxide_thickness", value=h_oxide) # This name must match devsim_runner
    devsim.set_parameter(device=DEVICE_NAME, name="substrate_doping", value=default_nsub_cm3) # This name must match devsim_runner

    # --- 7. Define Doping Profiles ---
    logger.info("Defining doping profiles...")
    # Substrate Doping (P-type) - Value comes from parameter set by RL agent
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="Acceptors",
                      equation=f"{default_nsub_cm3}") # Use default for setup
    # Modify Acceptors based on the parameter that will be set by the RL agent
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="Acceptors",
                      equation="substrate_doping")


    # Source/Drain Doping (N-type, simple step functions)
    # Use x coordinates defined earlier
    eq_donors = f"{N_SOURCE_DRAIN}*(step({x_src_right}-x) + step(x-{x_drain_left}))"
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="Donors",
                      equation=eq_donors)

    # Net Doping
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="NetDoping",
                      equation="Donors - Acceptors")

    # --- 8. Define Physics Models & Equations ---
    logger.info("Setting up physics models and equations...")

    # Create potential, electron, and hole solution variables
    devsim.node_solution(device=DEVICE_NAME, region=REGION_SILICON, name="Potential")
    devsim.node_solution(device=DEVICE_NAME, region=REGION_SILICON, name="Electrons")
    devsim.node_solution(device=DEVICE_NAME, region=REGION_SILICON, name="Holes")

    # --- Silicon Models ---
    # Electric field
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="ElectricField",
                      equation="(Potential@n0 - Potential@n1)*EdgeInverseLength")
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="Potential_edge",
                      equation="(Potential@n0 + Potential@n1)*0.5") # For mobility models

    # Electron/Hole concentration using Boltzmann statistics (simplification)
    # More accurate: use Fermi statistics, especially for high doping/low temp
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="IntrinsicElectrons",
                      equation="n_i*exp(Potential/Vt)")
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="IntrinsicHoles",
                       equation="n_i*exp(-Potential/Vt)")
    # Initialize solution variables based on doping and intrinsic potential
    devsim.set_node_value(device=DEVICE_NAME, region=REGION_SILICON, name="Potential")
    devsim.set_node_value(device=DEVICE_NAME, region=REGION_SILICON, name="Electrons")
    devsim.set_node_value(device=DEVICE_NAME, region=REGION_SILICON, name="Holes")


    # Current calculations (Scharfetter-Gummel)
    # Electron current Jn = q * mu_n * n * E + q * Dn * grad(n) --> Use SG form
    devsim.edge_from_node_model(device=DEVICE_NAME, region=REGION_SILICON, node_model="Potential")
    devsim.edge_from_node_model(device=DEVICE_NAME, region=REGION_SILICON, node_model="Electrons")
    devsim.edge_from_node_model(device=DEVICE_NAME, region=REGION_SILICON, node_model="Holes")

    # Create Bernoulli functions needed for SG currents
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="vdiff", equation="Potential@n0 - Potential@n1")
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="vdiff_norm", equation="vdiff/Vt")
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="Bernoulli_n", equation="B(vdiff_norm)")
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="Bernoulli_p", equation="B(-vdiff_norm)")
    # devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="DBernoulli_n", equation="DBDX(vdiff_norm)") # Derivatives needed if not using built-in SG
    # devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="DBernoulli_p", equation="DBDX(-vdiff_norm)")

    # Electron Current (using simplified SG form available via equation variables)
    # Jn = -q * mu_n * Vt * EdgeCouple * (Electrons@n1*B(Vd/Vt) - Electrons@n0*B(-Vd/Vt))
    # Note: devsim's SG implementation might be slightly different; check manual/examples.
    # Using a more direct approach often found in examples:
    eq_Jn = "ElectronCharge*mu_n*EdgeCouple*Vt*(Electrons@n1*Bernoulli_n - Electrons@n0*Bernoulli_p)"
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="ElectronCurrent", equation=eq_Jn)

    # Hole Current
    # Jp = -q * mu_p * Vt * EdgeCouple * (Holes@n1*B(-Vd/Vt) - Holes@n0*B(Vd/Vt))
    eq_Jp = "-ElectronCharge*mu_p*EdgeCouple*Vt*(Holes@n1*Bernoulli_p - Holes@n0*Bernoulli_n)"
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="HoleCurrent", equation=eq_Jp)

    # SRH Recombination model (basic)
    srh_expr = "ElectronCharge * (Electrons*Holes - n_i^2) / (taup*(Electrons + n_i) + taun*(Holes + n_i))"
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="SRH_Recombination", equation=srh_expr)

    # --- Silicon Equations ---
    # Poisson Equation: div(eps * grad(Potential)) = -rho = -q * (p - n + Nd - Na)
    devsim.equation(device=DEVICE_NAME, region=REGION_SILICON, name="PotentialEquation", variable_name="Potential",
                    node_model="", edge_model="ElectricField", variable_update="default",
                    edge_charge_model="PotentialEdgeFlux", node_charge_model="PotentialNodeCharge")
    # Define the charge models used in Poisson
    devsim.edge_model(device=DEVICE_NAME, region=REGION_SILICON, name="PotentialEdgeFlux", equation="Permittivity*ElectricField")
    devsim.node_model(device=DEVICE_NAME, region=REGION_SILICON, name="PotentialNodeCharge", equation="-ElectronCharge*(Holes - Electrons + NetDoping)")

    # Electron Continuity: dN/dt + div(Jn)/q - Un = 0 --> div(Jn) = q*Un (for DC)
    devsim.equation(device=DEVICE_NAME, region=REGION_SILICON, name="ElectronContinuityEquation", variable_name="Electrons",
                    node_model="SRH_Recombination", edge_model="ElectronCurrent", variable_update="positive") # 'positive' ensures n>=0

    # Hole Continuity: dP/dt - div(Jp)/q - Up = 0 --> div(Jp) = -q*Up (for DC)
    devsim.equation(device=DEVICE_NAME, region=REGION_SILICON, name="HoleContinuityEquation", variable_name="Holes",
                    node_model="SRH_Recombination", edge_model="-HoleCurrent", variable_update="positive") # 'positive' ensures p>=0


    # --- Oxide Models and Equations ---
    # Only need Poisson equation in oxide: div(eps_ox * grad(Potential)) = 0 (no charge)
    devsim.node_solution(device=DEVICE_NAME, region=REGION_OXIDE, name="Potential")
    devsim.edge_model(device=DEVICE_NAME, region=REGION_OXIDE, name="ElectricField",
                      equation="(Potential@n0 - Potential@n1)*EdgeInverseLength")
    devsim.edge_model(device=DEVICE_NAME, region=REGION_OXIDE, name="PotentialEdgeFlux", equation="Permittivity*ElectricField")
    devsim.equation(device=DEVICE_NAME, region=REGION_OXIDE, name="PotentialEquation", variable_name="Potential",
                    edge_model="ElectricField", variable_update="default",
                    edge_charge_model="PotentialEdgeFlux")
    # Initialize potential in oxide (e.g., copy from silicon interface or set to 0)
    devsim.set_node_value(device=DEVICE_NAME, region=REGION_OXIDE, name="Potential", value=0.0)

    # --- Interface Conditions (Si/Oxide) ---
    # Potential continuity is implicitly handled by sharing nodes if mesh is conformal.
    # We might need specific interface models if charge trapping etc., is considered.
    # For simplicity, we assume continuity is handled by the solver/mesh setup.
    # If needed, use interface_model and interface_equation.
    # Example: Ensure potential continuity explicitly (often needed for non-conformal meshes)
    devsim.interface_model(device=DEVICE_NAME, interface=INTERFACE_SI_OX, name="continuousPotential",
                           equation="Potential@r0 - Potential@r1")
    devsim.interface_equation(device=DEVICE_NAME, interface=INTERFACE_SI_OX, name="PotentialEquation",
                              interface_model="continuousPotential", type="continuous")


    # --- Contact Boundary Conditions ---
    logger.info("Setting contact boundary conditions...")
    # Source, Drain, Body (Ohmic) - Set Potential, and equilibrium carrier concentrations
    for contact in [CONTACT_SOURCE, CONTACT_DRAIN, CONTACT_BODY]:
        devsim.contact_equation(device=DEVICE_NAME, contact=contact, name="PotentialEquation",
                                variable_name="Potential", node_model="Potential",
                                edge_charge_model="PotentialEdgeFlux")
        # Set equilibrium boundary conditions for carriers at ohmic contacts
        # n = n_i * exp( (phi_n - phi)/Vt ) => phi_n = phi + Vt * log(n/n_i)
        # p = n_i * exp( (phi - phi_p)/Vt ) => phi_p = phi - Vt * log(p/n_i)
        # For ohmic contacts phi_n = phi_p = V_contact
        # n ~ NetDoping if NetDoping > 0, else n ~ n_i^2/abs(NetDoping)
        # p ~ abs(NetDoping) if NetDoping < 0, else p ~ n_i^2/NetDoping
        # Simplified: Assume equilibrium based on NetDoping at contact edge
        devsim.contact_node_model(device=DEVICE_NAME, contact=contact, name="contact_electrons",
                                  equation="ifelse(NetDoping > 0, NetDoping, n_i^2/abs(NetDoping))") # Approximation
        devsim.contact_node_model(device=DEVICE_NAME, contact=contact, name="contact_holes",
                                  equation="ifelse(NetDoping < 0, abs(NetDoping), n_i^2/NetDoping)") # Approximation

        devsim.contact_equation(device=DEVICE_NAME, contact=contact, name="ElectronContinuityEquation",
                                variable_name="Electrons", node_model="contact_electrons",
                                edge_current_model="ElectronCurrent")
        devsim.contact_equation(device=DEVICE_NAME, contact=contact, name="HoleContinuityEquation",
                                variable_name="Holes", node_model="contact_holes",
                                edge_current_model="-HoleCurrent") # Note the minus sign convention

    # Gate Contact (Schottky or Ideal Insulator) - Set Potential only
    devsim.contact_equation(device=DEVICE_NAME, contact=contact, name="PotentialEquation",
                            variable_name="Potential", node_model="Potential",
                            edge_charge_model="PotentialEdgeFlux", region=REGION_OXIDE) # Apply in oxide region

    # Set initial bias parameters (will be overwritten during sweeps)
    devsim.set_parameter(device=DEVICE_NAME, name="Vgate", value=0.0)
    devsim.set_parameter(device=DEVICE_NAME, name="Vsource", value=0.0)
    devsim.set_parameter(device=DEVICE_NAME, name="Vdrain", value=0.0)
    devsim.set_parameter(device=DEVICE_NAME, name="Vbody", value=0.0) # Often tied to source (0V)

    # --- 9. Initial Solve ---
    logger.info("Performing initial solve...")
    try:
        devsim.solve(type="dc", absolute_error=1.0e-12, relative_error=1.0e-10, maximum_iterations=30)
        logger.info("Initial solve successful.")
    except Exception as e:
        logger.error(f"Initial solve failed: {e}")
        logger.error("Check mesh, doping, models, and boundary conditions.")
        raise RuntimeError("DEVSIM initial setup solve failed.") from e

    logger.info("--- DEVSIM MOSFET Structure Creation Complete ---")

# --- Main execution block (optional, for testing setup script) ---
if __name__ == "__main__":
    try:
        print("Running DEVSIM MOSFET setup script...")
        create_mosfet_structure()
        print("DEVSIM setup script completed successfully.")
        # Optional: Write out the initial device state for visualization
        devsim.write_devices(file="mosfet_initial_setup.tec", type="tecplot")
        print("Initial device state saved to mosfet_initial_setup.tec")
    except ImportError:
        print("Import Error: DEVSIM module not found. Cannot run setup.", file=sys.stderr)
    except RuntimeError as e:
        print(f"Runtime Error during setup: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during setup: {e}", file=sys.stderr)