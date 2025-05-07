# file: rl_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
# NOTE: devsim import might not be needed here if execution is truly deferred,
# but needed if we execute the script at the end for reward.
try:
    import devsim
except ImportError:
    print("WARNING: DEVSIM Python module not found. Final execution inside env will fail.")
    devsim = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Placeholder for final simulation (if executed by env) ---
def run_final_diode_simulation(design_params):
    # This function would normally run the DEVSIM IV sweep using the fully defined device
    # It needs access to the current DEVSIM state after the script executes.
    # For now, it remains a placeholder calculating metrics based on params.
    logger.info(f"Placeholder: Calculating final diode metrics based on params: {design_params}")
    sim_success = np.random.rand() > 0.1
    if sim_success:
        na = design_params.get('na', 1e17); nd = design_params.get('nd', 1e17)
        vf = 0.6 + 0.1 * np.log10(max(1e15, 1e17 / na)) + 0.1 * np.log10(max(1e15, 1e17 / nd))
        if_val = 1e-4 * (na/1e17)**0.5 * (nd/1e17)**0.5 * np.exp(vf / (1.5 * 0.0259))
        ir_val = 1e-11 * (1e17/na)**0.3 * (1e17/nd)**0.3
        logger.info(f"Placeholder Metrics: If={if_val:.2e}, Ir={ir_val:.2e}, Vf={vf:.3f}")
        return {'if': if_val, 'ir': ir_val, 'vf': vf}, True
    else:
        logger.warning("Placeholder Sim Failure")
        return {}, False

# --- Checklist Definition ---
DIODE_CHECKLIST = [
    "INIT_MESH", "DEFINE_GEOMETRY", "DEFINE_REGIONS", "DEFINE_CONTACTS",
    "FINALIZE_MESH", "CREATE_DEVICE", "SET_MATERIAL_PARAMS",
    "DEFINE_P_DOPING", "DEFINE_N_DOPING", "DEFINE_NET_DOPING",
    "DEFINE_VARIABLES", "SETUP_PHYSICS", "SETUP_EQUATIONS",
    "SETUP_CONTACT_BC", "FINALIZE_SETUP_RUN_TEST" # Triggers execution
]
NUM_CHECKLIST_ITEMS = len(DIODE_CHECKLIST)

# Indices, Obs Dim, Defaults
PARAM_NA_IDX = NUM_CHECKLIST_ITEMS + 0; PARAM_ND_IDX = NUM_CHECKLIST_ITEMS + 1
STEP_COUNT_IDX = NUM_CHECKLIST_ITEMS + 2; OBS_DIM = NUM_CHECKLIST_ITEMS + 3
DEFAULT_NA = 1e17; DEFAULT_ND = 1e17
DEVICE_LENGTH_UM = 1.0; JUNCTION_POS_UM = 0.5; UM_TO_CM = 1e-4
MESH_NAME = "diode_mesh"; DEVICE_NAME = "MyDiode"; REGION_NAME = "Silicon"
CONTACT_ANODE = "Anode"; CONTACT_CATHODE = "Cathode"

class DiodeDesignEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, max_steps=NUM_CHECKLIST_ITEMS + 3, target_metrics=None, execute_on_finalize=True):
        super().__init__()
        self._max_steps = max_steps
        self.target_metrics = target_metrics or {"if_min": 1e-4, "ir_max": 1e-9}
        self.execute_on_finalize = execute_on_finalize # Control if env runs script

        # Internal state
        self._completed_steps = set()
        self._design_params = {} # Stores key values like Na, Nd
        self._generated_script_lines = [] # Stores lines of generated Python code

        # Obs/Action Spaces
        self.na_bounds_log10 = (15.0, 19.0); self.nd_bounds_log10 = (15.0, 19.0)
        low_bounds = np.array([0.0] * NUM_CHECKLIST_ITEMS + [0.0] * 2 + [0.0], dtype=np.float32)
        high_bounds = np.array([1.0] * NUM_CHECKLIST_ITEMS + [1.0] * 2 + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_CHECKLIST_ITEMS)

        self._current_step = 0
        self._geometry_params = {'device_length_cm': DEVICE_LENGTH_UM * UM_TO_CM, 'junction_pos_cm': JUNCTION_POS_UM * UM_TO_CM}
        logger.info("DiodeDesignEnv initialized (Script Generation Mode).")

    def _normalize_param(self, value, bounds_log10):
        # ... (same as before) ...
        if value is None or value <= 0: return 0.0
        val_log10 = np.log10(value)
        norm_val = (val_log10 - bounds_log10[0]) / (bounds_log10[1] - bounds_log10[0])
        return np.clip(norm_val, 0.0, 1.0)


    def _get_observation(self):
        # ... (same as before, reads _completed_steps and _design_params) ...
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i, item in enumerate(DIODE_CHECKLIST):
            if item in self._completed_steps: obs[i] = 1.0
        na_val = self._design_params.get('na', None)
        nd_val = self._design_params.get('nd', None)
        obs[PARAM_NA_IDX] = self._normalize_param(na_val, self.na_bounds_log10)
        obs[PARAM_ND_IDX] = self._normalize_param(nd_val, self.nd_bounds_log10)
        obs[STEP_COUNT_IDX] = float(self._current_step) / float(self._max_steps)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # No need to reinit devsim here if script is executed externally
        # However, if self.execute_on_finalize is True, we DO need it.
        if self.execute_on_finalize and devsim:
             try:
                 devsim.reinit()
                 logger.info("DEVSIM reinitialized for internal execution.")
             except Exception as e:
                 logger.error(f"devsim.reinit() failed: {e}")
        elif not devsim and self.execute_on_finalize:
             logger.error("execute_on_finalize=True but devsim module failed to import.")

        self._completed_steps = set()
        self._design_params = {}
        self._generated_script_lines = [] # Reset script buffer
        self._current_step = 0
        self._geometry_params = {'device_length_cm': DEVICE_LENGTH_UM * UM_TO_CM, 'junction_pos_cm': JUNCTION_POS_UM * UM_TO_CM}
        logger.info("DiodeDesignEnv reset (Script Generation Mode).")
        observation = self._get_observation()
        info = {"completed_steps": sorted(list(self._completed_steps)), "params": self._design_params, "script": ""}
        return observation, info

    def _generate_devsim_step_code(self, checklist_item_index):
        """
        Generates a string containing the Python code for a DEVSIM setup step.
        Returns the code string on success, None on dependency failure.
        Updates self._design_params if the step defines Na/Nd.
        """
        item_name = DIODE_CHECKLIST[checklist_item_index]
        logger.info(f"Generating code for step '{item_name}'")

        # --- Check Dependencies ---
        required_deps = {
            "DEFINE_GEOMETRY": {"INIT_MESH"}, "DEFINE_REGIONS": {"DEFINE_GEOMETRY"},
            "DEFINE_CONTACTS": {"DEFINE_REGIONS"}, "FINALIZE_MESH": {"DEFINE_CONTACTS", "DEFINE_REGIONS"},
            "CREATE_DEVICE": {"FINALIZE_MESH"}, "SET_MATERIAL_PARAMS": {"CREATE_DEVICE"},
            "DEFINE_P_DOPING": {"CREATE_DEVICE", "DEFINE_REGIONS"},
            "DEFINE_N_DOPING": {"CREATE_DEVICE", "DEFINE_REGIONS"},
            "DEFINE_NET_DOPING": {"DEFINE_P_DOPING", "DEFINE_N_DOPING"},
            "DEFINE_VARIABLES": {"CREATE_DEVICE"},
            "SETUP_PHYSICS": {"CREATE_DEVICE", "SET_MATERIAL_PARAMS"},
            "SETUP_EQUATIONS": {"DEFINE_VARIABLES", "SETUP_PHYSICS", "DEFINE_NET_DOPING"},
            "SETUP_CONTACT_BC": {"CREATE_DEVICE", "DEFINE_CONTACTS", "SETUP_EQUATIONS"}
        }
        deps = required_deps.get(item_name, set())
        if not deps.issubset(self._completed_steps):
            logger.error(f"Dependency failed for '{item_name}'. Requires: {deps}, Have: {self._completed_steps}")
            return None # Indicate failure

        # --- Generate Code String ---
        code = f"\n# --- Code for Step: {item_name} ---\n"
        try:
            # Use f-strings to build the code lines
            if item_name == "INIT_MESH":
                code += f"devsim.create_1d_mesh(mesh='{MESH_NAME}')\n"
            elif item_name == "DEFINE_GEOMETRY":
                L = self._geometry_params['device_length_cm']
                xj = self._geometry_params['junction_pos_cm']
                code += f"devsim.add_1d_mesh_line(mesh='{MESH_NAME}', pos=0.0, ps={L*0.05}, tag='{CONTACT_ANODE}')\n"
                code += f"devsim.add_1d_mesh_line(mesh='{MESH_NAME}', pos={xj}, ps={L*0.001})\n"
                code += f"devsim.add_1d_mesh_line(mesh='{MESH_NAME}', pos={L}, ps={L*0.05}, tag='{CONTACT_CATHODE}')\n"
            elif item_name == "DEFINE_REGIONS":
                code += f"devsim.add_1d_region(mesh='{MESH_NAME}', region='{REGION_NAME}', material='Silicon', tag1='{CONTACT_ANODE}', tag2='{CONTACT_CATHODE}')\n"
            elif item_name == "DEFINE_CONTACTS":
                code += f"devsim.add_1d_contact(mesh='{MESH_NAME}', name='{CONTACT_ANODE}', tag='{CONTACT_ANODE}', material='metal')\n"
                code += f"devsim.add_1d_contact(mesh='{MESH_NAME}', name='{CONTACT_CATHODE}', tag='{CONTACT_CATHODE}', material='metal')\n"
            elif item_name == "FINALIZE_MESH":
                code += f"devsim.finalize_mesh(mesh='{MESH_NAME}')\n"
            elif item_name == "CREATE_DEVICE":
                code += f"devsim.create_device(mesh='{MESH_NAME}', device='{DEVICE_NAME}')\n"
            elif item_name == "SET_MATERIAL_PARAMS":
                 # Can define constants within the generated script too
                 code += "q = 1.602176634e-19; eps_0 = 8.854187817e-14; k_boltzmann = 8.617333262e-5; T = 300.0\n"
                 code += "eps_si = 11.7; n_i_si = 1.0e10\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Permittivity', value=eps_si * eps_0)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='ElectronCharge', value=q)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='n_i', value=n_i_si)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='T', value=T)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='kT', value=k_boltzmann * T)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Vt', value=k_boltzmann * T)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='mu_n', value=400.0)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='mu_p', value=200.0)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='taun', value=1e-7)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', region='{REGION_NAME}', name='taup', value=1e-7)\n"
            elif item_name == "DEFINE_P_DOPING":
                 na_default = DEFAULT_NA # Use default value for now
                 self._design_params['na'] = na_default # Store the value used
                 xj = self._geometry_params['junction_pos_cm']
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Acceptors', equation='{na_default}*step({xj}-x)')\n"
                 logger.info(f" > Generated code to set Na={na_default:.1e}")
            elif item_name == "DEFINE_N_DOPING":
                 nd_default = DEFAULT_ND
                 self._design_params['nd'] = nd_default
                 xj = self._geometry_params['junction_pos_cm']
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Donors', equation='{nd_default}*step(x-{xj})')\n"
                 logger.info(f" > Generated code to set Nd={nd_default:.1e}")
            elif item_name == "DEFINE_NET_DOPING":
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='NetDoping', equation='Donors - Acceptors')\n"
            elif item_name == "DEFINE_VARIABLES":
                 code += f"devsim.node_solution(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Potential')\n"
                 code += f"devsim.node_solution(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Electrons')\n"
                 code += f"devsim.node_solution(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Holes')\n"
                 # Initial guess (important!) - include definition of models needed
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='IntrinsicPotential', equation='Vt * asinh(NetDoping / (2.0 * n_i))')\n"
                 code += f"devsim.set_node_value(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Potential', init_from='IntrinsicPotential')\n"
                 # Need to ensure Potential is initialized before these are used if called sequentially
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='IntrinsicElectrons', equation='n_i*exp(Potential/Vt)')\n"
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='IntrinsicHoles', equation='n_i*exp(-Potential/Vt)')\n"
                 code += f"devsim.set_node_value(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Electrons', init_from='IntrinsicElectrons')\n"
                 code += f"devsim.set_node_value(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Holes', init_from='IntrinsicHoles')\n"
            elif item_name == "SETUP_PHYSICS":
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='ElectricField', equation='(Potential@n0 - Potential@n1)*EdgeInverseLength')\n"
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='vdiff', equation='Potential@n0 - Potential@n1')\n"
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='vdiff_norm', equation='vdiff/Vt')\n"
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Bernoulli_n', equation='B(vdiff_norm)')\n"
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='Bernoulli_p', equation='B(-vdiff_norm)')\n"
                 code += f"eq_Jn = 'ElectronCharge*mu_n*EdgeCouple*Vt*(Electrons@n1*Bernoulli_n - Electrons@n0*Bernoulli_p)'\n"
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='ElectronCurrent', equation=eq_Jn)\n"
                 code += f"eq_Jp = '-ElectronCharge*mu_p*EdgeCouple*Vt*(Holes@n1*Bernoulli_p - Holes@n0*Bernoulli_n)'\n"
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='HoleCurrent', equation=eq_Jp)\n"
                 code += f"srh_expr = 'ElectronCharge * (Electrons*Holes - n_i^2) / (taup*(Electrons + n_i) + taun*(Holes + n_i))'\n"
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='SRH_Recombination', equation=srh_expr)\n"
            elif item_name == "SETUP_EQUATIONS":
                 code += f"devsim.edge_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='PotentialEdgeFlux', equation='Permittivity*ElectricField')\n"
                 code += f"devsim.node_model(device='{DEVICE_NAME}', region='{REGION_NAME}', name='PotentialNodeCharge', equation='-ElectronCharge*(Holes - Electrons + NetDoping)')\n"
                 code += f"devsim.equation(device='{DEVICE_NAME}', region='{REGION_NAME}', name='PotentialEquation', variable_name='Potential', node_model='PotentialNodeCharge', edge_model='PotentialEdgeFlux', variable_update='default')\n"
                 code += f"devsim.equation(device='{DEVICE_NAME}', region='{REGION_NAME}', name='ElectronContinuityEquation', variable_name='Electrons', node_model='SRH_Recombination', edge_model='ElectronCurrent', variable_update='positive')\n"
                 code += f"devsim.equation(device='{DEVICE_NAME}', region='{REGION_NAME}', name='HoleContinuityEquation', variable_name='Holes', node_model='SRH_Recombination', edge_model='-HoleCurrent', variable_update='positive')\n"
            elif item_name == "SETUP_CONTACT_BC":
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', name='Vanode_bias', value=0.0)\n"
                 code += f"devsim.set_parameter(device='{DEVICE_NAME}', name='Vcathode_bias', value=0.0)\n"
                 code += f"for contact in ['{CONTACT_ANODE}', '{CONTACT_CATHODE}']:\n"
                 code += f"    bias_name = f'{{contact}}_bias'\n"
                 code += f"    devsim.contact_node_model(device='{DEVICE_NAME}', contact=contact, name=f'{{contact}}_potential_bc', equation=f'Potential - {{bias_name}}')\n"
                 code += f"    devsim.contact_equation(device='{DEVICE_NAME}', contact=contact, name='PotentialEquation', node_model=f'{{contact}}_potential_bc')\n"
                 # Simplified carrier BCs
                 code += f"    devsim.contact_node_model(device='{DEVICE_NAME}', contact=contact, name='contact_electrons', equation='Electrons - ifelse(NetDoping > 0, NetDoping, n_i^2/abs(NetDoping+1e-30))')\n"
                 code += f"    devsim.contact_node_model(device='{DEVICE_NAME}', contact=contact, name='contact_holes', equation='Holes - ifelse(NetDoping < 0, abs(NetDoping), n_i^2/(NetDoping+1e-30))')\n"
                 code += f"    devsim.contact_equation(device='{DEVICE_NAME}', contact=contact, name='ElectronContinuityEquation', node_model='contact_electrons')\n"
                 code += f"    devsim.contact_equation(device='{DEVICE_NAME}', contact=contact, name='HoleContinuityEquation', node_model='contact_holes')\n"

            else:
                 logger.warning(f"No code generation logic for step '{item_name}'")
                 code += f"# No code generated for step {item_name}\n"

            logger.info(f" > Generated {len(code.splitlines())-1} lines for '{item_name}'")
            return code

        except Exception as e:
            # Catch potential errors during string formatting or logic
            logger.error(f"Error generating code string for step '{item_name}': {e}", exc_info=True)
            return None # Indicate failure


    def _calculate_final_reward(self, metrics):
        # ... (same as before) ...
        if not metrics: return -100.0
        if_val = metrics.get('if', 0); ir_val = abs(metrics.get('ir', 1e-3))
        if_target = self.target_metrics['if_min']; ir_target = self.target_metrics['ir_max']
        ratio = (if_val / ir_val) if ir_val > 1e-18 else 1e18
        reward = np.log10(max(1.0, ratio))
        if if_val < if_target: reward -= 10 * (1 - if_val/(if_target + 1e-12))
        if ir_val > ir_target: reward -= 10 * max(0, ir_val/ir_target - 1)
        reward -= 0.05 * self._current_step
        logger.info(f"  Final Metrics: If={if_val:.2e}, Ir={ir_val:.2e}, Ratio={ratio:.2e}")
        logger.info(f"  Reward calculated: {reward:.3f}")
        return reward


    def _execute_generated_script_and_get_reward(self):
        """
        Executes the accumulated script string and calculates final reward.
        WARNING: Uses exec(), which can be a security risk if script content
                 is not fully controlled.
        """
        if not self.execute_on_finalize:
             logger.warning("Execution on finalize is disabled. Returning 0 reward.")
             # Return a neutral reward or based only on completion status?
             return 0.0 # No performance feedback

        if not devsim:
             logger.error("Cannot execute script: DEVSIM module not imported.")
             return -200.0 # Indicate critical failure

        full_script = "# Automatically generated DEVSIM script\n"
        full_script += "import devsim\n"
        full_script += "import numpy as np\n"
        full_script += "\n".join(self._generated_script_lines)
        # Add initial solve command to the end of setup part
        full_script += "\n# --- Initial DC Solve ---\n"
        full_script += "try:\n"
        full_script += f"    devsim.solve(type='dc', absolute_error=1e-10, relative_error=1e-10, maximum_iterations=30)\n"
        full_script += f"    print('Initial DC solve successful.')\n"
        full_script += f"    initial_solve_success = True\n"
        full_script += "except Exception as e:\n"
        full_script += f"    print(f'FATAL: Initial DC solve failed: {{e}}')\n"
        full_script += f"    initial_solve_success = False\n"


        logger.info(f"Attempting to execute {len(self._generated_script_lines)} lines of generated DEVSIM code...")
        print("--- Generated Script ---")
        print(full_script)
        print("--- End Script ---")

        try:
            # VERY IMPORTANT: Ensure the environment where exec runs is safe.
            # We execute it here, modifying the current devsim state.
            # We add a dictionary to capture the success flag from the script.
            script_globals = {'devsim': devsim, 'np': np}
            script_locals = {}
            exec(full_script, script_globals, script_locals)

            # Check if the initial solve within the script succeeded
            if not script_locals.get('initial_solve_success', False):
                 logger.error("Generated script executed, but initial solve failed.")
                 return -110.0 # Penalty for valid script but non-converging setup

            logger.info("Generated script executed successfully (including initial solve).")
            # Now run the final simulation using the state modified by exec()
            metrics, sim_success = run_final_diode_simulation(self._design_params)
            if sim_success:
                reward = self._calculate_final_reward(metrics)
            else:
                reward = -100.0 # Final sim logic failed (placeholder)
            return reward

        except Exception as e:
            logger.error(f"FAILED to execute generated script: {e}", exc_info=True)
            # print("--- Failing Script ---")
            # print(full_script)
            # print("--- End Failing Script ---")
            return -200.0 # Major penalty for script execution error


    def step(self, action):
        self._current_step += 1
        action_item_index = int(action)
        if not (0 <= action_item_index < len(DIODE_CHECKLIST)):
            logger.error(f"Invalid action index received: {action_item_index}")
            observation = self._get_observation()
            return observation, -1.0, True, False, {"error": "Invalid action index"}
        action_item_name = DIODE_CHECKLIST[action_item_index]
        terminated = False; truncated = False; reward = -0.01 # Step cost
        info = {"action_name": action_item_name}
        logger.info(f"--- Step {self._current_step}: Agent selected action '{action_item_name}' ({action_item_index}) ---")

        if action_item_name in self._completed_steps:
            logger.warning(f"Action '{action_item_name}' already completed. Applying penalty.")
            reward = -1.0
            info["error"] = "Step already completed"
        elif action_item_name == "FINALIZE_SETUP_RUN_TEST":
            required_prereqs = {"DEFINE_P_DOPING", "DEFINE_N_DOPING", "SETUP_CONTACT_BC", "SETUP_EQUATIONS"}
            if not required_prereqs.issubset(self._completed_steps):
                 logger.error(f"Cannot finalize: Prerequisites {required_prereqs} not met. Completed: {self._completed_steps}")
                 reward = -20.0; terminated = True
                 info["error"] = "Prerequisites for finalize not met"
            else:
                 logger.info("Action 'FINALIZE_SETUP_RUN_TEST' selected.")
                 # Mark step as completed before execution attempt
                 self._completed_steps.add(action_item_name)
                 # Execute script and get final reward (if enabled)
                 reward = self._execute_generated_script_and_get_reward()
                 terminated = True # Always terminate on finalize attempt
                 # Info will be updated by the execution function or reflect failure
                 info["final_script"] = "\n".join(self._generated_script_lines) # Optionally store final script
        else:
            # Generate the code string for this step
            code_string = self._generate_devsim_step_code(action_item_index)
            if code_string is not None:
                self._generated_script_lines.append(code_string) # Add code lines
                self._completed_steps.add(action_item_name)
                # Use the intermediate reward (step cost)
                info["code_generated"] = True
            else:
                # Failed to generate code (e.g., dependency error)
                reward = -10.0; terminated = True
                info["error"] = "Step generation failed (dependency error)"
                info["code_generated"] = False

        observation = self._get_observation()
        # Update info dict *after* getting observation
        info["completed_steps"] = sorted(list(self._completed_steps))
        info["params"] = self._design_params

        if not terminated and self._current_step >= self._max_steps:
            truncated = True
            if "FINALIZE_SETUP_RUN_TEST" not in self._completed_steps:
                reward = -20.0 # Penalize if max steps reached without finalizing
            logger.info("Max steps reached, truncating episode.")

        # Final check on reward value
        if not isinstance(reward, (int, float)):
             logger.error(f"Invalid reward type generated: {reward} ({type(reward)})")
             reward = -500 # Assign a default error penalty

        return observation, float(reward), terminated, truncated, info # Ensure reward is float


    def close(self):
        pass

# ======================================================
# --- Test Cases (Should still work conceptually) ---
# ======================================================
if __name__ == '__main__':
    # Set logging level to DEBUG to see more detail during tests
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger.setLevel(logging.DEBUG)

    print("\n--- Testing DiodeDesignEnv (Script Generation Mode) ---")
    # Test with execute_on_finalize=True to test reward mechanism
    env = DiodeDesignEnv(max_steps=16, execute_on_finalize=True)

    # Test 1: Reset and Initial Observation
    print("\n--- Test 1: Reset ---"); obs, info = env.reset()
    print(f"Initial Obs: {obs}\nInitial Info: {info}")
    assert obs.shape == (OBS_DIM,), "Obs shape mismatch"
    assert not info['completed_steps'], "Should start empty"
    print("Reset Test Passed.")

    # Test 2: Single Valid Step
    print("\n--- Test 2: Single Valid Step (INIT_MESH) ---")
    action = DIODE_CHECKLIST.index("INIT_MESH")
    obs, reward, term, trunc, info = env.step(action)
    print(f"Obs: {obs}\nReward: {reward:.3f}, Term: {term}, Trunc: {trunc}\nInfo: {info}")
    assert not term and not trunc, "Should not end"
    assert "INIT_MESH" in info['completed_steps'], "Step not completed"
    assert obs[action] == 1.0, "Obs flag not set"
    assert len(env._generated_script_lines) > 0, "Script lines not generated"
    print(f" Generated lines:\n{env._generated_script_lines[-1]}") # Print last generated code
    print("Single Step Test Passed.")

    # Test 3: Invalid Step (Dependency Fail)
    print("\n--- Test 3: Invalid Step (Dependency Fail - DEFINE_P_DOPING early) ---")
    action = DIODE_CHECKLIST.index("DEFINE_P_DOPING")
    obs, reward, term, trunc, info = env.step(action) # Depends on DEFINE_REGIONS
    print(f"Obs: {obs}\nReward: {reward:.3f}, Term: {term}, Trunc: {trunc}\nInfo: {info}")
    assert term, "Should terminate on dependency failure"
    assert reward <= -10.0, "Should have penalty"
    assert not env._generated_script_lines or "DEFINE_P_DOPING" not in env._generated_script_lines[-1], "Code should not be generated on failure"
    print("Invalid Step (Dependency) Test Passed.")

    # Test 4: Sequence leading to Finalize (using generated code)
    print("\n--- Test 4: Sequence leading to Finalize ---")
    obs, info = env.reset()
    valid_sequence = [
        "INIT_MESH", "DEFINE_GEOMETRY", "DEFINE_REGIONS", "DEFINE_CONTACTS",
        "FINALIZE_MESH", "CREATE_DEVICE", "SET_MATERIAL_PARAMS",
        "DEFINE_P_DOPING", "DEFINE_N_DOPING", "DEFINE_NET_DOPING",
        "DEFINE_VARIABLES", "SETUP_PHYSICS", "SETUP_EQUATIONS",
        "SETUP_CONTACT_BC", "FINALIZE_SETUP_RUN_TEST"
    ]
    final_reward = 0; final_info = {}; step_count = 0; sequence_success = True
    for item_name in valid_sequence:
        step_count += 1; action = DIODE_CHECKLIST.index(item_name)
        print(f"\nTaking action: {item_name} ({action})")
        obs, reward, term, trunc, info = env.step(action)
        print(f" Obs[{item_name} flag]: {obs[action]:.1f}, Obs[Na]: {obs[PARAM_NA_IDX]:.2f}, Obs[Nd]: {obs[PARAM_ND_IDX]:.2f}, Obs[Step]: {obs[STEP_COUNT_IDX]:.2f}")
        print(f" Reward: {reward:.3f}, Term: {term}, Trunc: {trunc}")
        # print(f" Completed: {info.get('completed_steps', [])}")
        final_reward = reward; final_info = info
        if info.get("error", None): print(f" ERROR: {info['error']}"); sequence_success = False
        if term or trunc: print(f"Episode ended at step {step_count}."); break
    assert sequence_success, "Valid sequence resulted in an error"
    assert term, "Episode should terminate after finalize action"
    assert not trunc, "Episode should not truncate if finalized correctly"
    assert "FINALIZE_SETUP_RUN_TEST" in info['completed_steps'], "Finalize step should be marked completed"
    # Check if reward seems plausible (depends on placeholder run_final_diode_simulation)
    print(f"Final Info after sequence: {final_info}")
    print(f"Final Reward: {final_reward}")
    print("Valid Sequence Test Passed.")

    # Test 5 & 6 remain the same conceptually
    print("\n--- Running Test 5: Max Steps Truncation ---")
    obs, info = env.reset()
    final_reward = 0; partial_sequence = ["INIT_MESH", "DEFINE_GEOMETRY", "DEFINE_REGIONS", "DEFINE_CONTACTS"]
    for i in range(env._max_steps):
        action_name = partial_sequence[i % len(partial_sequence)]
        action = DIODE_CHECKLIST.index(action_name)
        print(f" Step {i+1}/{env._max_steps}, Action: {action_name}")
        obs, reward, terminated, truncated, info = env.step(action)
        final_reward = reward;
        if terminated or truncated: break
    print(f"Final state after {env._current_step} steps: Term: {terminated}, Trunc: {truncated}, Reward: {final_reward:.3f}")
    assert truncated, "Episode should truncate"
    assert not terminated, "Episode should not terminate"
    print("Max Steps Truncation Test Passed.")

    print("\n--- Running Test 6: Repeating an action ---")
    obs, info = env.reset()
    action = DIODE_CHECKLIST.index("INIT_MESH")
    print("Taking action INIT_MESH..."); obs, reward1, term1, trunc1, info1 = env.step(action)
    print(f"Reward1: {reward1:.3f}")
    print("Taking action INIT_MESH again..."); obs, reward2, term2, trunc2, info2 = env.step(action) # Repeat
    print(f"Reward2: {reward2:.3f}")
    assert reward1 != reward2 and reward2 < -0.5, "Repeating action should have penalty"
    print("Repeat Action Test Passed.")

    print("\n--- All Tests Completed ---")