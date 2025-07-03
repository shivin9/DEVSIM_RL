# file: devsim_environment.py
# NEW, FULLY REVISED VERSION - 2025-06-13

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
try:
    import devsim
except ImportError:
    print("WARNING: DEVSIM Python module not found. Final execution inside env will fail.")
    devsim = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Placeholder for final I-V curve simulation ---
def run_final_diode_iv_sweep():
    # In a real scenario, this function would sweep the anode voltage
    # and return the calculated If and Ir.
    # For now, it returns fixed placeholder values for a "good" diode.
    logger.info("Placeholder: Running final I-V sweep.")
    return {'if': 1.5e-4, 'ir': 1e-12}, True

# --- RESTRUCTURED CHECKLIST FOR TWO-STAGE SOLVE ---
DIODE_CHECKLIST = [
    "INIT_MESH",
    "DEFINE_GEOMETRY",
    "DEFINE_REGIONS",
    "DEFINE_CONTACTS",
    "FINALIZE_MESH",
    "CREATE_DEVICE",
    "SET_MATERIAL_PARAMS",
    "DEFINE_DOPING",
    "DEFINE_VARIABLES",
    "SETUP_POTENTIAL_EQUATION",
    "SETUP_CONTACT_BC_POTENTIAL",
    "SOLVE_POTENTIAL_ONLY",
    "SETUP_CONTINUITY_EQUATIONS",
    "SETUP_CONTACT_BC_CARRIERS",
    "FINALIZE_AND_SOLVE_DD"
]
NUM_CHECKLIST_ITEMS = len(DIODE_CHECKLIST)

# --- Global constants ---
PARAM_NA_IDX = NUM_CHECKLIST_ITEMS + 0
PARAM_ND_IDX = NUM_CHECKLIST_ITEMS + 1
STEP_COUNT_IDX = NUM_CHECKLIST_ITEMS + 2
OBS_DIM = NUM_CHECKLIST_ITEMS + 3
DEFAULT_NA = 1e17
DEFAULT_ND = 1e17
DEVICE_LENGTH_UM = 1.0
JUNCTION_POS_UM = 0.5
UM_TO_CM = 1e-4
MESH_NAME = "diode_mesh"
DEVICE_NAME = "MyDiode"
REGION_NAME = "Silicon"
CONTACT_ANODE = "Anode"
CONTACT_CATHODE = "Cathode"

class DiodeDesignEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, max_steps=25, target_metrics=None, execute_on_finalize=True):
        super().__init__()
        self._max_steps = max_steps
        self.target_metrics = target_metrics or {"if_min": 1e-4, "ir_max": 1e-9}
        self.execute_on_finalize = execute_on_finalize
        self._completed_steps = set()
        self._design_params = {}
        self._generated_script_lines = []
        self._current_step = 0
        self._geometry_params = {'device_length_cm': DEVICE_LENGTH_UM * UM_TO_CM, 'junction_pos_cm': JUNCTION_POS_UM * UM_TO_CM}
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_CHECKLIST_ITEMS)
        logger.info("DiodeDesignEnv initialized.")

    def _get_observation(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for i, item in enumerate(DIODE_CHECKLIST):
            if item in self._completed_steps:
                obs[i] = 1.0
        obs[PARAM_NA_IDX] = (np.log10(self._design_params.get('na', DEFAULT_NA)) - 15) / 4
        obs[PARAM_ND_IDX] = (np.log10(self._design_params.get('nd', DEFAULT_ND)) - 15) / 4
        obs[STEP_COUNT_IDX] = float(self._current_step) / float(self._max_steps)
        return np.clip(obs, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.execute_on_finalize and devsim:
            try:
                devsim.reset_devsim()
                logger.info("DEVSIM session reset successfully.")
            except Exception as e:
                logger.error(f"devsim.reset_devsim() failed: {e}")
        self._completed_steps = set()
        self._design_params = {}
        self._generated_script_lines = []
        self._current_step = 0
        return self._get_observation(), {"completed_steps": set(), "params": {}}

    def _calculate_final_reward(self, metrics):
        if not metrics:
            return -100.0
        SUCCESS_BONUS = 50.0
        if_val = metrics.get('if', 0)
        ir_val = abs(metrics.get('ir', 1e-3))
        ratio = (if_val / ir_val) if ir_val > 1e-18 else 1e18
        performance_reward = np.log10(max(1.0, ratio))
        total_reward = SUCCESS_BONUS + performance_reward
        logger.info(f"Final reward calculated: {total_reward:.2f}")
        return total_reward

    def _execute_full_script(self):
        if not self.execute_on_finalize or not devsim:
            return 0.0
        
        full_script = "# Automatically generated DEVSIM script\n"
        full_script += "import devsim\nimport numpy as np\n"
        full_script += "\n".join(self._generated_script_lines)
        
        logger.info("Attempting to execute full generated script...")
        try:
            exec(full_script, {'devsim': devsim, 'np': np})
            logger.info("Script executed successfully.")
            metrics, sim_success = run_final_diode_iv_sweep()
            return self._calculate_final_reward(metrics) if sim_success else -100.0
        except Exception as e:
            logger.error(f"FAILED to execute generated script: {e}", exc_info=True)
            return -200.0

    def step(self, action):
        self._current_step += 1
        action_item_name = DIODE_CHECKLIST[action]
        info = {"action_name": action_item_name}
        terminated = False
        truncated = False
        
        deps = self._get_dependencies(action_item_name)
        if action_item_name in self._completed_steps:
            reward = -2.0
            terminated = True
            info["error"] = "Step already completed"
        elif not deps.issubset(self._completed_steps):
            reward = -10.0
            terminated = True
            info["error"] = f"Dependency failed. Requires: {deps}"
        else:
            code_to_add = self._generate_devsim_step_code(action)
            self._generated_script_lines.append(code_to_add)
            self._completed_steps.add(action_item_name)
            
            if action_item_name == "FINALIZE_AND_SOLVE_DD":
                reward = self._execute_full_script()
                terminated = True
                info["final_script"] = "\n".join(self._generated_script_lines)
            elif action_item_name == 'INIT_MESH':
                reward = 15.0
            else:
                reward = 1.0

        if not terminated and self._current_step >= self._max_steps:
            truncated = True
            reward = -20.0
        
        observation = self._get_observation()
        info["completed_steps"] = self._completed_steps
        return observation, float(reward), terminated, truncated, info

    def _get_dependencies(self, item_name):
        # Centralized dependency logic
        deps = {
            "DEFINE_GEOMETRY": {"INIT_MESH"},
            "DEFINE_REGIONS": {"DEFINE_GEOMETRY"},
            "DEFINE_CONTACTS": {"DEFINE_REGIONS"},
            "FINALIZE_MESH": {"DEFINE_CONTACTS"},
            "CREATE_DEVICE": {"FINALIZE_MESH"},
            "SET_MATERIAL_PARAMS": {"CREATE_DEVICE"},
            "DEFINE_DOPING": {"CREATE_DEVICE"},
            "DEFINE_VARIABLES": {"DEFINE_DOPING"},
            "SETUP_POTENTIAL_EQUATION": {"DEFINE_VARIABLES"},
            "SETUP_CONTACT_BC_POTENTIAL": {"SETUP_POTENTIAL_EQUATION"},
            "SOLVE_POTENTIAL_ONLY": {"SETUP_CONTACT_BC_POTENTIAL"},
            "SETUP_CONTINUITY_EQUATIONS": {"SOLVE_POTENTIAL_ONLY"},
            "SETUP_CONTACT_BC_CARRIERS": {"SETUP_CONTINUITY_EQUATIONS"},
            "FINALIZE_AND_SOLVE_DD": {"SETUP_CONTACT_BC_CARRIERS"}
        }
        return deps.get(item_name, set())

    def _generate_devsim_step_code(self, checklist_item_index):
        # This is the full, corrected code generation logic from the last step
        item_name = DIODE_CHECKLIST[checklist_item_index]
        code = f"\n# --- Code for Step: {item_name} ---\n"
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
            code += "q=1.6e-19;eps_0=8.85e-14;k_boltzmann=8.617e-5;T=300;eps_si=11.7;n_i_si=1.0e10\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='Permittivity',value=eps_si*eps_0)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='ElectronCharge',value=q)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='n_i',value=n_i_si)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='T',value=T)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='Vt',value=k_boltzmann*T)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='mu_n',value=400.0)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='mu_p',value=200.0)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='taun',value=1e-7)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}',name='taup',value=1e-7)\n"
        elif item_name == "DEFINE_DOPING":
            self._design_params.update({'na': DEFAULT_NA, 'nd': DEFAULT_ND})
            xj = self._geometry_params['junction_pos_cm']
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Acceptors',equation='{DEFAULT_NA}*step({xj}-x)')\n"
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Donors',equation='{DEFAULT_ND}*step(x-{xj})')\n"
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='NetDoping',equation='Donors-Acceptors')\n"
        elif item_name == "DEFINE_VARIABLES":
            code += f"devsim.node_solution(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Potential')\n"
            code += f"devsim.node_solution(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Electrons')\n"
            code += f"devsim.node_solution(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Holes')\n"
            code += f"devsim.set_node_value(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Potential',value=0.0)\n"
            code += f"devsim.set_node_value(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Electrons',value=0.0)\n"
            code += f"devsim.set_node_value(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Holes',value=0.0)\n"
        elif item_name == "SETUP_POTENTIAL_EQUATION":
            code += f"devsim.edge_from_node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',node_model='Potential')\n"
            code += f"devsim.edge_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='ElectricField',equation='(Potential@n0-Potential@n1)*EdgeInverseLength')\n"
            code += f"devsim.edge_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='PotentialEdgeFlux',equation='Permittivity*ElectricField')\n"
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='PotentialNodeCharge',equation='-ElectronCharge*NetDoping')\n"
            code += f"devsim.equation(device='{DEVICE_NAME}',region='{REGION_NAME}',name='PotentialEquation',variable_name='Potential',node_model='PotentialNodeCharge',edge_model='PotentialEdgeFlux')\n"
        elif item_name == "SETUP_CONTACT_BC_POTENTIAL":
            code += f"devsim.set_parameter(device='{DEVICE_NAME}', name='VAnode_bias', value=0.0)\n"
            code += f"devsim.set_parameter(device='{DEVICE_NAME}', name='VCathode_bias', value=0.0)\n"
            code += f"for c in ['{CONTACT_ANODE}', '{CONTACT_CATHODE}']:\n"
            code +=  "    bias_name = f'V{c}_bias'\n"
            code += f"    devsim.contact_node_model(device='{DEVICE_NAME}', contact=c, name=f'{{c}}_potential_bc', equation=f'Potential - {{bias_name}}')\n"
            code += f"    devsim.contact_equation(device='{DEVICE_NAME}', contact=c, name='PotentialEquation', node_model=f'{{c}}_potential_bc')\n"
        elif item_name == "SOLVE_POTENTIAL_ONLY":
            code += "devsim.solve(type='dc',absolute_error=1e10,relative_error=1,maximum_iterations=30)\n"
        elif item_name == "SETUP_CONTINUITY_EQUATIONS":
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='IntrinsicElectrons',equation='n_i*exp(Potential/Vt)')\n"
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='IntrinsicHoles',equation='n_i*exp(-Potential/Vt)')\n"
            code += f"devsim.set_node_values(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Electrons',init_from='IntrinsicElectrons')\n"
            code += f"devsim.set_node_values(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Holes',init_from='IntrinsicHoles')\n"
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='PotentialNodeCharge',equation='-ElectronCharge*(Holes-Electrons+NetDoping)')\n"
            code += f"devsim.edge_from_node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',node_model='Electrons')\n"
            code += f"devsim.edge_from_node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',node_model='Holes')\n"
            code += f"devsim.edge_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='vdiff',equation='(Potential@n0-Potential@n1)/Vt')\n"
            code += f"devsim.edge_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Bernoulli_n',equation='B(vdiff)')\n"
            code += f"devsim.edge_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='Bernoulli_p',equation='B(-vdiff)')\n"
            code += "eq_Jn='ElectronCharge*mu_n*Vt*EdgeCouple*(Electrons@n1*Bernoulli_n-Electrons@n0*Bernoulli_p)'\n"
            code += f"devsim.edge_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='ElectronCurrent',equation=eq_Jn)\n"
            code += "eq_Jp='-ElectronCharge*mu_p*Vt*EdgeCouple*(Holes@n1*Bernoulli_p-Holes@n0*Bernoulli_n)'\n"
            code += f"devsim.edge_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='HoleCurrent',equation=eq_Jp)\n"
            code += "srh='ElectronCharge*(Electrons*Holes-n_i^2)/(taup*(Electrons+n_i)+taun*(Holes+n_i))'\n"
            code += f"devsim.node_model(device='{DEVICE_NAME}',region='{REGION_NAME}',name='SRH',equation=srh)\n"
            code += f"devsim.equation(device='{DEVICE_NAME}',region='{REGION_NAME}',name='e_continuity',variable_name='Electrons',node_model='SRH',edge_model='ElectronCurrent',variable_update='positive')\n"
            code += f"devsim.equation(device='{DEVICE_NAME}',region='{REGION_NAME}',name='h_continuity',variable_name='Holes',node_model='SRH',edge_model='-HoleCurrent',variable_update='positive')\n"
        elif item_name == "SETUP_CONTACT_BC_CARRIERS":
            code += f"for c in ['{CONTACT_ANODE}','{CONTACT_CATHODE}']:\n"
            code += f"    devsim.contact_node_model(device='{DEVICE_NAME}',contact=c,name=f'{{c}}_e_bc',equation='Electrons-ifelse(NetDoping>0,NetDoping,n_i^2/abs(NetDoping+1e-30))')\n"
            code += f"    devsim.contact_node_model(device='{DEVICE_NAME}',contact=c,name=f'{{c}}_h_bc',equation='Holes-ifelse(NetDoping<0,abs(NetDoping),n_i^2/(NetDoping+1e-30))')\n"
            code += f"    devsim.contact_equation(device='{DEVICE_NAME}',contact=c,name='e_continuity',node_model=f'{{c}}_e_bc')\n"
            code += f"    devsim.contact_equation(device='{DEVICE_NAME}',contact=c,name='h_continuity',node_model=f'{{c}}_h_bc')\n"
        elif item_name == "FINALIZE_AND_SOLVE_DD":
            code += "devsim.solve(type='dc',absolute_error=1e12,relative_error=1e-10,maximum_iterations=30)\n"
        return code