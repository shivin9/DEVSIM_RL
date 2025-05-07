# file: rl_environment.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
# NOTE: Assuming devsim is importable, though not strictly needed for placeholder tests
# import devsim

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Placeholder for final simulation ---
# Assume this function exists (needs implementation later)
# It takes the final state (e.g., doping params), runs the DEVSIM I-V sweep,
# and returns performance metrics like {'if': 1e-3, 'ir': 1e-10, 'vf': 0.7}
def run_final_diode_simulation(design_params):
    logger.info(f"Placeholder: Running final diode simulation with params: {design_params}")
    # Placeholder results
    sim_success = np.random.rand() > 0.1 # 90% success rate placeholder
    if sim_success:
        # Simulate some dependency on doping (highly simplified)
        na = design_params.get('na', 1e17)
        nd = design_params.get('nd', 1e17)
        vf = 0.6 + 0.1 * np.log10(1e17 / na) + 0.1 * np.log10(1e17 / nd)
        if_val = 1e-4 * (na/1e17)**0.5 * (nd/1e17)**0.5 * np.exp(vf / (1.5 * 0.0259)) # Simplified If
        ir_val = 1e-11 * (1e17/na)**0.3 * (1e17/nd)**0.3 # Simplified Ir
        logger.info(f"Placeholder Sim Success: If={if_val:.2e}, Ir={ir_val:.2e}, Vf={vf:.3f}")
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
    "SETUP_CONTACT_BC", "FINALIZE_SETUP_RUN_TEST"
]
NUM_CHECKLIST_ITEMS = len(DIODE_CHECKLIST)

# Define indices for key parameters we might track in the state
PARAM_NA_IDX = NUM_CHECKLIST_ITEMS + 0
PARAM_ND_IDX = NUM_CHECKLIST_ITEMS + 1
STEP_COUNT_IDX = NUM_CHECKLIST_ITEMS + 2
# Define total observation dimension
OBS_DIM = NUM_CHECKLIST_ITEMS + 3 # Checklist status + Na + Nd + Step count

class DiodeDesignEnv(gym.Env):
    """
    RL Environment for sequentially selecting DEVSIM setup steps (checklist)
    to design a simple 1D P-N diode.
    """
    metadata = {'render_modes': []}

    def __init__(self, max_steps=NUM_CHECKLIST_ITEMS + 3, target_metrics=None): # Allow a few extra steps
        super().__init__()

        self._max_steps = max_steps
        self.target_metrics = target_metrics or {"if_min": 1e-4, "ir_max": 1e-9}

        # --- State Representation ---
        self._completed_steps = set()
        self._design_params = {} # Stores key values like Na, Nd

        # Observation space:
        self.na_bounds_log10 = (15.0, 19.0) # e.g., 1e15 to 1e19 cm^-3
        self.nd_bounds_log10 = (15.0, 19.0) # e.g., 1e15 to 1e19 cm^-3

        low_bounds = np.array([0.0] * NUM_CHECKLIST_ITEMS + [0.0] * 2 + [0.0], dtype=np.float32)
        high_bounds = np.array([1.0] * NUM_CHECKLIST_ITEMS + [1.0] * 2 + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, shape=(OBS_DIM,), dtype=np.float32)

        # --- Action Space ---
        self.action_space = spaces.Discrete(NUM_CHECKLIST_ITEMS)

        self._current_step = 0
        logger.info("DiodeDesignEnv initialized.")

    def _normalize_param(self, value, bounds_log10):
        """Normalize parameter using log10 scale."""
        if value is None or value <= 0:
            return 0.0 # Represent unset or invalid with 0
        val_log10 = np.log10(value)
        norm_val = (val_log10 - bounds_log10[0]) / (bounds_log10[1] - bounds_log10[0])
        return np.clip(norm_val, 0.0, 1.0)

    def _get_observation(self):
        """
        Generates the observation vector based on completed checklist items
        and key design parameters.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # 1. Checklist completion status (binary flags)
        for i, item in enumerate(DIODE_CHECKLIST):
            if item in self._completed_steps:
                obs[i] = 1.0

        # 2. Key parameters (normalized)
        na_val = self._design_params.get('na', None)
        nd_val = self._design_params.get('nd', None)
        obs[PARAM_NA_IDX] = self._normalize_param(na_val, self.na_bounds_log10)
        obs[PARAM_ND_IDX] = self._normalize_param(nd_val, self.nd_bounds_log10)

        # 3. Normalized Step Count
        obs[STEP_COUNT_IDX] = float(self._current_step) / float(self._max_steps)

        # logger.debug(f"Observation generated: Checklist flags (first {NUM_CHECKLIST_ITEMS}), Na_norm, Nd_norm, Step_norm")
        # logger.debug(f"Observation values: {obs}")
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # In a real scenario, you MUST reinitialize DEVSIM fully here
        # devsim.reinit()
        logger.warning("DEVSIM reinit() skipped in placeholder reset.")
        self._completed_steps = set()
        self._design_params = {} # Clear stored parameters
        self._current_step = 0
        logger.info("DiodeDesignEnv reset.")
        observation = self._get_observation()
        info = {"completed_steps": sorted(list(self._completed_steps)), "params": self._design_params}
        return observation, info

    def _execute_devsim_step(self, checklist_item_index):
        """
        Placeholder function to execute the DEVSIM commands for a checklist item.
        Updates self._design_params if relevant (e.g., setting Na/Nd).
        Returns True on success, False on failure (e.g. dependency not met).
        """
        item_name = DIODE_CHECKLIST[checklist_item_index]
        #logger.info(f"Attempting to execute DEVSIM step '{item_name}'")

        # --- Dependency Checks (CRUCIAL!) ---
        # These ensure steps are performed in a somewhat logical order
        required_deps = {
            "DEFINE_GEOMETRY": {"INIT_MESH"},
            "DEFINE_REGIONS": {"DEFINE_GEOMETRY"},
            "DEFINE_CONTACTS": {"DEFINE_REGIONS"},
            "FINALIZE_MESH": {"DEFINE_CONTACTS", "DEFINE_REGIONS"}, # Needs regions & contacts defined on mesh
            "CREATE_DEVICE": {"FINALIZE_MESH"},
            "SET_MATERIAL_PARAMS": {"CREATE_DEVICE"},
            "DEFINE_P_DOPING": {"CREATE_DEVICE", "DEFINE_REGIONS"}, # Need device/regions
            "DEFINE_N_DOPING": {"CREATE_DEVICE", "DEFINE_REGIONS"},
            "DEFINE_NET_DOPING": {"DEFINE_P_DOPING", "DEFINE_N_DOPING"},
            "DEFINE_VARIABLES": {"CREATE_DEVICE"},
            "SETUP_PHYSICS": {"CREATE_DEVICE", "SET_MATERIAL_PARAMS"}, # Need params for physics models
            "SETUP_EQUATIONS": {"DEFINE_VARIABLES", "SETUP_PHYSICS", "DEFINE_NET_DOPING"}, # Eqns use vars, physics, doping
            "SETUP_CONTACT_BC": {"CREATE_DEVICE", "DEFINE_CONTACTS", "SETUP_EQUATIONS"} # Apply BCs to equations on contacts
        }
        # FINALIZE_SETUP_RUN_TEST handled separately in step()

        deps = required_deps.get(item_name, set())
        if not deps.issubset(self._completed_steps):
            logger.error(f"Dependency failed for '{item_name}'. Requires: {deps}, Have: {self._completed_steps}")
            return False

        # --- Parameter Setting (Example using defaults) ---
        if item_name == "DEFINE_P_DOPING":
            na_default = 1e17
            self._design_params['na'] = na_default
            # devsim.node_model(..., equation=f"{na_default}...") # Actual call
            logger.info(f" > Executing '{item_name}', Setting Na={na_default:.1e} (default)")
        elif item_name == "DEFINE_N_DOPING":
            nd_default = 1e17
            self._design_params['nd'] = nd_default
            # devsim.node_model(..., equation=f"{nd_default}...") # Actual call
            logger.info(f" > Executing '{item_name}', Setting Nd={nd_default:.1e} (default)")
        else:
             logger.info(f" > Executing '{item_name}' (Placeholder - No params set)")

        # --- Actual DEVSIM call placeholder ---
        # Simulate success unless it's a known problematic step for testing
        sim_success = True
        # if item_name == "SETUP_EQUATIONS": # Example: Simulate failure for this step sometimes
        #     sim_success = np.random.rand() > 0.3

        if not sim_success:
             logger.error(f"DEVSIM command placeholder failed for step '{item_name}'.")

        return sim_success

    def _calculate_final_reward(self, metrics):
        """Calculate reward based on final diode metrics."""
        if not metrics:
            return -100.0 # Failed final simulation

        if_val = metrics.get('if', 0)
        ir_val = metrics.get('ir', 1e-3) # Assume high leakage if missing

        if_target = self.target_metrics['if_min']
        ir_target = self.target_metrics['ir_max']

        # Reward for forward current (higher is better, must meet minimum)
        reward_if = 0
        if if_val >= if_target:
            reward_if = 10 + 5 * np.log10(max(1, if_val / if_target)) # Bonus for exceeding target
        else:
            reward_if = -50 * (1 - if_val / (if_target + 1e-12)) # Penalty for missing target

        # Reward for reverse current (lower is better, must meet maximum)
        reward_ir = 0
        if abs(ir_val) <= ir_target:
             # Bonus for being much lower (use log scale relative to target)
             reward_ir = 10 + 5 * np.log10(max(1, ir_target / (abs(ir_val) + 1e-15)))
        else:
             # Penalty for exceeding target
             reward_ir = -50 * max(0, abs(ir_val) / ir_target - 1)

        total_reward = reward_if + reward_ir - 0.1 * self._current_step # Penalize steps
        logger.info(f"  Final Metrics: If={if_val:.2e}, Ir={ir_val:.2e}")
        logger.info(f"  Reward Breakdown: If={reward_if:.2f}, Ir={reward_ir:.2f}, Steps={-0.1*self._current_step:.2f}. Total={total_reward:.2f}")
        return total_reward


    def step(self, action):
        """Executes one step corresponding to a checklist item."""
        self._current_step += 1
        action_item_index = int(action) # Action is the index of the checklist item

        # Ensure action is within bounds
        if not (0 <= action_item_index < len(DIODE_CHECKLIST)):
             logger.error(f"Invalid action index received: {action_item_index}")
             # Return minimal penalty, terminate, let agent learn valid range
             observation = self._get_observation()
             return observation, -1.0, True, False, {"error": "Invalid action index"}

        action_item_name = DIODE_CHECKLIST[action_item_index]

        terminated = False
        truncated = False
        reward = -0.01 # Small cost per step
        info = {"action_name": action_item_name}

        logger.info(f"--- Step {self._current_step}: Agent selected action '{action_item_name}' ({action_item_index}) ---")

        # --- Check validity / dependencies ---
        if action_item_name in self._completed_steps:
            logger.warning(f"Action '{action_item_name}' already completed. Applying penalty.")
            reward = -1.0 # Penalize selecting an already completed step
            info["error"] = "Step already completed"
            # Do not terminate, allow agent to choose a different action next time
        elif action_item_name == "FINALIZE_SETUP_RUN_TEST":
            required_prereqs = {"DEFINE_P_DOPING", "DEFINE_N_DOPING", "SETUP_CONTACT_BC", "SETUP_EQUATIONS"} # Example minimum set
            if not required_prereqs.issubset(self._completed_steps):
                 logger.error(f"Cannot finalize: Prerequisites {required_prereqs} not met. Completed: {self._completed_steps}")
                 reward = -20.0 # Heavy penalty for trying to finalize too early
                 terminated = True # End episode if trying to finalize incorrectly
                 info["error"] = "Prerequisites for finalize not met"
            else:
                 logger.info("Finalizing design and running test simulation...")
                 metrics, sim_success = run_final_diode_simulation(self._design_params)
                 info["final_metrics"] = metrics if sim_success else {}
                 info["final_sim_success"] = sim_success
                 self._completed_steps.add(action_item_name)

                 if sim_success:
                     reward = self._calculate_final_reward(metrics)
                 else:
                     reward = -100.0 # Final sim failed
                 terminated = True # Episode ends after finalize action
        else:
            # Execute the DEVSIM commands for this step (placeholder checks dependencies)
            success = self._execute_devsim_step(action_item_index)
            info["step_execution_success"] = success
            if success:
                self._completed_steps.add(action_item_name)
                # Use the intermediate reward (step cost)
            else:
                # Failed to execute step (e.g., dependency error, DEVSIM error)
                reward = -10.0 # Penalty for invalid step execution
                terminated = True # End episode on failure
                info["error"] = "Step execution failed (dependency or placeholder error)"

        # Get observation *after* potentially updating state
        observation = self._get_observation()
        # Update info dict *after* getting observation
        info["completed_steps"] = sorted(list(self._completed_steps))
        info["params"] = self._design_params

        # Check truncation only if not already terminated
        if not terminated and self._current_step >= self._max_steps:
            truncated = True
            # Decide if penalty applies on truncation, maybe only if not finalized?
            if "FINALIZE_SETUP_RUN_TEST" not in self._completed_steps:
                reward = -20.0 # Penalize for running out of steps without finalizing
            logger.info("Max steps reached, truncating episode.")

        return observation, reward, terminated, truncated, info

    def close(self):
        pass


# ======================================================
# --- Test Cases ---
# ======================================================
if __name__ == '__main__':
    print("\n--- Testing DiodeDesignEnv ---")
    env = DiodeDesignEnv(max_steps=16) # Set max_steps slightly > checklist items

    # Test 1: Reset and Initial Observation
    print("\n--- Test 1: Reset ---")
    obs, info = env.reset()
    print(f"Initial Observation (shape {obs.shape}):\n{obs}")
    print(f"Initial Info: {info}")
    assert obs.shape == (OBS_DIM,), "Observation shape mismatch"
    assert len(info['completed_steps']) == 0, "Initial state should have no completed steps"
    print("Reset Test Passed.")

    # Test 2: Single Valid Step
    print("\n--- Test 2: Single Valid Step (INIT_MESH) ---")
    action = DIODE_CHECKLIST.index("INIT_MESH")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation after step:\n{obs}")
    print(f"Reward: {reward:.3f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info: {info}")
    assert not terminated and not truncated, "Should not terminate/truncate after one valid step"
    assert "INIT_MESH" in info['completed_steps'], "INIT_MESH should be completed"
    assert obs[action] == 1.0, "Observation flag for INIT_MESH should be 1.0"
    print("Single Step Test Passed.")

    # Test 3: Invalid Step (Dependency Fail)
    print("\n--- Test 3: Invalid Step (Dependency Fail - DEFINE_P_DOPING early) ---")
    action = DIODE_CHECKLIST.index("DEFINE_P_DOPING")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation after invalid step:\n{obs}")
    print(f"Reward: {reward:.3f}")
    print(f"Terminated: {terminated}, Truncated: {truncated}")
    print(f"Info: {info}")
    assert terminated, "Episode should terminate on dependency failure"
    assert reward < -5.0, "Should receive significant penalty for dependency failure" # Check reward range
    assert "DEFINE_P_DOPING" not in info['completed_steps'], "DEFINE_P_DOPING should not be completed"
    print("Invalid Step (Dependency) Test Passed.")

    # Test 4: Sequence leading to Finalize (using placeholders)
    print("\n--- Test 4: Sequence leading to Finalize ---")
    obs, info = env.reset()
    # Define a plausible (though not necessarily optimal) sequence based on dependencies
    # This sequence assumes default Na/Nd are set by the placeholder _execute_devsim_step
    valid_sequence = [
        "INIT_MESH", "DEFINE_GEOMETRY", "DEFINE_REGIONS", "DEFINE_CONTACTS",
        "FINALIZE_MESH", "CREATE_DEVICE", "SET_MATERIAL_PARAMS",
        "DEFINE_P_DOPING", "DEFINE_N_DOPING", "DEFINE_NET_DOPING",
        "DEFINE_VARIABLES", "SETUP_PHYSICS", "SETUP_EQUATIONS",
        "SETUP_CONTACT_BC", "FINALIZE_SETUP_RUN_TEST"
    ]
    final_reward = 0
    final_info = {}
    step_count = 0
    for item_name in valid_sequence:
        step_count += 1
        action = DIODE_CHECKLIST.index(item_name)
        print(f"\nTaking action: {item_name} ({action})")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f" Obs[{item_name} flag]: {obs[action]:.1f}, Obs[Na]: {obs[PARAM_NA_IDX]:.2f}, Obs[Nd]: {obs[PARAM_ND_IDX]:.2f}, Obs[Step]: {obs[STEP_COUNT_IDX]:.2f}")
        print(f" Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}")
        print(f" Completed: {info.get('completed_steps', [])}")
        final_reward = reward
        final_info = info
        if terminated or truncated:
            print(f"Episode ended early at step {step_count}.")
            break
    assert terminated, "Episode should terminate after finalize action"
    assert not truncated, "Episode should not truncate if finalized correctly"
    assert "FINALIZE_SETUP_RUN_TEST" in info['completed_steps'], "Finalize step should be marked completed"
    assert 'final_metrics' in info, "Info should contain final metrics after successful finalize"
    print(f"Final Info after sequence: {final_info}")
    print("Valid Sequence Test Passed.")


    # Test 5: Max Steps Truncation
    print("\n--- Test 5: Max Steps Truncation ---")
    obs, info = env.reset()
    final_reward = 0
    # Take valid steps but fewer than required to finalize
    partial_sequence = ["INIT_MESH", "DEFINE_GEOMETRY", "DEFINE_REGIONS", "DEFINE_CONTACTS"]
    for i in range(env._max_steps):
        # Cycle through a few valid early steps or choose randomly from allowed?
        # For simplicity, let's just repeat the first few valid steps
        action_name = partial_sequence[i % len(partial_sequence)]
        # Ensure dependencies met (implicitly true for first few)
        action = DIODE_CHECKLIST.index(action_name)
        print(f" Step {i+1}/{env._max_steps}, Action: {action_name}")
        obs, reward, terminated, truncated, info = env.step(action)
        final_reward = reward
        if terminated or truncated:
            break

    print(f"Final state after {env._current_step} steps:")
    print(f" Observation:\n{obs}")
    print(f" Reward: {final_reward:.3f}")
    print(f" Terminated: {terminated}, Truncated: {truncated}")
    print(f" Info: {info}")
    assert truncated, "Episode should truncate after max steps"
    assert not terminated, "Episode should not terminate if finalize wasn't called"
    assert final_reward < -1.0, "Should receive penalty for truncation without finalization"
    print("Max Steps Truncation Test Passed.")


    # Test 6: Repeating an action
    print("\n--- Test 6: Repeating an action ---")
    obs, info = env.reset()
    action = DIODE_CHECKLIST.index("INIT_MESH")
    print("Taking action INIT_MESH...")
    obs, reward1, terminated1, truncated1, info1 = env.step(action)
    print(f"Reward1: {reward1:.3f}, Completed: {info1.get('completed_steps')}")
    assert reward1 == -0.01, "First step should have small cost"

    print("Taking action INIT_MESH again...")
    obs, reward2, terminated2, truncated2, info2 = env.step(action) # Repeat
    print(f"Reward2: {reward2:.3f}, Completed: {info2.get('completed_steps')}")
    assert reward2 == -1.0, "Repeating action should have penalty"
    assert not terminated2 and not truncated2, "Repeating action should not terminate/truncate"
    print("Repeat Action Test Passed.")

    print("\n--- All Tests Completed ---")