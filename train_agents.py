import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the custom environment from your file
# Make sure devsim_environment.py is in the same directory or accessible
from devsim_environment import DiodeDesignEnv

# --- Configuration ---
LOG_DIR = "logs_diode"
MODEL_DIR = "models_diode"
MODEL_NAME = "ppo_diode_designer"
TOTAL_TIMESTEPS = 1000000  # Start with a smaller number to test, increase for real training
EVAL_FREQ = 500  # How often to evaluate the model
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# In train_diode_agent.py, add this function somewhere after the imports.

def sanity_check_environment():
    """
    Manually steps through the environment with the correct action sequence
    to ensure the logic and final reward are working as expected.
    """
    print("\n" + "="*50)
    print("--- RUNNING ENVIRONMENT SANITY CHECK ---")
    print("="*50)

    # A single instance of the environment, not wrapped
    env = DiodeDesignEnv()
    obs, info = env.reset()

    # The perfect sequence of actions
    # Note: DIODE_CHECKLIST is imported from the environment
    from devsim_environment import DIODE_CHECKLIST
    perfect_sequence = DIODE_CHECKLIST

    total_reward = 0
    final_reward = 0

    for i, action_name in enumerate(perfect_sequence):
        action_index = DIODE_CHECKLIST.index(action_name)
        print(f"\n--- Manual Step {i+1}: Taking action '{action_name}' ---")
        obs, reward, terminated, truncated, info = env.step(action_index)

        print(f"Reward received: {reward:.4f}")
        print(f"Completed Steps: {info.get('completed_steps')}")
        if info.get("error"):
            print(f"ERROR DETECTED: {info.get('error')}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")

        if not terminated and not truncated:
            total_reward += reward
        else:
            final_reward = reward # The last reward is the final one
            break

    print("\n" + "="*50)
    print("--- SANITY CHECK COMPLETE ---")
    print(f"Total intermediate rewards: {total_reward:.4f}")
    print(f"Final reward: {final_reward:.4f}")

    if final_reward > 40: # Check if we got the SUCCESS_BONUS
        print("RESULT: PASSED! The final reward is positive and large as expected.")
    else:
        print("RESULT: FAILED! The environment did not produce a large positive reward for a perfect sequence.")
    print("="*50 + "\n")


if __name__ == "__main__":
    print("Setting up DiodeDesignEnv environment...")

    # Create a function to instantiate the environment
    def make_env():
        # Use the Monitor wrapper to log rewards and episode lengths
        env = Monitor(DiodeDesignEnv(execute_on_finalize=True))
        return env

    # Create a vectorized environment
    env = DummyVecEnv([make_env])

    # It's a good practice to check if the custom environment follows the Gym API
    print("Checking environment...")
    # The check is done on an un-vectorized environment instance
    check_env(make_env())
    print("Environment check passed.")

    sanity_check_environment()

    # # --- Agent Training ---
    # print(f"Initializing PPO agent for Diode Design...")
    # # PPO is a good default choice for this kind of discrete action space
    # model = PPO("MlpPolicy",
    #             env,
    #             verbose=0,  # Print training progress
    #             tensorboard_log=LOG_DIR,
    #             # Hyperparameters can be tuned for better performance
    #             learning_rate=0.0003,
    #             n_steps=512,
    #             batch_size=64,
    #             gamma=0.99,
    #             )

    # print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")

    # # Setup callback for evaluation and saving the best model
    # # This will periodically test the agent's performance on a separate instance
    # # of the environment and save the model if it's the best one seen so far.
    # eval_env = make_env()
    # eval_callback = EvalCallback(eval_env,
    #                              best_model_save_path=BEST_MODEL_SAVE_PATH,
    #                              log_path=LOG_DIR,
    #                              eval_freq=EVAL_FREQ,
    #                              deterministic=True,
    #                              render=False)

    # # Train the agent!
    # try:
    #     model.learn(total_timesteps=TOTAL_TIMESTEPS,
    #                 callback=eval_callback,
    #                 tb_log_name=MODEL_NAME,
    #                 progress_bar=True)
    # except Exception as e:
    #     print(f"An error occurred during training: {e}")
    #     print("Saving model before exiting...")
    #     model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_error_exit"))
    # finally:
    #     # Save the final model
    #     final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final")
    #     model.save(final_model_path)
    #     print(f"Training finished. Final model saved to {final_model_path}")
    #     env.close()
    #     eval_env.close()

    # # --- Example of Using the Trained Model ---
    # print("\nLoading the best model for a test run...")
    # try:
    #     best_model = PPO.load(os.path.join(BEST_MODEL_SAVE_PATH, "best_model"))

    #     # Test the loaded model
    #     test_env = make_env()
    #     obs, _ = test_env.reset()
    #     print("Initial test observation received.")
    #     terminated = False
    #     truncated = False
    #     total_reward = 0
    #     step = 0
    #     while not terminated and not truncated:
    #         action, _states = best_model.predict(obs, deterministic=True)
    #         obs, reward, terminated, truncated, info = test_env.step(action)
    #         total_reward += reward
    #         step += 1
    #         print(f"Test Step {step}: Action={info['action_name']}, Reward={reward:.3f}")
    #         if terminated or truncated:
    #             print("Episode finished.")
    #             if 'final_script' in info:
    #                 print("\n--- Generated Script by Agent ---")
    #                 print(info['final_script'])
    #                 print("--- End of Script ---")

    #     print(f"\nTest finished. Total reward: {total_reward:.2f}")
    #     test_env.close()
    # except FileNotFoundError:
    #     print(f"Could not find a saved best model. Please ensure training ran long enough to produce one.")
    # except Exception as e:
    #     print(f"An error occurred while testing the model: {e}")
