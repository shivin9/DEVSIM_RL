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
TOTAL_TIMESTEPS = 10000000  # Start with a smaller number to test, increase for real training
EVAL_FREQ = 1000  # How often to evaluate the model
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

    # --- Agent Training ---
    print(f"Initializing PPO agent for Diode Design...")
    # PPO is a good default choice for this kind of discrete action space
    model = PPO("MlpPolicy",
                env,
                verbose=1,  # Print training progress
                tensorboard_log=LOG_DIR,
                # Hyperparameters can be tuned for better performance
                # learning_rate=3e-4,
                n_steps=1024*10,
                # batch_size=64,
                # gamma=0.99,
                )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")

    # Setup callback for evaluation and saving the best model
    # This will periodically test the agent's performance on a separate instance
    # of the environment and save the model if it's the best one seen so far.
    eval_env = make_env()
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=BEST_MODEL_SAVE_PATH,
                                 log_path=LOG_DIR,
                                 eval_freq=EVAL_FREQ,
                                 deterministic=True,
                                 render=False)

    # Train the agent!
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=eval_callback,
                    tb_log_name=MODEL_NAME,
                    progress_bar=True)
    except Exception as e:
        print(f"An error occurred during training: {e}")
        print("Saving model before exiting...")
        model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_error_exit"))
    finally:
        # Save the final model
        final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final")
        model.save(final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")
        env.close()
        eval_env.close()

    # --- Example of Using the Trained Model ---
    print("\nLoading the best model for a test run...")
    try:
        best_model = PPO.load(os.path.join(BEST_MODEL_SAVE_PATH, "best_model"))

        # Test the loaded model
        test_env = make_env()
        obs, _ = test_env.reset()
        print("Initial test observation received.")
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        while not terminated and not truncated:
            action, _states = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            step += 1
            print(f"Test Step {step}: Action={info['action_name']}, Reward={reward:.3f}")
            if terminated or truncated:
                print("Episode finished.")
                if 'final_script' in info:
                    print("\n--- Generated Script by Agent ---")
                    print(info['final_script'])
                    print("--- End of Script ---")

        print(f"\nTest finished. Total reward: {total_reward:.2f}")
        test_env.close()
    except FileNotFoundError:
        print(f"Could not find a saved best model. Please ensure training ran long enough to produce one.")
    except Exception as e:
        print(f"An error occurred while testing the model: {e}")