import os
import numpy as np
from stable_baselines3 import PPO  # Or SAC, TD3 etc.
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv # For parallel envs

# Import the custom environment
from rl_environment import TransistorOptEnv

# --- Configuration ---
LOG_DIR = "logs"
MODEL_DIR = "models"
MODEL_NAME = "ppo_transistor_opt"
TOTAL_TIMESTEPS = 5000 # Increase significantly for real training (e.g., 100k+)
N_ENVS = 4 # Number of parallel environments (adjust based on CPU cores)
EVAL_FREQ = 500 # How often to evaluate the model
BEST_MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_best")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if __name__ == "__main__":
    print("Setting up environment(s)...")

    # Function to create an environment instance
    def make_env(rank=0, seed=0):
        def _init():
            # Use Monitor wrapper to log rewards and episode lengths
            env = Monitor(TransistorOptEnv(max_steps=100), # Longer episodes for training
                          filename=os.path.join(LOG_DIR, f'monitor_{rank}.csv'))
            env.reset(seed=seed + rank)
            return env
        return _init

    # Create vectorized environment (runs multiple envs in parallel)
    if N_ENVS > 1:
        env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    else:
        env = DummyVecEnv([make_env()])

    # It's recommended to check the custom environment conforms to the Gym API
    # print("Checking environment...")
    # check_env(env.envs[0]) # Check the underlying single env
    # print("Environment check passed.")

    # --- Agent Training ---
    print(f"Initializing PPO agent...")
    # PPO is a good default choice for continuous action spaces
    model = PPO("MlpPolicy", # Multi-layer perceptron policy
                env,
                verbose=1, # Print training progress
                tensorboard_log=LOG_DIR,
                # Adjust hyperparameters as needed:
                # learning_rate=3e-4,
                # n_steps=2048 // N_ENVS, # Steps per env before update
                # batch_size=64,
                # gamma=0.99, # Discount factor
                # ent_coef=0.0, # Entropy coefficient
                # vf_coef=0.5, # Value function coefficient
                )

    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")

    # Setup callback for evaluation and saving the best model
    # Use a separate evaluation environment (optional but good practice)
    eval_env = Monitor(TransistorOptEnv(max_steps=100))
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=BEST_MODEL_SAVE_PATH,
                                 log_path=LOG_DIR,
                                 eval_freq=max(EVAL_FREQ // N_ENVS, 1),
                                 deterministic=True,
                                 render=False)

    # Train the agent
    try:
         model.learn(total_timesteps=TOTAL_TIMESTEPS,
                     callback=eval_callback,
                     tb_log_name=MODEL_NAME,
                     progress_bar=True)
    except Exception as e:
         print(f"Error during training: {e}")
         print("Saving model before exiting...")
         model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_error_exit"))
    finally:
         # Save the final model
         final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final")
         model.save(final_model_path)
         print(f"Training finished. Final model saved to {final_model_path}")
         env.close()
         eval_env.close()

    # --- Using the trained model (Example) ---
    print("\nLoading the best model for prediction...")
    try:
        best_model = PPO.load(os.path.join(BEST_MODEL_SAVE_PATH, "best_model"))

        # Test the loaded model
        test_env = TransistorOptEnv(max_steps=10) # Short episode for testing
        obs, _ = test_env.reset()
        print(f"Initial test observation: {obs}")
        terminated = False
        truncated = False
        total_reward = 0
        step = 0
        while not terminated and not truncated:
            action, _states = best_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            step += 1
            print(f"Test Step {step}: Action={action}, Obs={obs}, Reward={reward:.2f}, Info={info}")

        print(f"Test finished. Total reward: {total_reward:.2f}")
        test_env.close()
    except FileNotFoundError:
        print(f"Could not find best model at {BEST_MODEL_SAVE_PATH}/best_model.zip")
    except Exception as e:
        print(f"Error loading or testing model: {e}")