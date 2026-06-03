from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from core.env import HAP_IoT_Env
import config as cfg

def run_training():
    # 1. Setup Parallel Training Environments (Wrapped in Monitor for TensorBoard)
    env = SubprocVecEnv([lambda: Monitor(HAP_IoT_Env()) for _ in range(cfg.N_ENVS)])
    
    # 2. Setup a single, separate environment strictly for Evaluating the agent
    eval_env = Monitor(HAP_IoT_Env()) 
    
    # 3. Configure the Callback (The "Safety Net")
    # It tests the agent every 2,500 steps (per env) and saves the best one.
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./best_model/',  # Automatically saves best_model.zip here
        log_path='./logs/', 
        eval_freq=2500, 
        deterministic=True, 
        render=False
    )

    # 4. Initialize the Agent
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=cfg.LR,
        verbose=1,
        device="cuda",
        tensorboard_log="./hdrl_logs/"
    )

    print("--- 🚀 BLAST OFF ---")
    
    # 5. Start Training (and pass the callback to it)
    model.learn(total_timesteps=cfg.TOTAL_STEPS, callback=eval_callback) 
    
    # 6. Save the final state at exactly 1,000,000 steps
    model.save("train/Final_Agent_DDPG")