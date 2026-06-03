from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from core.env import HAP_IoT_Env
import config as cfg
import torch as th

def run_training():
    # 1. Setup Parallel Training Environments for the RTX 5000
    env = SubprocVecEnv([lambda: Monitor(HAP_IoT_Env()) for _ in range(cfg.N_ENVS)])
    
    # 2. Setup a single evaluation environment
    eval_env = Monitor(HAP_IoT_Env()) 
    
    # 3. Configure the HPPO Callback
    # Saving to a separate folder so it DOES NOT overwrite your DDPG research data!
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./best_model_hppo/',  
        log_path='./logs_hppo/', 
        eval_freq=2500, 
        deterministic=True, 
        render=False
    )

    # 4. HPPO Custom Architecture
    # We split the Actor (pi) and Critic (vf) into heterogeneous, deep neural networks
    hppo_policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
    )

    # 5. Initialize the HPPO Agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.LR, 
        n_steps=2048,           # The rollout buffer size before updating
        batch_size=128,         # Standard batch size for stability
        policy_kwargs=hppo_policy_kwargs, # Injecting the custom Heterogeneous layers
        verbose=1,
        device="cuda",
        tensorboard_log="./hdrl_logs/"
    )

    print("--- 🚀 HPPO BLAST OFF ---")
    
    # 6. Start Training
    model.learn(total_timesteps=cfg.TOTAL_STEPS, callback=eval_callback) 
    
    # 7. Save the final model state
    model.save("train/Final_Agent_HPPO")