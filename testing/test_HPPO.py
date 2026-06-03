from stable_baselines3 import PPO  # <-- IMPORTANT: Changed from DDPG to PPO
from core.env import HAP_IoT_Env
import numpy as np
import config as cfg

# Import your high-tier plotting tools
from evaluate.plots import save_performance_table, plot_aoi_v_aoli

def run_evaluation():
    env = HAP_IoT_Env()
    
    # LOAD THE HPPO MODEL: Pointing to the new folder we created!
    model = PPO.load("best_model_hppo/best_model.zip") 
    
    original_gains = cfg.G_LINKS.copy()
    
    # --- THE ULTIMATE ABLATION & SCENARIO LIST ---
    scenarios = [
        {
            "name": "1. HPPO Standard (Optimal)",
            "gains": {}, 
            "force_pt": None, 
            "force_pj": None
        },
        {
            "name": "2. HPPO Close UAV (High Threat)",
            "gains": {'IU': original_gains['IU'] * 10, 'RU': original_gains['RU'] * 10},
            "force_pt": None, "force_pj": None
        },
        {
            "name": "3. HPPO Distant UAV (Low Threat)",
            "gains": {'IU': original_gains['IU'] * 0.1, 'RU': original_gains['RU'] * 0.1},
            "force_pt": None, "force_pj": None
        },
        {
            "name": "4. HPPO High Alt HAP (Weak Signal)",
            "gains": {'IH': original_gains['IH'] * 0.1, 'RH': original_gains['RH'] * 0.1},
            "force_pt": None, "force_pj": None
        },
        {
            "name": "5. HPPO Low Transmit Power",
            "gains": {},
            "force_pt": -0.6, 
            "force_pj": None
        },
        {
            "name": "6. HPPO No Jamming Allowed",
            "gains": {},
            "force_pt": None,
            "force_pj": -1.0 
        }
    ]
    
    final_data = []
    
    print("--- ⚡ RUNNING HPPO 6G EVALUATION ---")
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}...")
        
        cfg.G_LINKS.update(original_gains) 
        if scenario["gains"]:
            cfg.G_LINKS.update(scenario["gains"])
        
        obs, _ = env.reset()
        aoi_list, aoli_list = [], []
        
        for _ in range(1000):
            # PPO prediction logic
            action, _ = model.predict(obs, deterministic=True)
            
            # Action Overrides for Hardware Ablation
            if scenario["force_pt"] is not None:
                action[0] = scenario["force_pt"]
            if scenario["force_pj"] is not None:
                action[1] = scenario["force_pj"]
                
            obs, _, _, _, _ = env.step(action)
            
            aoi_list.append(obs[0] * 100)
            aoli_list.append(obs[1] * 100)
            
        avg_aoi = np.mean(aoi_list)
        avg_aoli = np.mean(aoli_list)
        print(f"Result -> Avg AoI: {avg_aoi:.2f} | Avg AoLI: {avg_aoli:.2f}")
        
        final_data.append({
            'Test': scenario['name'], 
            'Avg AoI': avg_aoi, 
            'Avg AoLI': avg_aoli
        })

    cfg.G_LINKS.update(original_gains)

    print("\n--- 📊 GENERATING HPPO PAPER METRICS ---")
    save_performance_table(final_data)
    plot_aoi_v_aoli(final_data)
    print("✅ All done! Check results/tradeoff_plot.png")