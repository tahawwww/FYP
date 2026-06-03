from stable_baselines3 import DDPG
from core.env import HAP_IoT_Env
import numpy as np
import config as cfg

# Import your charting tools
from evaluate.plots import save_performance_table, plot_aoi_v_aoli

def run_evaluation():
    env = HAP_IoT_Env()
    
    # Always load your peak performance weights
    model = DDPG.load("best_model/best_model.zip") 
    
    # Save the original distances/gains so we can restore them safely
    original_gains = cfg.G_LINKS.copy()
    
    # --- THE ULTIMATE ABLATION & SCENARIO LIST ---
    scenarios = [
        {
            "name": "1. Standard (Optimal)",
            "gains": {}, 
            "force_pt": None, 
            "force_pj": None
        },
        # --- LOCATION TESTS ---
        {
            "name": "2. Close UAV (High Threat)",
            "gains": {'IU': original_gains['IU'] * 10, 'RU': original_gains['RU'] * 10},
            "force_pt": None, "force_pj": None
        },
        {
            "name": "3. Distant UAV (Low Threat)",
            "gains": {'IU': original_gains['IU'] * 0.1, 'RU': original_gains['RU'] * 0.1},
            "force_pt": None, "force_pj": None
        },
        {
            "name": "4. High Alt HAP (Weak Signal)",
            # HAP moves further up. The signal to the IoT and RIS gets weaker.
            "gains": {'IH': original_gains['IH'] * 0.1, 'RH': original_gains['RH'] * 0.1},
            "force_pt": None, "force_pj": None
        },
        # --- HARDWARE TESTS ---
        {
            "name": "5. Low Transmit Power",
            "gains": {},
            "force_pt": -0.6, # Forces agent to use low Pt
            "force_pj": None
        },
        {
            "name": "6. No Jamming Allowed",
            "gains": {},
            "force_pt": None,
            "force_pj": -1.0 # Forces agent to use 0W Pj
        }
    ]
    
    final_data = []
    
    print("--- ⚡ RUNNING ULTIMATE 6G EVALUATION ---")
    
    for scenario in scenarios:
        print(f"\nTesting: {scenario['name']}...")
        
        # 1. Reset Physics to Default, then apply the scenario's specific gains
        cfg.G_LINKS.update(original_gains) 
        if scenario["gains"]:
            cfg.G_LINKS.update(scenario["gains"])
        
        obs, _ = env.reset()
        aoi_list, aoli_list = [], []
        
        for _ in range(1000):
            # Ask the AI what it *wants* to do
            action, _ = model.predict(obs, deterministic=True)
            
            # OVERRIDE the AI based on the hardware constraints
            if scenario["force_pt"] is not None:
                action[0] = scenario["force_pt"]
            if scenario["force_pj"] is not None:
                action[1] = scenario["force_pj"]
                
            # Step the environment
            obs, _, _, _, _ = env.step(action)
            
            aoi_list.append(obs[0] * 100)
            aoli_list.append(obs[1] * 100)
            
        # Calculate averages
        avg_aoi = np.mean(aoi_list)
        avg_aoli = np.mean(aoli_list)
        print(f"Result -> Avg AoI: {avg_aoi:.2f} | Avg AoLI: {avg_aoli:.2f}")
        
        final_data.append({
            'Test': scenario['name'], 
            'Avg AoI': avg_aoi, 
            'Avg AoLI': avg_aoli
        })

    # Restore the original config physics just to be totally safe
    cfg.G_LINKS.update(original_gains)

    # Generate the ultimate side-by-side plot
    print("\n--- 📊 GENERATING FULL PAPER METRICS ---")
    save_performance_table(final_data)
    plot_aoi_v_aoli(final_data)
    print("✅ All done! Check results/tradeoff_plot.png")