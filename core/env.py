import gymnasium as gym
from gymnasium import spaces
import numpy as np
import config as cfg
from core.physics import calculate_h_total, get_sinr
from core.metrics import SixGMetrics

class HAP_IoT_Env(gym.Env):
    def __init__(self):
        super(HAP_IoT_Env, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(cfg.N_RIS_ELEMENTS + 2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.metrics = SixGMetrics()
        self.time_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.metrics = SixGMetrics()
        self.time_step = 0
        return np.array([0.01, 0.01, 0.5], dtype=np.float32), {}

    def step(self, action):
        self.time_step += 1
        
        # Decode Action
        Pt, Pj = ((action[0:2] + 1) / 2) * cfg.MAX_POWER_WATT
        phi = np.exp(1j * action[2:] * np.pi)
        
        # Physics & Metrics
        h_total = calculate_h_total(phi)
        sinr_h = get_sinr(Pt, Pj, h_total)
        aoi, aoli = self.metrics.step(sinr_h)
        
        # Output
        reward = float(self.metrics.get_reward() * 0.001)

        h_mag = float(np.abs(h_total).item() if hasattr(h_total, "item") else np.abs(h_total))
        obs = np.array([
                float(aoi / 100.0), 
                float(aoli / 100.0), 
                h_mag
            ], dtype=np.float32)
        
        return obs, reward, False, self.time_step >= 1000, {}