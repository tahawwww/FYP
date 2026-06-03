import config as cfg

class SixGMetrics:
    def __init__(self):
        self.aoi = 1.0
        self.aoli = 1.0

    def step(self, sinr_h):
        # Update Age of Information (AoI)
        if sinr_h >= cfg.GAMMA_TH:
            self.aoi = 1.0
        else:
            self.aoi = min(self.aoi + 1.0, cfg.MAX_AOI)
        
        # Update Age of Leakage (AoLI) - Increments until UAV eavesdrops
        self.aoli = min(self.aoli + 1.0, cfg.MAX_AOI)
        
        return self.aoi, self.aoli

    def get_reward(self):
        # The raw optimization function
        return self.aoli - (cfg.OMEGA * self.aoi)