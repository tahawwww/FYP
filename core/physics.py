import numpy as np
import config as cfg

def get_nakagami_fading(m, size=1):
    # Nakagami-m fading generation
    shape = m
    scale = 1/m
    magnitude = np.sqrt(np.random.gamma(shape, scale, size))
    phase = np.exp(1j * 2 * np.pi * np.random.rand(size))
    return magnitude * phase

def calculate_h_total(phi):
    # Calculate the total channel gain h_total for the HAP link
    h_ih = np.sqrt(cfg.G_LINKS['IH']) * get_nakagami_fading(3.0)
    g_ir = np.sqrt(cfg.G_LINKS['IR']) * get_nakagami_fading(2.0, cfg.N_RIS_ELEMENTS)
    u_rh = np.sqrt(cfg.G_LINKS['RH']) * get_nakagami_fading(3.0, cfg.N_RIS_ELEMENTS)
    
    return h_ih + np.sum(g_ir * phi * u_rh)

def get_sinr(Pt, Pj, h_total):
    # Calculate the SINR for the HAP link
    h_sq = np.abs(h_total)**2
    return (Pt * h_sq) / (cfg.SIGMA2 + cfg.KAPPA * Pj)