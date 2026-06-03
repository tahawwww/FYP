# %% 
# ==========================================
# Section 1: Creating 3D Model
# ==========================================
import numpy as np
import matplotlib.pyplot as plt

# Coordinates
IoT = np.array([0, 0, 0])
RIS = np.array([1000, 100, 750])
HAP = np.array([-3000, -2000, 20000])
UAV = np.array([2000, 2000, 5000])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter Points
ax.scatter(*IoT, c='b', marker='o', s=50, label='IoT Source')
ax.scatter(*RIS, c='g', marker='s', s=50, label='RIS Array')
ax.scatter(*HAP, c='m', marker='d', s=50, label='HAP Platform')
ax.scatter(*UAV, c='r', marker='^', s=50, label='UAV Eavesdropper')

# Draw Atmospheric Layers
x_vals = np.arange(-5000, 6000, 1000)
y_vals = np.arange(-5000, 6000, 1000)
X, Y = np.meshgrid(x_vals, y_vals)
Z_ground = np.zeros_like(X)
Z_atmos = np.ones_like(X) * 10000

ax.plot_surface(X, Y, Z_ground, alpha=0.1, color='gray', edgecolor='none')
ax.plot_surface(X, Y, Z_atmos, alpha=0.05, color='cyan', edgecolor='none')

# Draw Links
ax.plot([IoT[0], RIS[0], HAP[0]], [IoT[1], RIS[1], HAP[1]], [IoT[2], RIS[2], HAP[2]], 'b-', linewidth=1.5, label='Legitimate Link')
ax.plot([IoT[0], UAV[0]], [IoT[1], UAV[1]], [IoT[2], UAV[2]], 'r--', label='Leakage Link')

# Formatting
ax.view_init(elev=30, azim=45)
ax.set_zlim([0, 22000])
ax.set_xlabel('X(m)')
ax.set_ylabel('Y(m)')
ax.set_zlabel('Alt(m)')
ax.legend()
plt.title('3D RIS-Assisted HAP-IoT Network')
plt.show()

# %% 
# ==========================================
# Section 2: Implementing Path Loss
# ==========================================
freq_GHz = 28
alpha = 0.06
zeta_U = 6

# Euclidean distances (Converted to km)
d_IoT_RIS = np.linalg.norm(IoT - RIS) / 1000
d_RIS_HAP = np.linalg.norm(RIS - HAP) / 1000
d_IoT_HAP = np.linalg.norm(IoT - HAP) / 1000
d_IoT_UAV = np.linalg.norm(IoT - UAV) / 1000
d_RIS_UAV = np.linalg.norm(RIS - UAV) / 1000
d_HAP_UAV = np.linalg.norm(HAP - UAV) / 1000

# Pathlosses (dB)
pathloss_IoT_RIS = 92.45 + 20*np.log10(freq_GHz) + 20*np.log10(d_IoT_RIS) + alpha*d_IoT_RIS
pathloss_RIS_HAP = 92.45 + 20*np.log10(freq_GHz) + 20*np.log10(d_RIS_HAP) + alpha*d_RIS_HAP
pathloss_IoT_HAP = 92.45 + 20*np.log10(freq_GHz) + 20*np.log10(d_IoT_HAP) + alpha*d_IoT_HAP

pathloss_IoT_UAV = 92.45 + 20*np.log10(freq_GHz) + 20*np.log10(d_IoT_UAV) + alpha*d_IoT_UAV + zeta_U
pathloss_RIS_UAV = 92.45 + 20*np.log10(freq_GHz) + 20*np.log10(d_RIS_UAV) + alpha*d_RIS_UAV + zeta_U
pathloss_HAP_UAV = 92.45 + 20*np.log10(freq_GHz) + 20*np.log10(d_HAP_UAV) + alpha*d_HAP_UAV + zeta_U

# Linear scale power gains
G_IoT_RIS = 10**(-pathloss_IoT_RIS / 10)
G_RIS_HAP = 10**(-pathloss_RIS_HAP / 10)
G_IoT_HAP = 10**(-pathloss_IoT_HAP / 10)
G_IoT_UAV = 10**(-pathloss_IoT_UAV / 10)
G_RIS_UAV = 10**(-pathloss_RIS_UAV / 10)
G_HAP_UAV = 10**(-pathloss_HAP_UAV / 10)

print("Path Loss Initialization Complete.")

# %% 
# ==========================================
# Section 3: Implementing SINR for HAP and UAV
# ==========================================
# Nakagami-m parameters
m_IoT_RIS = 2.0
m_RIS_HAP = 3.0
m_IoT_HAP = 3.0
m_RIS_UAV = 1.5
m_IoT_UAV = 2.0

# Simulation parameters
N = 64
time_to_run = 1000 
Pt = 10**((15 - 30) / 10)       # 15 dBm to Watts
Pj = 10**((10 - 30) / 10)       # 10 dBm to Watts
sigma2 = 10**((-100 - 30) / 10) # -100 dBm to Watts
kappa = 10**-10 

# Hardware parameters (Linear)
Gt = 10**(8 / 10) 
G_HAP = 10**(20 / 10) 
G_UAV = 10**(1 / 10) 

SINR_HAP = np.zeros(time_to_run)
SINR_UAV = np.zeros(time_to_run)

# Helper function for Nakagami generation
def generate_nakagami(m, size=1):
    return np.sqrt(np.random.gamma(shape=m, scale=1/m, size=size)) * np.exp(1j * 2 * np.pi * np.random.rand(size))

for t in range(time_to_run):
    
    # Direct Links
    h_IoT_HAP_f = np.sqrt(G_IoT_HAP) * generate_nakagami(m_IoT_HAP)[0]
    h_IoT_UAV_f = np.sqrt(G_IoT_UAV) * generate_nakagami(m_IoT_UAV)[0]
    h_HAP_UAV_f = np.sqrt(G_HAP_UAV) * generate_nakagami(m_IoT_UAV)[0] 
    
    # RIS Links (Arrays of size N)
    g_RIS = np.sqrt(G_IoT_RIS) * generate_nakagami(m_IoT_RIS, N)
    u_HAP = np.sqrt(G_RIS_HAP) * generate_nakagami(m_RIS_HAP, N)
    u_UAV = np.sqrt(G_RIS_UAV) * generate_nakagami(m_RIS_UAV, N)

    # Self interference fading (m=3)
    h_SI = generate_nakagami(3.0)[0]
    
    # Phases of channels
    theta_rand = np.random.rand()
    phi = np.random.rand() * np.exp(1j * theta_rand) 
        
    # Total channel gain
    H_total = h_IoT_HAP_f + np.sum(g_RIS * phi * u_HAP)
    U_total = h_IoT_UAV_f + np.sum(g_RIS * phi * u_UAV)

    # Power gains
    GH = (np.abs(H_total)**2) * Gt * G_HAP
    GU = (np.abs(U_total)**2) * Gt * G_UAV
    G_JU_eff = (np.abs(h_HAP_UAV_f)**2) * G_HAP * G_UAV 
    G_JH_eff = np.abs(h_SI)**2

    # SINR Calculation
    SINR_HAP[t] = (Pt * GH) / (sigma2 + kappa * Pj * G_JH_eff)
    SINR_UAV[t] = (Pt * GU) / (sigma2 + Pj * G_JU_eff)

# Convert to dB for visualization
SINR_HAP_dB = 10 * np.log10(SINR_HAP)
SINR_UAV_dB = 10 * np.log10(SINR_UAV)

plt.figure(figsize=(10, 5))
plt.plot(range(time_to_run), SINR_HAP_dB, 'b', linewidth=1.2, label='HAP (Legitimate)')
plt.plot(range(time_to_run), SINR_UAV_dB, 'r', linewidth=1.2, label='UAV (Eavesdropper)')
plt.grid(True)
plt.xlabel('Time (Seconds)')
plt.ylabel('SINR (dB)')
plt.title(f'1000-Second Real-Time SINR with N = {N}')
plt.legend(loc='best')
plt.show()

# %% 
# ==========================================
# Section 4: Transmission Rate and Secrecy Capacity
# ==========================================
system_bandwidth = 20e6
transmission_rate_HAP = system_bandwidth * np.log2(1 + SINR_HAP)
transmission_rate_UAV = system_bandwidth * np.log2(1 + SINR_UAV)

# Secrecy capacity (Element-wise maximum of array and 0)
secrecy_capacity = np.maximum(0, transmission_rate_HAP - transmission_rate_UAV)

plt.figure(figsize=(10, 5))
plt.plot(secrecy_capacity / 1e6, 'g')
plt.ylabel('Capacity (Mbps)')
plt.xlabel('Time (s)')
plt.title('Secrecy Capacity / Time')
plt.grid(True)
plt.show()

# %% 
# ==========================================
# Section 5: AoI and AoLI Standard Model
# ==========================================
gamma_th = 10**(-10 / 10) 
gamma_E = 10**(-18 / 10)  

AoI = np.zeros(time_to_run)
AoLI = np.zeros(time_to_run)

current_aoi = 1
current_aoli = 1

for t in range(time_to_run):
    # AOI for HAP
    if SINR_HAP[t] >= gamma_th:
        current_aoi = 1 
    else:
        current_aoi += 1 
    AoI[t] = current_aoi
    
    # AOLI for UAV
    if SINR_UAV[t] >= gamma_E:
        current_aoli = 1 
    else:
        current_aoli += 1 
    AoLI[t] = current_aoli

plt.figure(figsize=(10, 5))
plt.plot(range(time_to_run), AoLI, 'r', linewidth=1.2, label='AoLI (Security Staleness)')
plt.plot(range(time_to_run), AoI, 'b', linewidth=1.2, label='AoI (Freshness)')
plt.ylabel('Age (Seconds)')
plt.title('Networking Layer: AoI vs AoLI')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# %% 
# ==========================================
# Section 6: Security-Aware AoI and AoLI Optimization
# ==========================================
AoI_sec = np.zeros(time_to_run)
AoLI_opt = np.zeros(time_to_run)
current_aoi = 1
current_aoli = 1
blocked_transmissions = 0 

for t in range(time_to_run):
    # Decision maker condition
    if SINR_UAV[t] >= gamma_E:
        transmit_decision = 0 
        blocked_transmissions += 1
    else:
        transmit_decision = 1 
    
    # Secure AOI
    if transmit_decision == 1 and SINR_HAP[t] >= gamma_th:
        current_aoi = 1              
    else:
        current_aoi += 1 
    AoI_sec[t] = current_aoi
    
    # Optimized AOLI
    if transmit_decision == 1 and SINR_UAV[t] >= gamma_E:
        current_aoli = 1   
    else:
        current_aoli += 1 
    AoLI_opt[t] = current_aoli

print('--- Security-Aware Results ---')
print(f'Transmissions Blocked for Security: {blocked_transmissions} out of {time_to_run}')
print(f'Avg Security-Aware AoI : {np.mean(AoI_sec):.2f} seconds')
print(f'Avg Optimized AoLI     : {np.mean(AoLI_opt):.2f} seconds')

plt.figure(figsize=(12, 5))
plt.plot(range(time_to_run), AoLI_opt, 'r', linewidth=1.5, label='Optimized AoLI (Staleness)')
plt.plot(range(time_to_run), AoI_sec, 'b', linewidth=1.2, label='Security-Aware AoI (Freshness)')
plt.ylabel('Age (Seconds)')
plt.xlabel('Time (s)')
plt.title('The Secrecy-Freshness Trade-off (Gated Transmission)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# %% 
# ==========================================
# Section 7: HDRL Integration Placeholder
# ==========================================
print("Environment successfully initialized. Ready to wrap in Gymnasium and apply DDPG.")