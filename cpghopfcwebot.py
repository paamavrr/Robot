import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# --- Parámetros compartidos con tu controlador ---
A = 0.76
f = 0.599
omega = 2 * np.pi * f
stance_ratio = 0.6
offset = 0.1322
phase_offsets = [0, 0, np.pi, np.pi]  # LH, RF, LF, RH
legs = ['LH', 'RF', 'LF', 'RH']
leg_colors = {
    'LF': 'tab:blue',
    'RF': 'tab:orange',
    'RH': 'tab:green',
    'LH': 'tab:red'
}

# --- Modelo Hopf acoplado ---
alpha = 10.0
mu = 1.0
K = 2.0
T = 5
t_eval = np.linspace(0, T, 1000)

# Estado inicial: [x_LH, y_LH, x_RF, y_RF, x_LF, y_LF, x_RH, y_RH]
y0 = []
for phi in phase_offsets:
    y0 += [np.cos(phi), np.sin(phi)]

def hopf_cpg(t, y):
    dydt = np.zeros_like(y)
    for i in range(4):
        xi, yi = y[2*i], y[2*i+1]
        coupling = 0
        for j in range(4):
            if i != j:
                phi_ij = phase_offsets[j] - phase_offsets[i]
                theta_i = np.arctan2(yi, xi)
                theta_j = np.arctan2(y[2*j+1], y[2*j])
                coupling += np.sin((theta_j - theta_i) - phi_ij)
        dydt[2*i]   = alpha * (mu - (xi**2 + yi**2)) * xi - omega * yi + K * coupling
        dydt[2*i+1] = alpha * (mu - (xi**2 + yi**2)) * yi + omega * xi
    return dydt

sol = solve_ivp(hopf_cpg, [0, T], y0, t_eval=t_eval)

# Ángulos articulares Hopf (con asimetría stance/swing igual a tu CPG simple)
theta_hopf = np.zeros((4, len(sol.t)))
for i in range(4):
    # Calcula la fase normalizada [0,1] a partir del oscilador Hopf
    phase = (np.arctan2(sol.y[2*i+1], sol.y[2*i]) + np.pi) / (2 * np.pi)
    for j, p in enumerate(phase):
        if p < stance_ratio:
            progress = p / stance_ratio
            target_angle = -A * (1 - progress) + offset
        else:
            swing_phase = (p - stance_ratio) / (1 - stance_ratio)
            target_angle = A * np.sin(swing_phase * np.pi) - offset
        theta_hopf[i, j] = target_angle

# Guardar ángulos Hopf en CSV
df_hopf = pd.DataFrame(theta_hopf.T, columns=legs)
df_hopf.to_csv("hopf_angles.csv", index=False)

# Graficar ángulos articulares Hopf
plt.figure(figsize=(10,5))
for i, leg in enumerate(legs):
    plt.plot(sol.t, theta_hopf[i], label=leg, color=leg_colors[leg])
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo articular (rad)')
plt.title('CPG Hopf para 4 patas (ángulos articulares, asimétrico)')
plt.legend()
plt.grid()
plt.show()

# Señal stance/swing Hopf (más tiempo en apoyo para estabilidad)
stance_threshold = -0.4  # Ajusta para más/menos tiempo en stance
stance_hopf = np.zeros_like(theta_hopf)
for i in range(4):
    stance_hopf[i] = (theta_hopf[i] > stance_threshold).astype(int)

# Guardar fases Hopf en CSV
#df_stance_hopf = pd.DataFrame(stance_hopf.T, columns=legs)
#df_stance_hopf.to_csv("hopf_stance.csv", index=False)

# Graficar diagrama de fases de marcha (Stance/Swing) - Hopf
fig, ax = plt.subplots(figsize=(10, 4))
for i, leg in enumerate(legs):
    ax.step(sol.t, stance_hopf[i] + 2*i, where='post', label=leg, color=leg_colors[leg])
ax.set_yticks([0,2,4,6])
ax.set_yticklabels(legs)
ax.set_xlabel('Tiempo (s)')
ax.set_title('Diagrama de fases de marcha (Stance/Swing) - Hopf')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()