import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from scipy.optimize import minimize

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
def evaluate_performance(params):
    # Parámetros físicos (puedes cambiarlos según tu robot)
    peso_robot = 4.4      # kg (ejemplo)
    gravedad = 9.81       # m/s^2
    largo_patas = [0.215, 0.215, 0.215, 0.215]  # metros, [LH, RF, LF, RH]

    # Desempaqueta los parámetros a optimizar
    A_opt, f_opt, stance_ratio_opt, offset_opt, K_opt = params
    omega_opt = 2 * np.pi * f_opt

    # Simulación Hopf con los nuevos parámetros
    def hopf_cpg_opt(t, y):
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
            dydt[2*i]   = alpha * (mu - (xi**2 + yi**2)) * xi - omega_opt * yi + K_opt * coupling
            dydt[2*i+1] = alpha * (mu - (xi**2 + yi**2)) * yi + omega_opt * xi
        return dydt

    sol_opt = solve_ivp(hopf_cpg_opt, [0, T], y0, t_eval=t_eval)
    theta_hopf_opt = np.zeros((4, len(sol_opt.t)))
    for i in range(4):
        phase = (np.arctan2(sol_opt.y[2*i+1], sol_opt.y[2*i]) + np.pi) / (2 * np.pi)
        for j, p in enumerate(phase):
            if p < stance_ratio_opt:
                progress = p / stance_ratio_opt
                target_angle = -A_opt * (1 - progress) + offset_opt
            else:
                swing_phase = (p - stance_ratio_opt) / (1 - stance_ratio_opt)
                target_angle = A_opt * np.sin(swing_phase * np.pi) - offset_opt
            theta_hopf_opt[i, j] = target_angle

    # --- Cálculo cinemático del avance usando el largo de cada pata ---
    pies_x = np.zeros_like(theta_hopf_opt)
    for i in range(4):
        pies_x[i] = largo_patas[i] * np.sin(theta_hopf_opt[i])

    # Solo cuenta el avance cuando la pata está en stance
    avance_patas = np.zeros_like(pies_x)
    for i in range(4):
        stance_mask = (theta_hopf_opt[i] > -0.4).astype(float)  # Ajusta el umbral si lo necesitas
        avance_patas[i] = pies_x[i] * stance_mask

    # Avance total del robot (promedio de patas en stance)
    avance_total = np.mean(avance_patas[:, -1] - avance_patas[:, 0])

    # --- Penalización por estabilidad física (opcional) ---
    # Ejemplo: penaliza si la suma de fuerzas de soporte es menor que el peso del robot
    # (esto es solo ilustrativo, puedes mejorarlo según tu modelo)
    soporte_promedio = np.mean(np.abs(avance_patas))
    penalty_stabilidad = 0
    if soporte_promedio * gravedad < peso_robot * gravedad * 0.5:
        penalty_stabilidad += 10

    # Penalización por valores extremos (opcional)
    penalty = 0
    if not (0.4 < stance_ratio_opt < 0.9):
        penalty += 10
    if not (0.05 < offset_opt < 0.3):
        penalty += 10
    if not (0.5 < K_opt < 5.0):
        penalty += 10

    # Penalización total
    penalty += penalty_stabilidad

    # Negativo porque minimize busca mínimos
    return -avance_total + penalty

# Parámetros iniciales (puedes usar los tuyos)
x0 = [A, f, stance_ratio, offset, K]
bounds = [
    (0.5, 2),    # Amplitud
    (0.3, 1.0),    # Frecuencia
    (0.4, 0.9),    # stance_ratio
    (0.05, 2),   # offset
    (0.5, 5.0)     # K (acoplamiento)
]

result = minimize(evaluate_performance, x0, bounds=bounds, method='L-BFGS-B')
A_opt, f_opt, stance_ratio_opt, offset_opt, K_opt = result.x

print(f"Mejor amplitud: {A_opt:.3f}, Mejor frecuencia: {f_opt:.3f}, Mejor stance_ratio: {stance_ratio_opt:.3f}, Mejor offset: {offset_opt:.3f}, Mejor K: {K_opt:.3f}")

# Usa estos valores para tu simulación final si quieres mantener los resultados optimizados