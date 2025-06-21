import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros Hopf ajustados
alpha = 10.0
mu = 1.0
A = 0.6           # Amplitud recomendada (rango seguro para tus patas)
f = 0.3           # Frecuencia más baja para máxima estabilidad
omega = 2 * np.pi * f
T = 5
t_eval = np.linspace(0, T, 1000)

# Desfases para marcha tipo "trote" diagonal
phase_offsets = {
    'LH': 0,
    'RF': 0,
    'LF': np.pi,
    'RH': np.pi
}
legs = ['LH', 'RF', 'LF', 'RH']
leg_colors = {
    'LF': 'tab:blue',
    'RF': 'tab:orange',
    'RH': 'tab:green',
    'LH': 'tab:red'
}

# Estado inicial: [x_LF, y_LF, x_RF, y_RF, x_RH, y_RH, x_LH, y_LH]
y0 = []
for leg in legs:
    phi = phase_offsets[leg]
    y0 += [np.cos(phi), np.sin(phi)]

def hopf_cpg(t, y):
    dydt = np.zeros_like(y)
    for i, leg in enumerate(legs):
        xi, yi = y[2*i], y[2*i+1]
        coupling = 0
        for j, leg2 in enumerate(legs):
            if i != j:
                phi_ij = phase_offsets[leg2] - phase_offsets[leg]
                theta_i = np.arctan2(yi, xi)
                theta_j = np.arctan2(y[2*j+1], y[2*j])
                coupling += np.sin((theta_j - theta_i) - phi_ij)
        K = 2.0
        dydt[2*i]   = alpha * (mu - (xi**2 + yi**2)) * xi - omega * yi + K * coupling
        dydt[2*i+1] = alpha * (mu - (xi**2 + yi**2)) * yi + omega * xi
    return dydt

sol = solve_ivp(hopf_cpg, [0, T], y0, t_eval=t_eval)

# Ángulos articulares (ajustados)
theta = {}
for i, leg in enumerate(legs):
    theta[leg] = A * sol.y[2*i]

# Graficar ángulos articulares
plt.figure(figsize=(10,5))
for leg in legs:
    plt.plot(sol.t, theta[leg], label=leg, color=leg_colors[leg])
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo articular (rad)')
plt.title('CPG Hopf para 4 patas (ángulos articulares)')
plt.legend()
plt.grid()
plt.show()

# Señal stance/swing (más tiempo en apoyo para estabilidad)
stance = {}
stance_threshold = -0.1  # Más tiempo en stance (apoyo)
for leg in legs:
    stance[leg] = (theta[leg] > stance_threshold).astype(int)

# Graficar diagrama de fases de marcha (Stance/Swing)
fig, ax = plt.subplots(figsize=(10, 4))
for i, leg in enumerate(legs):
    ax.step(sol.t, stance[leg] + 2*i, where='post', label=leg, color=leg_colors[leg])

ax.set_yticks([0,2,4,6])
ax.set_yticklabels(legs)
ax.set_xlabel('Tiempo (s)')
ax.set_title('Diagrama de fases de marcha (Stance/Swing) - Hopf')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Parámetros del cuerpo y patas (ajustados a tu robot)
body_length = 0.3
body_width = 0.2
leg_length = 0.24

# Posiciones de anclaje de las patas (coinciden con tu modelo Webots)
anchors = {
    'LF': [ body_length/2,  body_width/2],
    'RF': [ body_length/2, -body_width/2],
    'LH': [-body_length/2,  body_width/2],
    'RH': [-body_length/2, -body_width/2]
}

fig, ax = plt.subplots(figsize=(6,4))
ax.set_xlim(-0.25, 0.25)
ax.set_ylim(-0.15, 0.15)
ax.set_aspect('equal')
ax.set_title('Animación 2D de la marcha cuadrúpeda')
lines = {leg: ax.plot([], [], 'o-', color=leg_colors[leg], label=leg)[0] for leg in legs}
ax.legend()

def update(frame):
    for leg in legs:
        angle = theta[leg][frame]
        anchor = anchors[leg]
        tip = [anchor[0] + leg_length * np.sin(angle),
               anchor[1]]
        lines[leg].set_data([anchor[0], tip[0]], [anchor[1], tip[1]])
    return lines.values()

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=20, blit=True)
plt.show()
