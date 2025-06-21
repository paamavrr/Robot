import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parámetros Hopf
alpha = 10.0
mu = 1.0
A = 0.5
f = 1.0
omega = 2 * np.pi * f
T = 5
t_eval = np.linspace(0, T, 1000)

# Desfases para marcha tipo "trote"
phase_offsets = {
    'LH': 0,
    'LF': np.pi/2,
    'RH': np.pi,
    'RF': 3*np.pi/2
}
legs = ['LH', 'LF', 'RH', 'RF']
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

# Acoplamiento de fases (simple: fuerza a mantener el desfase deseado)
def hopf_cpg(t, y):
    dydt = np.zeros_like(y)
    for i, leg in enumerate(legs):
        xi, yi = y[2*i], y[2*i+1]
        # Acoplamiento: suma de diferencias de fase con las otras patas
        coupling = 0
        for j, leg2 in enumerate(legs):
            if i != j:
                # Diferencia de fase deseada
                phi_ij = phase_offsets[leg2] - phase_offsets[leg]
                # Diferencia real
                theta_i = np.arctan2(yi, xi)
                theta_j = np.arctan2(y[2*j+1], y[2*j])
                coupling += np.sin((theta_j - theta_i) - phi_ij)
        # Peso de acoplamiento
        K = 2.0
        dydt[2*i]   = alpha * (mu - (xi**2 + yi**2)) * xi - omega * yi + K * coupling
        dydt[2*i+1] = alpha * (mu - (xi**2 + yi**2)) * yi + omega * xi
    return dydt

# Integrar el sistema
sol = solve_ivp(hopf_cpg, [0, T], y0, t_eval=t_eval)

# Obtener ángulos articulares (usamos x como señal de salida)
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

# Generar señales cuadradas para stance/swing
stance = {}
for leg in legs:
    stance[leg] = (theta[leg] > 0).astype(int)

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


# Agrega esto al final de tu script para animar el movimiento de las patas
import matplotlib.animation as animation

# Parámetros del cuerpo y patas
body_length = 0.3
body_width = 0.1
leg_length = 0.16

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
        # Ángulo actual de la pata
        angle = theta[leg][frame]
        anchor = anchors[leg]
        # Posición de la punta de la pata
        tip = [anchor[0] + leg_length * np.sin(angle),
               anchor[1]]
        lines[leg].set_data([anchor[0], tip[0]], [anchor[1], tip[1]])
    return lines.values()

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=20, blit=True)
plt.show()


