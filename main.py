import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def cuadrupedo_edo(t, y, params):
    """
    y: vector de estado [θ1, ω1, θ2, ω2, θ3, ω3, θ4, ω4]
    params: diccionario con parámetros {k, b, I}
    """
    k = params['k']      # constante de resorte
    b = params['b']      # amortiguamiento
    I = params['I']      # inercia
    # Torques periódicos sincronizados por pares: (1,3) y (2,4)
    tau = [np.sin(2*np.pi*0.5*t),  # Pata 1 (delantera izquierda)
           np.sin(2*np.pi*0.5*t + np.pi),  # Pata 2 (delantera derecha)
           np.sin(2*np.pi*0.5*t),  # Pata 3 (trasera izquierda)
           np.sin(2*np.pi*0.5*t + np.pi)]  # Pata 4 (trasera derecha)

    dydt = np.zeros(8)
    for i in range(4):
        theta = y[2*i]
        omega = y[2*i+1]
        dydt[2*i] = omega
        dydt[2*i+1] = (1/I) * (tau[i] - b*omega - k*theta)
    return dydt

# Parámetros del sistema
params = {
    'k': 2.0,      # constante de resorte
    'b': 0.5,      # amortiguamiento
    'I': 1.0       # inercia
}

# Condiciones iniciales: [θ1, ω1, θ2, ω2, θ3, ω3, θ4, ω4]
y0 = [0.0, 0.0, 0.2, 0.0, -0.2, 0.0, 0.1, 0.0]

# Intervalo de tiempo
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Resolver el sistema de EDOs
sol = solve_ivp(lambda t, y: cuadrupedo_edo(t, y, params), t_span, y0, t_eval=t_eval)

# Graficar resultados
plt.figure(figsize=(10,6))
for i in range(4):
    plt.plot(sol.t, sol.y[2*i], label=f'Pata {i+1} (θ{i+1})')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.title('Movimiento angular de las patas del robot cuadrúpedo')
plt.legend()
plt.grid()
plt.show()

# Graficar ángulos articulares
plt.figure(figsize=(10,6))
for i in range(4):
    plt.plot(sol.t, sol.y[2*i], label=f'Pata {i+1} (θ{i+1})')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo articular (rad)')
plt.title('Ángulos articulares de las patas')
plt.legend()
plt.grid()
plt.show()

# Visualizar fases de marcha (simplificado: fase de apoyo si θ>0, fase de vuelo si θ<=0)
fases = np.array(sol.y[::2])  # Solo ángulos
plt.figure(figsize=(10,4))
for i in range(4):
    fase = np.where(sol.y[2*i]>0, 1, 0)
    plt.plot(sol.t, fase + i*1.2, label=f'Pata {i+1}')
plt.yticks([0,1.2,2.4,3.6], ["Pata 1","Pata 2","Pata 3","Pata 4"])
plt.xlabel('Tiempo (s)')
plt.title('Fases de marcha (1=apoyo, 0=vuelo)')
plt.grid()
plt.legend()
plt.show()

# Trayectoria del centro de masa (simplificado: promedio de posiciones angulares)
com = np.mean([sol.y[0], sol.y[2], sol.y[4], sol.y[6]], axis=0)
plt.figure(figsize=(10,4))
plt.plot(sol.t, com, label='Centro de masa (aprox)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición angular promedio (rad)')
plt.title('Trayectoria aproximada del centro de masa')
plt.grid()
plt.legend()
plt.show()

# --- Optimización del desfase entre pares de patas ---
def evaluar_desfase(phase_diff, params, y0, t_span, t_eval):
    def edo_opt(t, y):
        # Par 1: patas 1 y 3, Par 2: patas 2 y 4
        tau = [np.sin(2*np.pi*0.5*t),
               np.sin(2*np.pi*0.5*t + phase_diff),
               np.sin(2*np.pi*0.5*t),
               np.sin(2*np.pi*0.5*t + phase_diff)]
        k = params['k']
        b = params['b']
        I = params['I']
        dydt = np.zeros(8)
        for i in range(4):
            theta = y[2*i]
            omega = y[2*i+1]
            dydt[2*i] = omega
            dydt[2*i+1] = (1/I) * (tau[i] - b*omega - k*theta)
        return dydt
    sol = solve_ivp(edo_opt, t_span, y0, t_eval=t_eval)
    com = np.mean([sol.y[0], sol.y[2], sol.y[4], sol.y[6]], axis=0)
    desplazamiento = np.ptp(com)  # peak-to-peak (amplitud)
    return desplazamiento, sol, com

# Grid search sobre el desfase
mejor_desfase = 0
mejor_desplazamiento = -np.inf
mejor_sol = None
mejor_com = None
for phase in np.linspace(0, 2*np.pi, 30):
    desplazamiento, sol_tmp, com_tmp = evaluar_desfase(phase, params, y0, t_span, t_eval)
    if desplazamiento > mejor_desplazamiento:
        mejor_desplazamiento = desplazamiento
        mejor_desfase = phase
        mejor_sol = sol_tmp
        mejor_com = com_tmp

print(f"Mejor desfase entre pares de patas: {mejor_desfase:.2f} rad")

# Forzar desfase de π radianes para marcha tipo trote
mejor_desfase = np.pi
print(f"Desfase forzado entre pares de patas (trote): {mejor_desfase:.2f} rad")

# Simulación final con el desfase forzado
def cuadrupedo_edo(t, y, params):
    k = params['k']
    b = params['b']
    I = params['I']
    phase_diff = mejor_desfase
    tau = [np.sin(2*np.pi*0.5*t),
           np.sin(2*np.pi*0.5*t + phase_diff),
           np.sin(2*np.pi*0.5*t),
           np.sin(2*np.pi*0.5*t + phase_diff)]
    dydt = np.zeros(8)
    for i in range(4):
        theta = y[2*i]
        omega = y[2*i+1]
        dydt[2*i] = omega
        dydt[2*i+1] = (1/I) * (tau[i] - b*omega - k*theta)
    return dydt

sol = solve_ivp(lambda t, y: cuadrupedo_edo(t, y, params), t_span, y0, t_eval=t_eval)
com = np.mean([sol.y[0], sol.y[2], sol.y[4], sol.y[6]], axis=0)

# Graficar resultados de la simulación optimizada
plt.figure(figsize=(10,6))
for i in range(4):
    plt.plot(sol.t, sol.y[2*i], label=f'Pata {i+1} (θ{i+1})')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo (rad)')
plt.title('Movimiento angular de las patas del robot cuadrúpedo (simulación optimizada)')
plt.legend()
plt.grid()
plt.show()

# Graficar ángulos articulares de la simulación optimizada
plt.figure(figsize=(10,6))
for i in range(4):
    plt.plot(sol.t, sol.y[2*i], label=f'Pata {i+1} (θ{i+1})')
plt.xlabel('Tiempo (s)')
plt.ylabel('Ángulo articular (rad)')
plt.title('Ángulos articulares de las patas (simulación optimizada)')
plt.legend()
plt.grid()
plt.show()

# Visualizar fases de marcha de la simulación optimizada
fases = np.array(sol.y[::2])  # Solo ángulos
plt.figure(figsize=(10,4))
for i in range(4):
    fase = np.where(sol.y[2*i]>0, 1, 0)
    plt.plot(sol.t, fase + i*1.2, label=f'Pata {i+1}')
plt.yticks([0,1.2,2.4,3.6], ["Pata 1","Pata 2","Pata 3","Pata 4"])
plt.xlabel('Tiempo (s)')
plt.title('Fases de marcha (1=apoyo, 0=vuelo) - Simulación optimizada')
plt.grid()
plt.legend()
plt.show()

# Trayectoria del centro de masa de la simulación optimizada
com = np.mean([sol.y[0], sol.y[2], sol.y[4], sol.y[6]], axis=0)
plt.figure(figsize=(10,4))
plt.plot(sol.t, com, label='Centro de masa (aprox)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Posición angular promedio (rad)')
plt.title('Trayectoria aproximada del centro de masa - Simulación optimizada')
plt.grid()
plt.legend()
plt.show()
