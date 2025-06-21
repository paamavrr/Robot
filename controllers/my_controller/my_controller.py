from controller import Robot
import math
import csv

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Orden: LH, RF, LF, RH (para coincidir con tus gráficas)
motor_names = ['LH_motor', 'RF_motor', 'LF_motor', 'RH_motor']
touch_names = ['LH_touch', 'RF_touch', 'LF_touch', 'RH_touch']

motors = [robot.getDevice(name) for name in motor_names]
touch_sensors = [robot.getDevice(name) for name in touch_names]
for ts in touch_sensors:
    if ts is not None:
        ts.enable(timestep)

# Parámetros CPG (idénticos a tus gráficas)
A = 0.745 # Amplitud de oscilación
f = 0.222 # Frecuencia de oscilación (ajustada para un ciclo de 5 segundos)
omega = 2 * math.pi * f

# Desfases para trote diagonal: LH y RF en fase, LF y RH en fase opuesta
phase_offsets = [0, 0, math.pi, math.pi]  # LH, RF, LF, RH

t = 0.0
last_angles = [0.0] * 4
smooth_factor = 0.122

# Parámetro para asimetría (más tiempo empujando)
stance_ratio = 0.8  # 65% del ciclo en apoyo
offset = 0.1322         # Offset para evitar verticalidad pura

while robot.step(timestep) != -1:
    t += timestep / 1000.0

    for i, motor in enumerate(motors):
        phase = omega * t + phase_offsets[i]
        phase_mod = (phase % (2 * math.pi)) / (2 * math.pi)

        # Ciclo asimétrico: más tiempo empujando (stance), menos en el aire (swing)
        if phase_mod < stance_ratio:
            progress = phase_mod / stance_ratio
            target_angle = -A * (1 - progress) + offset
        else:
            swing_phase = (phase_mod - stance_ratio) / (1 - stance_ratio)
            target_angle = A * math.sin(swing_phase * math.pi) - offset

        # Suavizado para evitar movimientos bruscos
        angle = last_angles[i] + smooth_factor * (target_angle - last_angles[i])
        last_angles[i] = angle

        # Limitar al rango físico seguro
        angle = max(min(angle, 1.0), -1.0)

        motor.setPosition(angle)

    # (Opcional) Imprimir estado cada segundo
    if int(t) % 1 == 0 and t % 1 < 0.1:
        contacts = [ts.getValue() > 0.01 if ts else False for ts in touch_sensors]
        print(f"T: {t:.1f}s | Contactos: LH:{contacts[0]} RF:{contacts[1]} LF:{contacts[2]} RH:{contacts[3]}")