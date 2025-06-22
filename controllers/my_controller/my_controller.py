from controller import Robot
import math
import csv

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Orden: LH, RF, LF, RH (para coincidir con tus gráficas y CSV)
motor_names = ['LH_motor', 'RF_motor', 'LF_motor', 'RH_motor']
touch_names = ['LH_touch', 'RF_touch', 'LF_touch', 'RH_touch']

motors = [robot.getDevice(name) for name in motor_names]
touch_sensors = [robot.getDevice(name) for name in touch_names]
for ts in touch_sensors:
    if ts is not None:
        ts.enable(timestep)

# --- Leer ángulos desde CSV ---
import pandas as pd
csv_file = "hopf_angles.csv"  # Cambia a "cpg_angles.csv" si quieres probar el otro
angles_df = pd.read_csv(csv_file)
angles = angles_df.values  # shape: (num_steps, 4)
num_steps = angles.shape[0]

# --- Reproducir los ángulos en el robot ---
step_idx = 0
while robot.step(timestep) != -1:
    # Si llegamos al final del CSV, repetimos el ciclo
    if step_idx >= num_steps:
        step_idx = 0

    for i, motor in enumerate(motors):
        angle = float(angles[step_idx, i])
        # Limitar al rango físico seguro
        angle = max(min(angle, 1.0), -1.0)
        motor.setPosition(angle)

    # (Opcional) Imprimir estado cada segundo
    if step_idx % int(1.0 / (timestep / 1000.0)) == 0:
        contacts = [ts.getValue() > 0.01 if ts else False for ts in touch_sensors]
        print(f"Step: {step_idx} | Contactos: LH:{contacts[0]} RF:{contacts[1]} LF:{contacts[2]} RH:{contacts[3]}")

    step_idx += 1


