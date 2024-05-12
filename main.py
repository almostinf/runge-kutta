from runge_kutta import RungeKutta
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

m = float(input("m: "))
p = float(input("p: "))
k = float(input("k: "))
y0 = float(input("y0: "))
dy0 = float(input("dy0: "))
step = float(input("step: "))
num_steps = int(input("number of steps: "))

rk_solver = RungeKutta(m, p, k, y0, dy0, step, num_steps)
result = rk_solver.solve()

print("Result:")
for i in range(num_steps):
    print(f"step={i}, y={result['y'][i]}, dy={result['dy'][i]}")

def model(y, x):
        return [y[1], -(p * y[1] + k * y[0]) / m]

t_values = np.linspace(0, step * num_steps, num_steps)
scipy_solution = odeint(model, [y0, dy0], t_values)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

axs[0].plot(t_values, result['y'], label='Custom Solution', color='blue')
axs[0].plot(t_values, scipy_solution[:, 0], label='Scipy Solution', color='red')
axs[0].set_title('Comparison of Custom Runge-Kutta and Scipy Solutions for y')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_values, result['dy'], label='Custom Solution', color='blue')
axs[1].plot(t_values, scipy_solution[:, 1], label='Scipy Solution', color='red')
axs[1].set_title('Comparison of Custom Runge-Kutta and Scipy Solutions for dy')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('dy')
axs[1].legend()
axs[1].grid(True)

plt.show()
