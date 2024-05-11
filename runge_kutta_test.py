import numpy as np
from scipy.integrate import odeint
from runge_kutta import RungeKutta


def test_runge_kutta():
    m = 1.0
    p = 0.1
    k = 1.0
    y0 = 1.0
    dy0 = 0.0
    step = 0.00001
    num_steps = 1000

    rk_solver = RungeKutta(m, p, k, y0, dy0, step, num_steps)
    custom_solution = rk_solver.solve()

    def model(y, x):
        return [y[1], -(p * y[1] + k * y[0]) / m]
    
    t_values = np.linspace(0, step * num_steps, num_steps)
    scipy_solution = odeint(model, [y0, dy0], t_values)

    for i in range(num_steps):
        print(f"y={custom_solution['y'][i]}, dy={custom_solution['dy'][i]}")
        print(f"skipy_y={scipy_solution[i, 0]}, skipy_dy={scipy_solution[i, 1]}")
    
    # Compare results
    assert np.allclose(custom_solution['y'], scipy_solution[:, 0], atol=step)
    assert np.allclose(custom_solution['dy'], scipy_solution[:, 1], atol=step)
