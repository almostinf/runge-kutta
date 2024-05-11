from runge_kutta import RungeKutta

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
