class RungeKutta:
    def __init__(self, m, p, k, y0, dy0, step, num_steps):
        self.m = m
        self.p = p
        self.k = k
        self.y0 = y0
        self.dy0 = dy0
        self.step = step
        self.num_steps = num_steps

    def derivative2(self, y, dy):
        return - (self.p * dy + self.k * y) / self.m

    # https://math.stackexchange.com/questions/721076/help-with-using-the-runge-kutta-4th-order-method-on-a-system-of-2-first-order-od
    def solve(self):
        res = {'y': [], 'dy': []}

        y = self.y0
        dy = self.dy0

        for _ in range(self.num_steps):
            res['y'].append(y)
            res['dy'].append(dy)

            # k1, k2, k3, k4 derivatives at 4 points
            k1 = self.step * dy
            l1 = self.step * self.derivative2(y, dy)

            k2 = self.step * (dy + 1/2*l1)
            l2 = self.step * self.derivative2(y + 1/2 * k1, dy + 1/2 * l1)

            k3 = self.step * (dy + 1/2*l2)
            l3 = self.step * self.derivative2(y + 1/2 * k2, dy + 1/2 * l2)

            k4 = self.step * (dy + l3)
            l4 = self.step * self.derivative2(y + k3, dy + l3)

            # average increment y
            y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
            # average increment dy
            dy += (l1 + 2 * l2 + 2 * l3 + l4) / 6

        return res
