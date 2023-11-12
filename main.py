import numpy as np


class Solver:
    def __init__(self, nx: int, width: float):
        self.nx = nx
        self.width = width
        self.mu_c = 0.599e-3
        self.Ub = self.mu_c / self.width
        self.const = 12 * (self.Ub ** 2) / self.width
        self.dh = self.width / (self.nx - 1)

        self.X = np.linspace(0, self.width, num=self.nx)
        self.U = np.empty(self.nx)
        self.U_AN = np.empty(self.nx)
        self.A = np.zeros(self.nx)
        self.B = np.zeros(self.nx)
        self.C = np.zeros(self.nx)
        self.D = np.zeros(self.nx)

    def tridiagonal_matrix_algorithm(self, A, B, C, D):
        C[0] = C[0] / B[0]
        D[0] = D[0] / B[0]
        B[0] = 1
        for i in range(1, self.nx):
            B[i] = B[i] - A[i] * C[i - 1]
            D[i] = D[i] - A[i] * D[i - 1]
            D[i] = D[i] / B[i]
            C[i] = C[i] / B[i]
            B[i] = 1
        for i in range(self.nx - 2, -1, -1):
            D[i] = D[i] - C[i] * D[i + 1]
        return D

    def isothermal_calc(self):
        for i in range(1, self.nx - 1):
            self.A[i] = self.mu_c / (self.dh * self.dh)
            self.B[i] = -2 * self.mu_c / (self.dh * self.dh)
            self.C[i] = self.mu_c / (self.dh * self.dh)
            self.D[i] = -self.const
        self.B[0] = 1
        self.B[self.nx - 1] = 1

        velocity = self.tridiagonal_matrix_algorithm(self.A, self.B, self.C, self.D)
        velocity_b_A = np.trapz(velocity, self.X) / self.width

        print(f'Analytical mass-weighed average velocity: {self.Ub}; Mass-weighed average velocity: {velocity_b_A}')
        self.U_AN[:] = (-self.const / (2 * self.mu_c) * (self.X[:]) ** 2 +
                        self.const * self.width / (2 * self.mu_c) * self.X[:])

    def nonisothermal_calc(self, mode=False, delta_temp=40):
        def interpolation_polynomial_4th_degree(x):
            return 3.381e-11 * (x ** 4) - 9.304e-9 * (x ** 3) + 9.963e-7 * (x ** 2) - 5.531e-5 * x + 1.781e-3

        T1 = 45 - delta_temp
        T2 = 45 + delta_temp
        Temp = np.empty(self.nx)
        Temp[:] = T1 + (T2 - T1) * self.X[:] / self.width

        MU_I = np.empty(self.nx)

        for i in range(self.nx):
            MU_I[i] = interpolation_polynomial_4th_degree(Temp[i])

        for i in range(1, self.nx - 1):
            if mode:
                MU_PLUS = (MU_I[i + 1] + MU_I[i]) / 2
                MU_MINUS = (MU_I[i] + MU_I[i - 1]) / 2
            else:
                X_PLUS = (self.X[i + 1] + self.X[i]) / 2
                X_MINUS = (self.X[i] + self.X[i - 1]) / 2
                MU_PLUS = interpolation_polynomial_4th_degree(T1 + (T2 - T1) * X_PLUS / self.width)
                MU_MINUS = interpolation_polynomial_4th_degree(T1 + (T2 - T1) * X_MINUS / self.width)

            self.A[i] = MU_MINUS / (self.dh * self.dh)
            self.B[i] = -(MU_MINUS + MU_PLUS) / (self.dh * self.dh)
            self.C[i] = MU_PLUS / (self.dh * self.dh)
            self.D[i] = - self.const

        self.B[0] = 1
        self.B[self.nx - 1] = 1

        velocity = self.tridiagonal_matrix_algorithm(self.A, self.B, self.C, self.D)
        velocity_b_A = np.trapz(velocity, self.X) / self.width

        print(f'Analytical mass-weighed average velocity: {self.Ub}; Mass-weighed average velocity: {velocity_b_A}')
        print(f'Hydraulic resistance coefficient: {self.const * self.width / (500 * velocity_b_A * velocity_b_A)}')


# Solver(nx=21, width=0.002).isothermal_calc()
Solver(nx=21, width=0.002).nonisothermal_calc()
