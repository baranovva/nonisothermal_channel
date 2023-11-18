import matplotlib.pyplot as plt
import numpy as np
from model import Model


class Solver:
    def __init__(self, nx=81, width=0.004):
        self.nx = nx
        self.width = width
        self.mu0 = 0.599e-3
        self.const = 12 * ((self.mu0 / self.width) ** 2) / self.width
        self.dh = self.width / (self.nx - 1)

        self.X = np.linspace(0, self.width, num=self.nx)
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
            self.A[i] = self.mu0 / (self.dh * self.dh)
            self.B[i] = -2 * self.mu0 / (self.dh * self.dh)
            self.C[i] = self.mu0 / (self.dh * self.dh)
            self.D[i] = -self.const
        self.B[0] = 1
        self.B[self.nx - 1] = 1

        velocity = self.tridiagonal_matrix_algorithm(self.A, self.B, self.C, self.D)
        velocity_average = np.trapz(velocity, self.X) / self.width

        print('=' * 20)
        print(f'isothermal, nx = {self.nx}')
        print(f'Analytical average velocity: {self.mu0 / self.width};'
              f' Average velocity: {velocity_average}'
              f', delta % {(1 - velocity_average / (self.mu0 / self.width)) * 100}')

        U_AN = - self.const / (2 * self.mu0) * self.X ** 2 + self.const * self.width / (2 * self.mu0) * self.X

        return self.X, U_AN, velocity

    def nonisothermal_calc(self, is_interpolation_from_nearby=True, delta_temp=40):
        t_1 = 45 - delta_temp
        t_2 = 45 + delta_temp
        temp = t_1 + (t_2 - t_1) * np.linspace(0, self.width, self.nx) / self.width

        model = Model()
        mu = model.predict_model(temp.reshape(-1, 1))

        for i in range(1, self.nx - 1):
            if is_interpolation_from_nearby:
                mu_temp_up = (mu[i + 1] + mu[i]) / 2
                mu_temp_down = (mu[i] + mu[i - 1]) / 2
            else:
                x_temp_up = (self.X[i + 1] + self.X[i]) / 2
                x_temp_down = (self.X[i] + self.X[i - 1]) / 2
                mu_temp_up = model.predict_model((t_1 + (t_2 - t_1) * x_temp_up / self.width).reshape(-1, 1))
                mu_temp_down = model.predict_model((t_1 + (t_2 - t_1) * x_temp_down / self.width).reshape(-1, 1))

            self.A[i] = mu_temp_down / (self.dh * self.dh)
            self.B[i] = -(mu_temp_down + mu_temp_up) / (self.dh * self.dh)
            self.C[i] = mu_temp_up / (self.dh * self.dh)
            self.D[i] = - self.const

        self.B[0] = 1
        self.B[self.nx - 1] = 1

        velocity = self.tridiagonal_matrix_algorithm(self.A, self.B, self.C, self.D)
        velocity_average = np.trapz(velocity, self.X) / self.width

        print('=' * 20)
        print(f'nonisothermal, nx = {self.nx}, delta_T = {delta_temp}')
        print(f'Average velocity: {velocity_average}')
        print(f'Hydraulic resistance coefficient:'
              f' {self.const * self.width / (500 * velocity_average * velocity_average)}')

        return velocity


def plot_decorator(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)
        plt.ylabel('H, m')
        plt.xlabel('Velocity, m/s')
        plt.legend()
        plt.grid(True)
        plt.show()

    return wrapper


@plot_decorator
def plot_isothermal() -> None:
    plt.plot(u11, x11, label='11 points', c='g')
    plt.plot(u21, x21, label='21 points', c='r')
    plt.plot(u41, x41, label='41 points', c='b')
    plt.plot(u81, x81, label='81 points', c='c')
    plt.scatter(u_analytical, x201, label='Analytical', c='m')


@plot_decorator
def plot_nonisothermal() -> None:
    plt.plot(u10, x81, label='10 degrees', c='g')
    plt.plot(u20, x81, label='20 degrees', c='r')
    plt.plot(u30, x81, label='30 degrees', c='b')
    plt.plot(u40, x81, label='40 degrees', c='c')


@plot_decorator
def plot_nonisothermal_compare() -> None:
    plt.plot(u40, x81, label='Interpolation from nearby', c='r')
    plt.plot(u40_from_centers, x81, label='Interpolation from centers', c='g')


x11, _, u11 = Solver(nx=11).isothermal_calc()
x21, _, u21 = Solver(nx=21).isothermal_calc()
x41, _, u41 = Solver(nx=41).isothermal_calc()
x81, _, u81 = Solver().isothermal_calc()
x201, u_analytical, _ = Solver(nx=201).isothermal_calc()
plot_isothermal()

u10 = Solver().nonisothermal_calc(delta_temp=10)
u20 = Solver().nonisothermal_calc(delta_temp=20)
u30 = Solver().nonisothermal_calc(delta_temp=30)
u40 = Solver().nonisothermal_calc()
plot_nonisothermal()

u40_from_centers = Solver().nonisothermal_calc(is_interpolation_from_nearby=False)
plot_nonisothermal_compare()
