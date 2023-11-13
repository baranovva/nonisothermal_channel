from pandas import read_csv
import matplotlib.pyplot as plt
from numpy import linspace, ndarray, array
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class Model:
    def __init__(self, filepath_or_buffer='mu.txt'):
        data = read_csv(filepath_or_buffer=filepath_or_buffer, header=None)
        self.futures = data.iloc[:16]
        self.targets = data.iloc[16:]

        self.model = self.fitted_model()

    def fitted_model(self) -> object:
        poly = PolynomialFeatures(degree=4)
        X_poly = poly.fit_transform(self.futures)

        base_model = LinearRegression()
        model = base_model.fit(X_poly, self.targets)

        return model

    def predict_model(self, value) -> object:
        if not isinstance(value, ndarray):
            value = array(value).reshape(-1, 1)

        value = PolynomialFeatures(degree=4).fit_transform(value)

        return self.model.predict(value)

    def plot(self) -> None:
        array = linspace(0, 100, 50).reshape(-1, 1)
        array_poly = PolynomialFeatures(degree=4).fit_transform(array)

        plt.plot(array, self.model.predict(array_poly), label='Polynomial regression', c='g')
        plt.scatter(self.futures, self.targets, label='Initial data', c='r')
        plt.ylabel('Mu')
        plt.xlabel('T, C')
        plt.legend()
        plt.grid(True)
        plt.show()


Model().plot()
