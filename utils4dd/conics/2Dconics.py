import numpy as np
import plotly.express as px
import pandas as pd


class PolyConics(object):
    def __init__(self, p, q, theta):
        self.p = p
        self.q = q
        self.theta = theta
        self.f = q*p/(q+p)
        self.a2 = np.sin(theta)/4/f
        self.a3 = ((p-q)*np.sin(theta)*np.cos(theta))/(8*f*p*q)
        self.a4 = (5*np.cos(theta)**2*(p-q)**2+4*p*q)*np.sin(theta)/(64*f*p**2*q**2)

    def get_height(self, x):
        return self.a2*x**2+self.a3*x**3+self.a4*x**4

    def get_residual(self):
        pass

    def plot(self, x):
        z = self.get_height(x)
        fig = px.scatter(x=x, y=z,
                         title=f"Ellipse (p={self.p}m, q={self.q}m, theta={self.theta}rad) <br>"
                               f"a2={self.a2:.3e}, a3={self.a3:.3e}, a4={self.a4:.3e}")
        fig.show()

    def get_dataframe(self, x):
        return pd.DataFrame({"x": x, "z": self.get_height(x)})