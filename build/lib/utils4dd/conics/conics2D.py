import numpy as np
import plotly.express as px
import pandas as pd


class PolyConics(object):
    def __init__(self, p, q, theta):
        """
        Polynomial development of a 2D conic described in a cartesian reference frame
        where theta (rad) is the grazing angle af a ray at the center of the conic portion, 
        p (m) is the distance from the conic portion center to the x- focus
        q (m) is the distance from the conic portion center to the x+ focus
        """
        self.p = p
        self.q = q
        self.theta = theta
        self.f = q*p/(q+p)
        self.a2 = np.sin(theta)/4/self.f
        self.a3 = ((p-q)*np.sin(theta)*np.cos(theta))/(8*self.f*p*q)
        self.a4 = (5*np.cos(theta)**2*(p-q)**2+4*p*q)*np.sin(theta)/(64*self.f*p**2*q**2)

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
    

class Conics(object):
    def __init__(self, p, q, theta):
        """
        Exact description of a 2D conic described in a cartesian reference frame as described by Sutter et al. Nucl Instrum Methods Phys Res A 621, 627–636 (2010)
        where theta (rad) is the grazing angle of a ray at the center of the conic portion, 
        p (m) is the distance from the conic portion center to the x- focus
        q (m) is the distance from the conic portion center to the x+ focus
        The conic portion is set so that height = 0 and slope = 0 at x = 0 on the concave side
        """
        self.p = p
        self.q = q
        self.theta = theta
        self.f = q*p/(q+p)
        self.z = None
        self.slope = None
        self.curvature = None
        self.radius = None

    def get_height(self, x, side="concave"):
        p = self.p
        q = self.q
        if side == "concave":
            sign = -1
        else:
            sign = +1
        theta = self.theta
        self.z = (p+q)*np.sin(theta)/((p+q)**2-(p-q)**2*np.sin(theta)**2)
        self.z *= 2*p*q + ((q-p)*np.cos(theta)*x + sign*2*np.sqrt(p*q)*np.sqrt(p*q + (q-p)*np.cos(theta)*x - x**2))
        return self.z

    def get_slope(self, x, side="concave"):
        p = self.p
        q = self.q
        theta = self.theta
        if side == "concave":
            sign = -1
        else:
            sign = +1
        self.slope = (p+q)*np.sin(theta)/((p+q)**2-(p-q)**2*np.sin(theta)**2)
        self.slope *= (q-p)*np.cos(theta) + sign*np.sqrt(p*q)*((q-p)*np.cos(theta) - 2*x)/np.sqrt(p*q + (q-p)*np.cos(theta)*x - x**2)
        return self.slope
    
    def get_curvature(self, x, side="concave"):
        self.curvature = np.diff(self.get_slope(x, side))/np.diff(x)
        self.curvature *= (1 + self.slope[1:]**2)**(3/2)
        return self.curvature
    
    def get_radius(self, x, side="concave"):
        return 1/self.get_curvature(x, side)
    
    def plot(self, x, quantity="height", side="concave"):
        if quantity == "height":
            z = self.get_height(x, side)
        elif quantity == "slope":
            z = self.get_slope(x, side)
        elif quantity == "curvature":
            z = self.get_curvature(x, side)
            x = x[1:]
        elif quantity == "radius":
            z = self.get_radius(x, side)
            x = x[1:]
        fig = px.scatter(x=x, y=z,
                         title=f"{side.capitalize()} ellipse {quantity} (p={self.p}m, q={self.q}m, theta={self.theta}rad)")
        fig.show()