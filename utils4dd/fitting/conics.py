import numpy as np
from plotly.express import line, scatter, imshow
import xarray
import logging
from scipy.optimize import minimize
import pandas as pd
logger = logging.getLogger(__name__)


class poly_ellipse(object):
    """
    class describing an ellipse portion which focii are at (-p*cos(theta), p*sin(theta)) and
    (q*cos(theta), q*sin(theta)).
    The ellipse is approximated by a 4 degree polynomial in height
    """
    def __init__(self, p, q, theta, tilt_x=0, tilt_y=0, offset=0):
        """
        Defines the ellipse by its coefficient
        """
        self.a4 = None
        self.a3 = None
        self.a2 = None
        self.f = None
        self.p = p
        self.q = q
        self.theta = theta
        self.tilt_x = tilt_x
        self.tilt_y = tilt_y
        self.offset = offset
        self.set_poly()

    def __repr__(self):
        return f"Polynomial approximated ellipse\n \tP = {self.p} m, q = {self.q} m, theta = {self.theta} rad, \n\t" \
               f"tilt_x = {self.tilt_x} rad, tilt_y = {self.tilt_y} rad, offset = {self.offset} m"

    def set_poly(self):
        self.f = self.q * self.p / (self.q + self.p)
        self.a2 = np.sin(self.theta) / 4 / self.f
        self.a3 = ((self.p - self.q) * np.sin(self.theta) * np.cos(self.theta)) / (8 * self.f * self.p * self.q)
        self.a4 = (5 * np.cos(self.theta) ** 2 * (self.p - self.q) ** 2 + 4 * self.p * self.q) * np.sin(self.theta) / (64 * self.f * self.p ** 2 * self.q ** 2)

    def get_1d_height(self, x=None, half_span=None):
        if x is None:
            x = np.linspace(-half_span, half_span, 100)
        z = self.offset + np.tan(self.tilt_x) * x + self.a2 * x ** 2 + self.a3 * x ** 3 + self.a4 * x ** 4
        return x, z

    def plot_1d(self, x=None, half_span=None):
        x, z = self.get_1d_height(x, half_span)
        fig = scatter(x=x, y=z,
                      title=f"Ellipse (p={self.p}m, q={self.q}m, theta={self.theta}rad) <br>"
                            f"a2={self.a2:.3e}, a3={self.a3:.3e}, a4={self.a4:.3e}")
        return fig

    def get_2d_cylinder(self, x=None, y=None, half_span_x=None, half_span_y=None):
        if x is None:
            x = np.linspace(-half_span_x, half_span_x, 100)
        if y is None:
            y = np.linspace(-half_span_y, half_span_y, 100)
        height = np.repeat(self.offset + self.a2 * x ** 2 + self.a3 * x ** 3 + self.a4 * x ** 4, y.shape[0])
        height = height.reshape((x.shape[0], y.shape[0]))
        xx, yy = np.meshgrid(y, x)
        height += np.tan(self.tilt_x) * xx + np.tan(self.tilt_y) * yy
        return xarray.DataArray(height.T, (("y", y), ("x", x)))

    def plot_2d_cylinder(self, x=None, y=None, half_span_x=None, half_span_y=None):
        data = self.get_2d_cylinder(x, y, half_span_x, half_span_y)
        fig = imshow(data)
        return fig

    def fit_2d_cylinder(self, data_array, x0, x=None, y=None):
        if isinstance(data_array, np.ndarray):
            data_array = xarray.DataArray(data_array, (("y", y), ("x", x)))
        x = np.array(data_array.x)
        y = np.array(data_array.y)

        def fun(x0):
            return np.sqrt(((data_array - poly_ellipse(*x0).get_2d_cylinder(x=x, y=y))**2).sum())
        res = minimize(fun, x0)
        logger.info(f"Result of minimizer : [{res.success}], best solution :{res.x}")
        logger.debug(str(res))
        self.p, self.q, self.theta = res.x
        self.set_poly()
        return {"success": res.success, "solution": res.x,
                "residual": data_array-self.get_2d_cylinder(x=x, y=y)}
