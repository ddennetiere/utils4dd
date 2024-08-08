from fitting.conics import poly_ellipse
import logging
import sys
import numpy as np
from plotly.express import imshow


if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)

    ell = poly_ellipse(22.74, 14, 3.8e-3)
    x = np.linspace(-350e-3/2, 350e-3/2, 500)
    y = np.linspace(-4e-3, 4e-3, 100)
    data = ell.get_2d_cylinder(x=x, y=y)
    res_fit = ell.fit_2d_cylinder(data, [22.74, 18, 3.5e-3])
    logging.info(str(ell))
    fig2 = imshow(res_fit["residual"])
    fig2.show()
    fig = ell.plot_2d_cylinder(x=x, y=y)
    fig.show()
