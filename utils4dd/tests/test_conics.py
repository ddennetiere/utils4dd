# coding: utf-8
import utils4dd.conics.conics2D as conics
import numpy as np
ellipse = conics.Conics(p=24, q=7, theta=3e-3)
side = "concave"
ellipse.plot(np.linspace(-23,6.9,500), quantity="height", side=side)
ellipse.plot(np.linspace(-23,6.9,500), quantity="slope", side=side)
ellipse.plot(np.linspace(-23,6.9,500), quantity="curvature", side=side)
ellipse.plot(np.linspace(-23,6.9,500), quantity="radius", side=side)
