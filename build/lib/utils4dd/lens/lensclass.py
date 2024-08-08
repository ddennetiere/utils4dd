#!- encoding:utf-8
"""
Provides classes for simulating simple lens based optical benches in a matrix formalism from J.P. Perez chap. 5
"""
import numpy as np


def propagation(e, n=1.):
    return np.matrix([[1, e/n], [0, 1]])


class BaseLens(object):
    def __init__(self):
        self.FFD = None
        self.BFD = None
        self.EFL = None
        self.matrix = np.matrix([[np.nan, np.nan], [np.nan, np.nan]])
        self.V = None

    def get_matrix(self, *args):
        pass

    def add(self, other, e, n=1.):
        assert issubclass(other.__class__, BaseLens)
        matrix_prop = propagation(e, n)
        matrix_total = other.matrix*matrix_prop*self.matrix
        equivalent = GenericLens()
        equivalent.generate_lens_from_matrix(matrix_total)
        return equivalent


class GenericLens(BaseLens):
    def generate_lens_from_matrix(self, matrix):
        self.matrix = matrix
        self.EFL = -1./l.matrix[1, 0]
        self.V = 1./self.EFL
        self.BFD = self.matrix[0, 0]*(-1./self.V)
        self.FFD = self.matrix[1, 1]*(1./self.V)


class PerfectLens(BaseLens):
    """
    Classe de lentille parfaite de focale f
    """
    def __init__(self, f):
        super(PerfectLens, self).__init__()
        self.EFL, self.FFD, self.BFD = (f, f, f)
        self.V = 1./f
        self.get_matrix()

    def get_matrix(self):
        self.matrix = np.matrix([[1, 0], [-1./self.EFL, 1]])
        assert np.linalg.det(self.matrix) == 1


class SphericalLens(BaseLens):
    """
    Classe de lentille sphérique de rayon r1 et r2, d'épaisseur e et d'indice n
    """
    def __init__(self, r1, r2, n, e):
        super(SphericalLens, self).__init__()
        self.V = (n-1)*(1./r1-1./r2+(n-1.)*e/(n*r1*r2))
        self.EFL = 1/self.V
        self.FFD = self.EFL*(1+(n-1)*e/(n*r2))
        self.BFD = -self.EFL*(1-(n-1)*e/(n*r1))
        self.get_matrix(r1, r2, n, e)

    def get_matrix(self, r1, r2, n, e):
        matrix_r1 = np.matrix([[1, 0], [-(n-1.)/r1, 1]])
        print (matrix_r1)
        matrix_prop = propagation(e, n)
        print (matrix_prop)
        matrix_r2 = np.matrix([[1, 0], [-(1.-n)/r2, 1]])
        print (matrix_r2)
        self.matrix = matrix_r2*matrix_prop*matrix_r1
        assert np.linalg.det(self.matrix) == 1


def compute_image(point, lens):
    x, d_object, alpha = point
    if abs(x) < np.inf:
        alpha = x/d_object
        x1 = x
        x2 = x
        alpha2 = 0
    else:
        x1 = 1.
        x2 = 2.
        alpha2 = alpha
    ray0 = lens.matrix*propagation(d_object)*np.matrix([[x1], [alpha]])
    ray1 = lens.matrix*propagation(d_object)*np.matrix([[x2], [alpha2]])
    if (ray1 - ray0)[1, 0] != 0:
        d_image = (ray0 - ray1)[0, 0]/(ray1 - ray0)[1, 0]
        point_image = propagation(d_image)*lens.matrix*propagation(d_object)*np.matrix([[x1], [alpha]])
        print ("Gt =", point_image[0, 0]/x)
    else:
        d_image = np.inf
        point_image = np.matrix([[np.inf], [ray0[1, 0]]])
        print ("Ga =", point_image[1, 0]/alpha)
    print ("for x0 =", x, ", x = %f at z = %f" % (point_image[0, 0], d_image))
    return point_image

if __name__ == "__main__":
    l = SphericalLens(0.015, -0.015, 1.5, 2*0.015)
    print ("matrice :")
    print (l.matrix)
    print ("effective focal length :", l.EFL, -1./l.matrix[1, 0])
    print ("back focal length :", l.BFD, l.matrix[0, 0]*(-1./l.V))
    print ("front focal length :", l.FFD, l.matrix[1, 1]*(1./l.V))

    print ("combinaison equivalente :")
    l2 = SphericalLens(0.015, np.inf, 1.5, 0.015)
    l3 = SphericalLens(np.inf, -0.015, 1.5, 0.015)
    ll = l2.add(l3, 0)
    print ("ll", ll.matrix)
    print ("l ", l.matrix)

#   calculs des points de focalisation :
    print ("image à l'infini :")
    lp = PerfectLens(20)
    image = propagation(20)*lp.matrix*propagation(20)*np.matrix([[0], [np.tan(.5)]])
    print (image)

    print ("image en 2f - 2f :")
    image = propagation(40)*lp.matrix*propagation(40)*np.matrix([[2], [np.tan(.5)]])
    print (image)

    print ("image quelconque :")
    X = np.inf
    L = 20.
    Alpha = 0.5
    print (compute_image((X, L, Alpha), lp))

    print ("cas d'un systeme afocal :")
    sys = GenericLens()
    lp2 = PerfectLens(40.)
    sys.generate_lens_from_matrix(lp2.matrix*propagation(60.)*lp.matrix)
    print (compute_image((X, L, Alpha), sys))
