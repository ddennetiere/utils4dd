import numpy as np
from scipy.interpolate import interp1d as interp
import os
import xraylib as xl
from scipy.constants import physical_constants, h, c, eV, N_A
from pathlib import Path


def loglog_negy_interp1d(x, y, xx):
    """
    1D log-log interpolation supporting negative y values:
    linear interpolation for data before the last negative data;
    log-log for data afterwards
    """
    if np.where(y <= 0)[0].shape[0] > 0:
        idx = np.where(y <= 0)[0][-1] + 1
        x1 = x[:idx + 1]
        y1 = y[:idx + 1]
        x2 = x[idx:]
        y2 = y[idx:]
        if xx[-1] <= x1[-1]:  # all data in linear interpolation region
            return interp(x1, y1)(xx)
        elif xx[0] >= x2[0]:  # all data in log-log interpolation region
            return np.exp(interp(np.log(x2), np.log(y2))(np.log(xx)))
        else:
            idxx = np.where(xx < x1[-1])[0][-1]
            xx1 = xx[:idxx + 1]
            xx2 = xx[idxx + 1:]
            yy1 = interp(x1, y1)(xx1)
            yy2 = np.exp(interp(np.log(x2), np.log(y2))(np.log(xx2)))
            return np.concatenate((yy1, yy2))
    else:  # all data are positive
        return np.exp(interp(np.log(x), np.log(y), bounds_error=False, fill_value='extrapolate')(np.log(xx)))


def get_Ef1f2(Z, datafile='f1f2_EPDL97.dat'):
    Ef1f2 = np.array([], dtype=np.float64)
    path = os.path.join(Path(__file__).resolve().parents[1], "data")
    with open(path + '/' + datafile, 'r') as inp:
        line = inp.readline()
        while line:
            if line.startswith('#S'):
                readZ = int(line.split()[1])
                if readZ == Z:
                    line = inp.readline()  # skip comment lines
                    while line[0] == '#':
                        line = inp.readline()
                    while line[0] != '#':
                        Ef1f2 = np.append(Ef1f2, np.array(line.split(), dtype=np.float64))
                        line = inp.readline()
                    break
            line = inp.readline()
    return Ef1f2.reshape((-1, 3))


def reflectivity_bulk(E, theta, sub_mat, f1f2data='default', f1f2interp='linear'):
    """
    Calculate the reflectivity of a thick mirror made of the specified material.

    Parameters:
    E : float or array-like
        The energy or energies (in eV) at which to calculate the reflectivity.
    theta : float
        The angle of incidence (in radians).
    sub_mat : str
        The chemical symbol of the material (e.g., 'Au' for gold).
    f1f2data : str, optional
        The path to the f1f2 data file, or 'default' to use default data.
    f1f2interp : str, optional
        The interpolation method for f1f2 data. Options are 'linear' (default)
        or 'loglog'.

    Returns:
    numpy.ndarray
        The calculated reflectivity at the specified energy/energies.

    Notes:
    - If `E` is a scalar, the function returns a single reflectivity value.
    - If `E` is an array, the function returns an array of reflectivity values
      corresponding to each energy value.
    - The function uses the xraylib library for various constants and functions.
    - The `f1f2` data is interpolated using the specified method to obtain the
      real (f1) and imaginary (f2) parts of the atomic scattering factors.
    - Reflectivity is calculated using the Fresnel equations for a thick mirror.
    """
    E = np.asarray(E, dtype=np.float64)
    scalar_E = False
    if E.ndim == 0:
        E = E[None]
        scalar_E = True

    sub_Z = xl.SymbolToAtomicNumber(sub_mat)
    if f1f2data == 'default':
        f1f2 = get_Ef1f2(sub_Z)
    else:
        f1f2 = get_Ef1f2(sub_Z, datafile=f1f2data)
    if f1f2interp == 'linear':
        f1 = interp(f1f2[:, 0], f1f2[:, 1])(E)
        f2 = interp(f1f2[:, 0], f1f2[:, 2])(E)
    else:
        f1 = loglog_negy_interp1d(f1f2[:, 0], f1f2[:, 1], E)
        f2 = loglog_negy_interp1d(f1f2[:, 0], f1f2[:, 2], E)

    lamda = h * c / (E * eV) + 0j
    K = 2 * np.pi / lamda
    ah = physical_constants["classical electron radius"][0] * lamda ** 2 / np.pi
    rho = xl.ElementDensity(xl.SymbolToAtomicNumber(sub_mat)) * 1.e6  # g/cm3 --> g/m3
    n_rho = rho / xl.AtomicWeight(xl.SymbolToAtomicNumber(sub_mat)) * N_A  # Atoms per m3

    Chi = ah * n_rho * (-f1 + f2 * 1j)
    K_z0 = K * np.sqrt(np.sin(theta) ** 2 + Chi)
    K_z1 = K * np.sin(theta)
    C1 = np.exp(1j * K_z1 * 1 / 2)
    R_1 = (K_z1 - K_z0) / (K_z1 + K_z0) * C1 ** 2
    R = np.abs(R_1) ** 2

    if scalar_E:
        return np.squeeze(R)
    return R