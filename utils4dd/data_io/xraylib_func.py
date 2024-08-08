import re
import numpy as np
from scipy.interpolate import interp1d as interp
import os
import xraylib
from scipy.constants import physical_constants, h, c, eV, N_A, degree
from pathlib import Path
from collections import Counter

compound_densities = {"B4C": 2.52, "SiC": 3.217}


def parse_formula(formula):
    """
    Parse a chemical formula and return a dictionary with the count of each element.

    Parameters:
    formula : str
        The chemical formula to parse (e.g., 'H2O', 'C6H12O6').

    Returns:
    dict
        A dictionary with elements as keys and their counts as values.
    """
    # Regex to match elements and their counts
    element_pattern = re.compile(r'([A-Z][a-z]*)(\d*)')

    elements = Counter()
    for (element, count) in element_pattern.findall(formula):
        if count == '':
            count = 1
        else:
            count = int(count)
        elements[element] += count * xraylib.AtomicWeight(xraylib.SymbolToAtomicNumber(element))

    return elements


def element_ratios(formula):
    """
    Calculate the ratios of each element in a chemical formula.

    Parameters:
    formula : str
        The chemical formula (e.g., 'H2O', 'C6H12O6').

    Returns:
    dict
        A dictionary with elements as keys and their ratios as values.
    """
    # Parse the formula to get counts of each element
    elements = parse_formula(formula)

    # Calculate the total count of atoms in the formula
    total_atoms = sum(elements.values())

    # Calculate the ratio of each element
    ratios = {element: count / total_atoms for element, count in elements.items()}

    return ratios


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


def get_Ef1f2(Z, datafile="f1f2CXRO.dat"):
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
                        if line.strip() == "":
                            break
                    break
            line = inp.readline()
    return Ef1f2.reshape((-1, 3))


def reflectivity_bulk(E, theta, sub_mat, f1f2data='default', f1f2interp='linear', mesh=False, density=-1.):
    """
    Calculate the reflectivity of a thick mirror made of the specified material.

    Parameters:
    E : float or array-like
        The energy or energies (in eV) at which to calculate the reflectivity.
    theta : float or array-like
        The angle of incidence (in radians). Must be scalar or array of same dimension as E if mesh = False.
    sub_mat : str
        The chemical symbol of the material (e.g., 'Au' for gold).
    f1f2data : str, optional
        The path to the f1f2 data file, or 'default' to use default data.
    f1f2interp : str, optional
        The interpolation method for f1f2 data. Options are 'linear' (default)
        or 'loglog'.
    mesh : bool
        If True, returns a map of reflectivity over E and theta.

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
    compound = element_ratios(sub_mat)
    E = np.asarray(E, dtype=np.float64)
    scalar_E = False
    if E.ndim == 0:
        E = E[None]
        scalar_E = True

    if mesh:
        assert isinstance(theta, np.ndarray)
        E, theta = np.meshgrid(E, theta)
    elif isinstance(theta, np.ndarray) and not scalar_E:
        assert E.shape == theta.shape

    f1 = np.zeros(E.shape)
    f2 = np.zeros(E.shape)
    for element, mass_fraction in compound.items():
        sub_Z = xraylib.SymbolToAtomicNumber(element)
        if f1f2data == 'default':
            f1f2 = get_Ef1f2(sub_Z)
        else:
            f1f2 = get_Ef1f2(sub_Z, datafile=f1f2data)
        if f1f2interp == 'linear':
            f1 += interp(f1f2[:, 0], f1f2[:, 1])(E) * mass_fraction
            f2 += interp(f1f2[:, 0], f1f2[:, 2])(E) * mass_fraction
        else:
            f1 += loglog_negy_interp1d(f1f2[:, 0], f1f2[:, 1], E) * mass_fraction
            f2 += loglog_negy_interp1d(f1f2[:, 0], f1f2[:, 2], E) * mass_fraction

    lamda = h * c / (E * eV) + 0j
    K = 2 * np.pi / lamda
    ah = physical_constants["classical electron radius"][0] * lamda ** 2 / np.pi
    n_rho = 0
    for element, mass_fraction in compound.items():
        sub_Z = xraylib.SymbolToAtomicNumber(element)
        if density == -1:
            rho = xraylib.ElementDensity(sub_Z) * 1.e6  # g/cm3 --> g/m3
        else:
            rho = density * 1e6
        n_rho += rho * mass_fraction / xraylib.AtomicWeight(sub_Z) * N_A  # Atoms per m3

    Chi = ah * n_rho * (-f1 + f2 * 1j)
    K_z0 = K * np.sqrt(np.sin(theta) ** 2 + Chi)
    K_z1 = K * np.sin(theta)
    C1 = np.exp(1j * K_z1 * 1 / 2)
    R_1 = (K_z1 - K_z0) / (K_z1 + K_z0) * C1 ** 2
    R = np.abs(R_1) ** 2

    if scalar_E:
        return np.squeeze(R)
    return R


def transmission_bulk(material, energy, thickness=0.2, density=-1, f1f2data='default', f1f2interp='linear'):
    """
    Calculate the transmission of a filter made of the specified material.

    Parameters:
    energy : float or array-like
        The energy or energies (in eV) at which to calculate the reflectivity.
    material : str
        The chemical symbol of the material (e.g., 'Au' for gold).
    f1f2data : str, optional
        The path to the f1f2 data file, or 'default' to use default data.
    f1f2interp : str, optional
        The interpolation method for f1f2 data. Options are 'linear' (default)
        or 'loglog'.
    density : float
        density of the material. If -1, default density of each component of the compound is assumed

    Returns:
    numpy.ndarray
        The calculated transmission at the specified energy/energies.

    Notes:
    - If `E` is a scalar, the function returns a single reflectivity value.
    - If `E` is an array, the function returns an array of reflectivity values
      corresponding to each energy value.
    - The function uses the xraylib library for various constants and functions.
    - The `f1f2` data is interpolated using the specified method to obtain the
      real (f1) and imaginary (f2) parts of the atomic scattering factors.
    - Transmission is calculated using the  equations from X-ray data booklet.
    """
    compound = element_ratios(material)
    energy = np.asarray(energy, dtype=np.float64)
    scalar_E = False
    if energy.ndim == 0:
        energy = energy[None]
        scalar_E = True

    lamda = h * c / (energy * eV) + 0j
    r_e = physical_constants["classical electron radius"][0]
    f1 = np.zeros(energy.shape)
    f2 = np.zeros(energy.shape)
    for element, mass_fraction in compound.items():
        sub_Z = xraylib.SymbolToAtomicNumber(element)
        if f1f2data == 'default':
            f1f2 = get_Ef1f2(sub_Z)
        else:
            f1f2 = get_Ef1f2(sub_Z, datafile=f1f2data)
        if f1f2interp == 'linear':
            f1 += interp(f1f2[:, 0], f1f2[:, 1])(energy) * mass_fraction
            f2 += interp(f1f2[:, 0], f1f2[:, 2])(energy) * mass_fraction
        else:
            f1 += loglog_negy_interp1d(f1f2[:, 0], f1f2[:, 1], energy) * mass_fraction
            f2 += loglog_negy_interp1d(f1f2[:, 0], f1f2[:, 2], energy) * mass_fraction
    n_rho = 0
    for element, mass_fraction in compound.items():
        sub_Z = xraylib.SymbolToAtomicNumber(element)
        if density == -1:
            rho = xraylib.ElementDensity(sub_Z) * 1.e6  # g/cm3 --> g/m3
        else:
            rho = density * 1e6
        n_rho += rho * mass_fraction / xraylib.AtomicWeight(sub_Z) * N_A  # Atoms per m3
    l_abs = np.real(1/(2*n_rho*r_e*lamda*f2))
    if scalar_E:
        return np.squeeze(0.5**(thickness/l_abs))
    return 0.5**(thickness/l_abs)


def make_CXRO_f1f2_file(dirname=None):
    """
    Compiles data from CXRO .nff files in a global f1f2 file compatible with the functions here

    Parameters:
    dirname : str
        directory holding the CXRO data, default is ../data
    Returns
        None
    """
    if dirname is None:
        dirname = os.path.join(Path(__file__).resolve().parents[1], "data")
    filename_out = os.path.join(Path(__file__).resolve().parents[1], "data/f1f2CXRO.dat")
    with open(filename_out, "w") as fout:
        fout.write("# File generated fom CXRO data\n")
        fout.write("# found in https://henke.lbl.gov/optical_constants/asf.html\n")
        for dirpath, dirnames, filenames in os.walk(dirname):
            for filename in filenames:
                element, file_extension = os.path.splitext(filename)
                if file_extension == ".nff":
                    element_z = xraylib.SymbolToAtomicNumber(element.capitalize())
                    fout.write(f"#S {element_z} {element.capitalize()}\n")
                    fout.write(f"#UF1ADD 0.0\n")
                    fout.write(f"#N 3\n")
                    fout.write(f"#L PhotonEnergy[eV]  f1  f2\n")
                    with open(os.path.join(dirpath,filename), 'r') as fin:
                        for line in fin:
                            if line.strip() != "" and line[0] != "#":
                                try:
                                    _ = float(line.split()[0])
                                    energy, f1, f2 = line.split()
                                    fout.write(f"{float(energy):14.4f}{float(f1):14.6f}{float(f2):14.6f}\n")
                                except ValueError:
                                    pass


if __name__ == "__main__":
    import plotly.graph_objects as go

    make_CXRO_f1f2_file()

    # Create the figure
    fig = go.Figure()
    energy = np.linspace(30, 150, 1000)
    # Add the first trace
    fig.add_trace(go.Scatter(x=energy,
                             y=reflectivity_bulk(energy, 2 * degree, "Si", density=-1,
                                                 f1f2data="f1f2CXRO.dat"),
                             mode='lines', name='CXRO'))

    # Add the second trace
    fig.add_trace(go.Scatter(x=energy,
                             y=reflectivity_bulk(energy, 2 * degree, "Si", density=-1,
                                                 f1f2data="f1f2_EPDL97.dat"),
                             mode='lines', name='EPDL'))

    # Update layout
    fig.update_layout(title='Si f1f2 data compared', xaxis_title='Energy (eV)', yaxis_title='Reflectivity')

    # Show the plot
    fig.show()

    energy = np.linspace(30, 1500, 1000)
    fig2 = go.Figure()
    # Add the first trace
    fig2.add_trace(go.Scatter(x=energy,
                              y=transmission_bulk("Al", energy, thickness=0.2e-6, density=-1,
                                                  f1f2data="f1f2CXRO.dat"),
                              mode='lines', name='CXRO'))
    fig2.update_layout(title='Transmission of Al', xaxis_title='Energy (eV)', yaxis_title='Transmission')
    fig2.show()