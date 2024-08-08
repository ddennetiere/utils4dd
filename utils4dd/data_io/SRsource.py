#!/usr/bin/env python
# coding: utf-8

"""Fichier de relecture des diagrammes d'émissions onduleurs calculés avec SR source

 Fichiers binaires ".dat"

 format du fichier :
 - double rank,
 - array de de doubles de taille Rank x3 contenant pour chaque rank : size, min , max
 - Tenseur de doubles de taille size0 x size1 x ... en column major (convention fortran)

 Note : les angles sont en mrad

 Le spectre est décomposé sur le vecteur de Stokes : (I, Q, U, V).
"""

import numpy as np
import pandas as pd
import struct

from scipy.constants import *
from scipy import integrate
from scipy.signal import find_peaks
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr

unit = "$ph.s^{-1}.mrad^{-2}.(0.1\%BP)^{-1}$"


def convert(tensor, distance_to_source=20):
    """
    Convert the intensity tensor to a different unit.

    Parameters:
    tensor : xarray.DataArray
        The intensity tensor to convert.
    distance_to_source : float, optional
        Distance to the source in meters. Default is 20.

    Returns:
    xarray.DataArray
        Converted intensity tensor.
    """
    converted_tensor = tensor.copy(deep=True)
    converted_tensor.values *= eV * 1000 / distance_to_source / distance_to_source
    return converted_tensor


converted_unit = "$J.s^{-1}.mm^{-2}.eV^{-1}$"
converted_unit_html = "J.s<sup>-1</sup>.mm<sup>-2</sup>.eV<sup>-1</sup>"


def read_srsource_file(filename, axes_names=None, log_E=False, verbose=0):
    """
    Read and parse a SR source file.

    Parameters:
    filename : str
        The name of the file to read.
    axes_names : list of str, optional
        Names of the axes. Default is None.
    log_E : bool, optional
        If True, use logarithmic scale for energy axis. Default is False.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns:
    float
        Mean energy.
    xarray.DataArray
        Parsed tensor data.
    """
    struct_rank_fmt = '=d'  # int[5], float, byte[255]
    struct_rank_len = struct.calcsize(struct_rank_fmt)
    struct_rank_unpack = struct.Struct(struct_rank_fmt).unpack_from
    with open(filename, "rb") as filin:
        data = filin.read(struct_rank_len)
        if not data: raise IOError("No rank data found")
        s = struct_rank_unpack(data)
        rank = int(s[0])
        if verbose:
            print("rank =", rank)
        if axes_names is None:
            if rank == 2:
                axes_names = ["X'", "Y'"]
            if rank == 3:
                axes_names = ["X'", "Y'", "E"]

        struct_header_fmt = '=' + '3d' * int(rank)  # int[5], float, byte[255]
        struct_header_len = struct.calcsize(struct_header_fmt)
        struct_header_unpack = struct.Struct(struct_header_fmt).unpack_from

        data = filin.read(struct_header_len)
        if not data: raise IOError("No header data found")
        s = struct_header_unpack(data)
        header = np.array(s)
        sizes = header[::3].astype(int)
        if verbose:
            print("header =", header)
            print(sizes)
        axes = []
        for i, ax in enumerate(axes_names):
            if ax != "E" or not log_E:
                axes.append(np.linspace(header[3 * i + 1], header[3 * i + 2], int(header[3 * i])))
            else:
                axes.append(np.logspace(np.log10(header[3 * i + 1]), np.log10(header[3 * i + 2]), int(header[3 * i])))
        if len(axes) > 2:
            E_H1 = axes[2].mean()
        else:
            E_H1 = np.nan

        struct_tensor_fmt = '=' + '%id' % int(np.prod(sizes))  # int[5], float, byte[255]
        if verbose:
            print(struct_tensor_fmt)
        struct_tensor_len = struct.calcsize(struct_tensor_fmt)
        struct_tensor_unpack = struct.Struct(struct_tensor_fmt).unpack_from

        data = filin.read(struct_tensor_len)
        if not data: raise IOError("No tensor data found")
        s = struct_tensor_unpack(data)
        if verbose:
            print(len(s))
        tensor = np.array(s, dtype=float, copy=True)
        tensor = tensor.reshape(sizes, order='F')
        # print(tensor.min(axis=1))
        tensor = xr.DataArray(tensor, coords=axes, dims=axes_names)
    return E_H1, tensor


def show_mag_field(filename, sep="\t", skiprows=4, title="Champ magnétique"):
    """
    Show the magnetic field from a file.

    Parameters:
    filename : str
        The name of the file to read.
    sep : str, optional
        The delimiter for the file. Default is "\t".
    skiprows : int, optional
        The number of rows to skip at the beginning of the file. Default is 4.
    title : str, optional
        The title for the plot. Default is "Champ magnétique".

    Returns:
    pandas.DataFrame
        DataFrame containing the magnetic field data.
    """
    field = pd.read_csv(filename, sep=sep, names=("S", "Bz"), skiprows=skiprows)
    field['Bz_integre1'] = integrate.cumtrapz(field["Bz"], field["S"], initial=0)
    field['Bz_integre2'] = integrate.cumtrapz(field["Bz_integre1"], field["S"], initial=0)
    layout = {'title': title, "height": 500}
    traces = []

    make_range = lambda arr: [-max(abs(np.min(arr)), np.max(arr)), max(abs(np.min(arr)), np.max(arr))]

    traces.append({"x": field['S'], 'y': field['Bz'], 'name': '$B_z$'})
    traces.append({"x": field['S'], 'y': field['Bz_integre1'], 'name': '$\int B_z \,ds$', 'yaxis': 'y2'})
    traces.append({"x": field['S'], 'y': field['Bz_integre2'], 'name': '$\int\int B_z \,ds^2$', 'yaxis': 'y3'})

    layout['xaxis'] = {'domain': [0.12, 0.95], "title": "Abscisse curviligne S (m)"}
    layout['yaxis1'] = {'title': 'Champ magnétique vertical (T)', 'titlefont': {'color': 'orange'},
                        'tickfont': {'color': 'orange'}, "range": make_range(field['Bz'])}
    layout['yaxis2'] = {'title': 'Première intégrale de champ magnétique vertical (T.m)', 'side': 'right',
                        'overlaying': 'y', 'anchor': 'free',
                        'titlefont': {'color': 'red'}, 'tickfont': {'color': 'red'},
                        "range": make_range(field['Bz_integre1'])}
    layout['yaxis3'] = {'title': 'Seconde intégrale de champ magnétique vertical (T.m²)', 'side': 'right',
                        'overlaying': 'y', 'anchor': 'x',
                        'titlefont': {'color': 'purple'}, 'tickfont': {'color': 'purple'},
                        "range": make_range(field['Bz_integre2'])}

    pio.show({'data': traces, 'layout': layout})
    return field


def show_tensor(filename, distance_to_source=20):
    """
    Show the intensity tensor from a SR source file.

    Parameters:
    filename : str
        The name of the file to read.
    distance_to_source : float, optional
        Distance to the source in meters. Default is 20.

    Returns:
    float
        Mean energy.
    xarray.DataArray
        Intensity tensor.
    """
    E_H1, tensor = read_srsource_file(filename)

    converted_tensor = convert(tensor, distance_to_source=distance_to_source)
    fig = px.imshow(converted_tensor, animation_frame='E',
                    color_continuous_scale='Viridis')  # , zmin=tensor.min(), zmax=tensor.max())
    fig.update_layout(title=f"Total intensity in {converted_unit}", height=800)
    fig.show()
    return E_H1, tensor


def show_absorbed_power(filename, reflectivity, verbose=0):
    """
    Show the absorbed power from a SR source file.

    Parameters:
    filename : str
        The name of the file to read.
    reflectivity : callable
        Function to calculate reflectivity.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns:
    None
    """

    E_H1, tensor = read_srsource_file(filename)

    index_E_H1 = np.argmin(np.abs(tensor.coords["E"].values - E_H1))
    if verbose:
        print(f"H1 @ {E_H1} eV, closest index : {index_E_H1} @ {tensor.coords['E'].values[index_E_H1]} eV")

    converted_tensor = convert(tensor, distance_to_source=20)
    for i in range(converted_tensor.shape[-1]):
        converted_tensor[:, :, i] *= 1 - reflectivity(tensor.coords["E"].values[i])
    ds = converted_tensor.integrate(coord="E")
    fig = px.imshow(ds, color_continuous_scale='Viridis')  # , zmin=tensor.min(), zmax=tensor.max())
    fig.update_layout(title=f"Total power absorbed in {converted_unit}", height=800)
    fig.show()


def export_spectrum(filename):
    """
    Export the spectrum from a SR source file to a text file.

    Parameters:
    filename : str
        The name of the file to read.

    Returns:
    pandas.DataFrame
        DataFrame containing the spectrum data.
    """
    _, spectrum = read_srsource_file(filename, log_E=True, axes_names=["Stokes", "E"])
    spectrum.coords["Stokes"] = ["I", "Q", "U", "V"]
    spectrum = spectrum.T.to_pandas()
    spectrum.to_csv(filename.split(".")[0] + ".txt", sep="\t")
    return spectrum


def show_spectrum(filename, logscale=True):
    """
    Show the spectrum from a SR source file.

    Parameters:
    filename : str
        The name of the file to read.
    logscale : bool, optional
        If True, use logarithmic scale for the y-axis. Default is True.

    Returns:
    xarray.DataArray
        Spectrum data.
    """
    _, spectrum = read_srsource_file(filename, log_E=True, axes_names=["Stokes", "E"])
    spectrum.coords["Stokes"] = ["I", "Q", "U", "V"]
    fig = px.line(spectrum.T.to_pandas(), y=["I", "Q", "U", "V"], height=500, log_y=logscale)
    fig.update_yaxes(title_text="Flux (ph/s)", secondary_y=False)
    fig.update_xaxes(title_text="Energie (eV)")
    fig.show()
    return spectrum


def show_trajectories(filename, axes=("X", "Y")):
    """
    Show the electron trajectories from a file.

    Parameters:
    filename : str
        The name of the file to read.
    axes : tuple of str, optional
        The axes to plot. Default is ("X", "Y").

    Returns:
    pandas.DataFrame
        DataFrame containing the trajectory data.
    """
    traj = pd.read_csv(filename, sep="\t")
    fig = px.line(traj, x='s', y=list(axes), height=500, title="Electron trajectories")
    fig.update_yaxes(title_text="Distance to 0 trajectory (m)", secondary_y=False)
    fig.update_xaxes(title_text="Undulator length (m)")
    fig.show()
    return traj


def analyze_harmonics(spectrum, E_H1, verbose=0):
    """
    Analyze the harmonics in a spectrum.

    Parameters:
    spectrum : xarray.DataArray
        Spectrum data.
    E_H1 : float
        Fundamental harmonic energy.
    verbose : int, optional
        Verbosity level. Default is 0.

    Returns:
    None
    """
    peaks, properties = find_peaks(spectrum.loc['I', :].values, distance=E_H1 / 20,
                                   threshold=spectrum.loc['I', :].values.max() / 10000)

    # fig = figure(y_axis_type="log")
    # fig.line(spectrum["E"], spectrum["I"], legend_label="I")
    # fig.scatter(spectrum["E"][peaks], spectrum["I"][peaks], marker="x", color="orange", size=20)
    # show(fig)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spectrum.coords["E"], y=spectrum.loc['I', :], name="I", mode="lines"))
    fig.add_trace(go.Scatter(x=spectrum.coords["E"][peaks], y=spectrum.loc['I', :].values[peaks], name="Harmonics",
                             mode="markers"))
    fig.update_yaxes(type="log")
    fig.update_layout(title="Harmonics detection")
    fig.show()

    if verbose:
        print(properties)

    d_harm = np.diff(spectrum.coords["E"][peaks]).mean()
    E_H1 = np.array(spectrum.coords["E"][peaks])[0]
    Ptot = 0
    for peak in peaks:
        Eharm = float(spectrum.coords["E"][peak])
        index_harm = (spectrum.coords["E"] > (Eharm - d_harm / 2)) & (spectrum.coords["E"] < (Eharm + d_harm / 2))
        Pharm = np.trapz(spectrum.loc['I', :].values[index_harm] * eV, axis=0, x=spectrum.coords["E"][index_harm])
        print(f"Puissance dans l'harmonique {int(np.rint(Eharm / E_H1))} @ {Eharm:.2f} eV: {Pharm:.2f} W")
        Ptot += Pharm
    print(f"Puissance sur l'axe totale dans les harmoniques : {Ptot:.2f} W")

