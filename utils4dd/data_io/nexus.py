import h5py
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from nexusformat.nexus import NXdata, NXentry, NXfield, nxopen, NXgroup, NXlink
import warnings
import xarray as xr
from scipy.interpolate import LinearNDInterpolator



class NexusFile(object):
    """NexusFile class for opening SOLEIL Nexus files and storing data.

    This class opens SOLEIL's Nexus files and stores tabular data in a pandas
    ``DataFrame`` (available as the ``df`` attribute) and provides a
    synchronized :mod:`xarray` view via the ``xarray`` property. Image data
    (3D arrays) are collected in the ``images`` attribute.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the Nexus file to open.
    data_root : str, optional
        Root path inside the file where scan data are located (default
        ``'scan/scan_data'``).
    x : str, optional
        Column name to set as index on the resulting DataFrame if provided.
    """
    def __init__(self, filename, data_root='scan/scan_data', x=None):
        data = {}
        self.images = {}
        with h5py.File(filename, 'r') as fin:
            try:
                group = fin[data_root]
            except KeyError:
                raise ValueError(f"Data root '{data_root}' not found in the file.")

            for key, node in group.items():
                if isinstance(node, h5py.Dataset):
                    node_data = np.array(node)
                    if node.attrs.get('long_name', key) is None:
                        name = key
                    else:
                        if isinstance(node.attrs.get('long_name', key), bytes):
                            name = node.attrs.get('long_name', key).decode()
                        else:
                            name = node.attrs.get('long_name', key)
                    if node_data.ndim == 1:
                        data[name] = node_data
                    elif node_data.ndim == 3:
                        self.images[name] = node_data

        self.homogeneous_data = False
        try:
            self.df = pd.DataFrame(data)
            if x is not None:
                self.df = self.df.set_index(x)
            self.homogeneous_data = True
        except ValueError:
            self.df = data 
            warnings.warn("Data could not be formed into a homogeneous DataFrame")
        self._xarray = None

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        if not self.homogeneous_data:
            return self.__repr__()
        return self.df._repr_html_()

    @property
    def xarray(self):
        assert self.homogeneous_data, "DataFrame is not homogeneous; cannot convert to xarray."
        return self.df.to_xarray()

    @xarray.setter
    def xarray(self, value):
        self.df = pd.DataFrame(value.to_pandas())

    def __getitem__(self, item):
        return self.df[item]

    def __setitem__(self, key, value):
        self.df[key] = value

    def plot(self, axtype="linlin"):
        assert self.homogeneous_data, "DataFrame is not homogeneous; cannot convert to xarray."
        fig = px.line(self.xarray)
        if axtype[3:] == 'log':
            fig.update_yaxes(type="log")
        if axtype[:3] == 'log':
            fig.update_xaxes(type="log")
        return fig


class PycarpemNexusFile(NexusFile):
    """Specialized NexusFile class for Pycarpem Nexus files.

    This class extends ``NexusFile`` to handle Pycarpem-specific Nexus
    files where data are stored under ``'entry/data'``.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the Pycarpem Nexus file to open.
    x : str, optional
        Column name to set as index on the resulting DataFrame if provided.
    """
    def __init__(self, filename, data_root='entry/data/reflectivity_data'):
        super().__init__(filename, data_root=data_root)

    def get_reflectivity(self, polarisation=0):
        """get_reflectivity Interpolates the sheared reflectivity data onto a regular grid.


        Parameters
        ----------
        polarisation : int, optional
            _description_, by default 0

        Returns
        -------
        xarray.DataArray
            dataarray with dimensions (E, theta) on a regular grid
        """
        Z = self.images["reflectivity"][polarisation]  # 2D array (E, theta)
        x = self.df["incidence"]
        y = self.df["photon_energy"]
        offsets = self.df["incidence_offset"]
        n, m = Z.shape # ligne = E = y, colonne = theta = x

        # --- 1) Build original sheared coordinates ---
        X, Y = np.meshgrid(x, y, indexing="xy")      # (n,m)
        X = X + offsets[:, None]                     # apply column offset
        # Flatten for griddata
        # pts = np.column_stack([X.ravel(), Y.ravel()])
        pts = []
        vals = []
        for i in range(m):
            for j in range(n):
                pts.append( (X[j,i], Y[j,i]) )
                vals.append( Z[j,i] )
        # vals = Z.ravel()

        # --- 2) Build new regular 1D target coordinates ---
        theta_points = np.linspace(X.min(), X.max(), n)
        E_points = y.copy()

        TH, EE = np.meshgrid(theta_points, E_points, indexing="xy")

        # --- 3) Interpolate ---
        interp = LinearNDInterpolator(pts, vals, fill_value=np.nan)
        Z_new = interp(TH, EE)

        # --- 4) Package into xarray ---
        da = xr.DataArray(
            Z_new,
            dims=("E", "theta"),
            coords={"theta": theta_points, "E": E_points},
            name="Z"
        )

        return da

def write_nexus_file(filename, data_dict, verbose=False):
    """Write a Nexus file from a structured (possibly nested) dictionary.

    This function supports nested dictionaries so you can group measured
    fields by instrument or actuator. Each top-level key in ``data_dict``
    becomes a group under the file's ``entry`` (e.g. ``/entry/Scan1``).

    Group specification rules
    -------------------------
    - Sub-dictionaries become nested NXgroups (for grouping instruments,
      actuators, etc.).
    - Values that are arrays/scalars or dicts with a ``'data'`` key become
      NXfield nodes. The dict form may include optional ``'name'`` and
      ``'units'`` keys.
    - Per-group metadata is provided with the ``'metadata'`` dict; each
      metadata item is written as an NXfield under ``.../metadata/``.
    - To mark the main dataset for a group, set ``'default'`` to a string
      path that points to the field (relative to the group's path, e.g.
      ``'instrument1/slope'``) or to an absolute path starting with ``/``.
    - To set axes for that main dataset, provide ``'default_axes'`` as a
      list of relative (or absolute) paths to the axis fields.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output Nexus filename (will be created/overwritten).
    data_dict : dict
        Mapping of group_name -> group_spec. ``group_spec`` may be nested
        to represent instruments/actuators. Use ``'default'`` and
        ``'default_axes'`` to link the main data to axes. Fields may be
        provided as raw numpy arrays/scalars or as dictionaries
        ``{'data': ..., 'name': ..., 'units': ...}``.
    verbose : bool, optional
        If True, prints the path of the written file.

    Returns
    -------
    pathlib.Path
        Path to the written Nexus file.

    Example
    -------
    The following example demonstrates the nested structure that the
    function supports. In particular ``Scan1`` has its main data stored in
    ``Scan1/instrument1/slope`` and its axes in
    ``Scan1/actuator1/y`` and ``Scan1/actuator1/x``::

        data_dict = {
            'Scan1': {
                'instrument1': {
                    'slope': {'data': np.random.rand(64, 128), 'name': 'slope', 'units': 'nm'},
                    'metadata': {'description': 'instrument measurement data', 'calibration': 1.23}
                },
                'actuator1': {
                    'x': {'data': np.linspace(0, 1, 128), 'name': 'x', 'units': 'mm'},
                    'y': {'data': np.linspace(0, 2, 64), 'name': 'y', 'units': 'mm'}
                },
                'default': 'instrument1/slope',
                'default_axes': ['actuator1/y', 'actuator1/x'],
                'metadata': {'operator': 'alice', 'date': '2025-10-31', 'scan_type': 'scan 2D'}
            }
        }

        write_nexus_file('out.nxs', data_dict, verbose=True)
    """

    nexus_file = Path(filename)
    with nxopen(nexus_file, 'w') as f:
        # Create a standard entry (optional but customary)
        f['entry'] = NXentry()

        def _write_field(base, key, val):
            """Write a single field under base/key from val which may be ndarray, scalar, list or dict with 'data'."""
            if isinstance(val, dict):
                data = val.get('data')
                name = val.get('name', key)
                units = val.get('units', None)
                if units is not None:
                    f[f"{base}/{key}"] = NXfield(data, name=name, units=units)
                else:
                    f[f"{base}/{key}"] = NXfield(data, name=name)
            else:
                print("raw data field")
                if isinstance(val, list):
                    val = np.array(val)
                try:
                    f[f"{base}/{key}"] = NXfield(val, name=key)
                except TypeError as e:
                    raise TypeError(f"Cannot write field '{key}' under '{base}': {e}")

        def write_spec(base_path, spec):
            """Recursively write a specification dictionary under base_path.

            - sub-dictionaries become NXgroup children
            - arrays/scalars become NXfield nodes
            - 'metadata' dict creates a metadata group of NXfields
            - if 'data' is present, create an NXdata node at base_path/data
              that links to the referenced field and its axes
            """

            # First create child groups and fields
            for key, val in spec.items():
                if key in ('default', 'default_axes', 'metadata'):
                    continue
                node_path = f"{base_path}/{key}"
                if isinstance(val, dict) and  val.get("data") is None:
                    # nested subgroup
                    f[node_path] = NXgroup()
                    write_spec(node_path, val)
                else:
                    # write scalar or array as field
                    _write_field(base_path, key, val)

            # Write metadata if present
            metadata = spec.get('metadata', {}) or {}
            if metadata:
                meta_path = f"{base_path}/metadata"
                f[meta_path] = NXgroup()
                for mk, mv in metadata.items():
                    f[f"{meta_path}/{mk}"] = NXfield(mv, name=mk)

            # Create NXdata if requested
            if 'default' in spec:
                data_ref = spec['default']
                # Resolve data reference to a full path (relative to base_path if no leading slash)
                if isinstance(data_ref, str):
                    if data_ref.startswith('/'):
                        data_path = data_ref.lstrip('/')
                    else:
                        data_path = f"{base_path}/{data_ref}"
                else:
                    # If user provided raw array as 'data', write it under base_path/data_array
                    tmp_name = 'data_array'
                    f[f"{base_path}/{tmp_name}"] = NXfield(data_ref, name=tmp_name)
                    data_path = f"{base_path}/{tmp_name}"

                # collect axes links
                axes = spec.get('default_axes', []) or []
                axis_links = []
                for ax in axes:
                    if isinstance(ax, str):
                        if ax.startswith('/'):
                            ax_path = ax.lstrip('/')
                        else:
                            ax_path = f"{base_path}/{ax}"
                    else:
                        raise TypeError("default_axes entries must be strings specifying relative paths to axis fields")
                    try:
                        axis_field = f[ax_path]
                    except Exception:
                        raise KeyError(f"Axis '{ax}' (resolved to '{ax_path}') not found under '{base_path}'")
                    axis_links.append(NXlink(axis_field))

                try:
                    main_field = f[data_path]
                except Exception:
                    raise KeyError(f"Data field '{data_ref}' (resolved to '{data_path}') not found under '{base_path}'")

                nxdata = NXdata(NXlink(main_field), axis_links)
                f[f"{base_path}/data"] = nxdata

        # Create each top-level scan group under the NXentry and write its spec
        for group_name, spec in data_dict.items():
            group_path = f"entry/{group_name}"
            f[group_path] = NXgroup()
            if not isinstance(spec, dict):
                raise TypeError(f"Expected a dict for group '{group_name}', got {type(spec)}")
            write_spec(group_path, spec)

    if verbose:
        print(f"Saved Nexus file: {nexus_file}")
    return nexus_file


if __name__ == "__main__":
    fin = NexusFile(r'\\ruche-hermes\hermes-soleil\com-hermes\COM-HERMES\2025\2025-10-07\scan_0002.nxs', data_root="scan/scan_data")
    print(fin.df)

    data_dict = {
         'Scan1': {  # Example of a nested scan: instrument + actuator
             'instrument1': {'slope': {"data": np.random.rand(64, 128), "name":"my scan main data", "units":"nm"},
                             'metadata': {'description': 'instrument measurement data', "calibration factor": 1.23}},
             'actuator1':{'x': {"data": np.linspace(0, 1, 128), "name":"measurement axis x", "units":"mm"},
                          'y': {"data": np.linspace(0, 2, 64), "name":"measurement axis y", "units":"mm"},},
             # top-level data points to nested instrument data; axes point to nested actuator fields
             'default': 'instrument1/slope',
             'default_axes': ['actuator1/y', 'actuator1/x'],
             'metadata': {'operator': 'alice', 'date': '2025-10-31', "scan_type": "scan 2D"}
         },
         'Scan2': {  # Example of a complete 1D scan without subgroups
             'scan_data': {'data':np.random.rand(64), "units":"counts", "name":"scan main data"},
             'x': {"data":np.linspace(0, 1, 64), "units":"mm", "name":"scan axis x"},
             'default': 'scan_data',
             'default_axes': ['x'],
             'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D"}
         },
         'Scan3': {  # Example of a 1D scan with a data field but no axes
             'scan_data': np.random.rand(64),
             'x': np.linspace(0, 1, 64),
             'default': 'scan_data',
             'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D no axes"}
         },
         'Scan4': {  # Example of a 1D scan without a data field
             'scan_data': np.random.rand(64),
             'x': np.linspace(0, 1, 64),
             'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D no main data"}
         },
         'Scan5': {  # Example of a scatter scan with a data field having the same dimension as axes
             'scatter': {'intensity': {"data": np.random.rand(1000), "name":"my scan main data"},
                         'metadata': {'description': 'instrument measurement data', "calibration factor": 1.23},
                         'x': {"data": np.random.normal(0, 1, 1000), "name":"x position", "units":"m"},
                         'y': {"data": np.random.normal(0, 2, 1000), "name":"y position", "units":"m"},
                         'xp': {"data": np.random.normal(0, 2, 1000)*1e-3 + np.random.rand(1000)*1e-4, "name":"x angle", "units":"rad"},
                         'yp': {"data": np.random.normal(0, 1, 1000)*1e-3 + np.random.rand(1000)*1e-4, "name":"y angle", "units":"rad"},
                         },
             'default': 'scatter/intensity',
             'default_axes': ['scatter/y', 'scatter/x','scatter/xp','scatter/yp'],
             'metadata': {'operator': 'bob', 'date': '2025-10-31', "scan_type": "scatter data where data and axes have same dimension"}
         },
         'metadata': {"file description": "Example Nexus file created with utils4dd" }
     }
    write_nexus_file(r'D:\Dennetiere\Programmes_Python\Sandbox\test_nexus_write_utils4dd.nxs', data_dict, verbose=True)
