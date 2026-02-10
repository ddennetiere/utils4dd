import h5py
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from nexusformat.nexus import NXdata, NXentry, NXfield, nxopen, NXgroup, NXlink, NXroot, NXinstrument, NXsample, NXcollection, NXpositioner, NXdetector
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

def _guess_nxclass(name):

    lname = name.lower()

    if "instrument" in lname:
        return NXinstrument
    if "detector" in lname:
        return NXdetector
    if any(k in lname for k in ["actuator", "motor", "stage", "positioner"]):
        return NXpositioner
    if "sample" in lname:
        return NXsample

    return NXcollection


def _set_interpretation(field):

    arr = np.asarray(field.nxdata)

    if arr.ndim == 1:
        field.attrs["interpretation"] = "spectrum"
    elif arr.ndim == 2:
        field.attrs["interpretation"] = "image"
    elif arr.ndim > 2:
        field.attrs["interpretation"] = "volume"


def _resolve(base, rel):
    if rel.startswith("/"):
        return rel.strip("/")
    return f"{base}/{rel}"


# -------------------------------------------------
# Main writer
# -------------------------------------------------
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
    - To mark the main dataset for a NXdata node, set ``'default'`` to a string
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

    root = NXroot()
    root.attrs["default"] = "entry"
    root["entry"] = NXentry()

    nxdata_requests = []

    # -------------------------------------------------
    # PASS 1 - build structure
    # -------------------------------------------------

    def write_structure(base_path, spec):

        group = root[base_path]

        for key, val in spec.items():

            if key in ("default", "default_axes", "metadata"):
                continue

            # subgroup
            if isinstance(val, dict) and "data" not in val:

                group[key] = _guess_nxclass(key)()
                write_structure(f"{base_path}/{key}", val)

            else:
                # field
                if isinstance(val, dict) and "data" in val:

                    field = NXfield(val["data"], 
                                    compression=None,
                                    shuffle=False,
                                    fletcher32=False,
                                    scaleoffset=None,
                                    chunks=None,)

                    if "name" in val:
                        field.attrs["long_name"] = val["name"]

                    if "units" in val:
                        field.attrs["units"] = val["units"]

                else:
                    field = NXfield(val, 
                                    compression=None,
                                    shuffle=False,
                                    fletcher32=False,
                                    scaleoffset=None,
                                    chunks=None,)

                group[key] = field
                _set_interpretation(field)

        if "metadata" in spec:
            group["metadata"] = NXcollection()
            for mk, mv in spec["metadata"].items():
                group["metadata"][mk] = NXfield(mv)

        if "default" in spec:
            nxdata_requests.append((base_path, spec))

    entry = root["entry"]

    for gname, spec in data_dict.items():

        if gname == "metadata":
            continue

        entry[gname] = NXcollection()
        write_structure(f"entry/{gname}", spec)

    if "metadata" in data_dict:
        entry["metadata"] = NXcollection()
        for k, v in data_dict["metadata"].items():
            entry["metadata"][k] = NXfield(v)

    # -------------------------------------------------
    # PASS 2 - create NXdata as CHILD
    # -------------------------------------------------

    created_paths = []

    for base_path, spec in nxdata_requests:

        group = root[base_path]

        # Convert existing group into NXdata
        group.nxclass = "NXdata"
        nxdata = group

        # ---------- signal ----------
        signal_path = _resolve(base_path, spec["default"])
        signal_field = root[signal_path]
        signal_key = signal_path.split("/")[-1]

        nxdata.attrs["signal"] = signal_key

        # link signal if not already present locally
        if signal_key not in nxdata:
            nxdata[signal_key] = NXlink(signal_field)

        signal = np.asarray(signal_field.nxdata)

        # ---------- axes ----------
        axes_keys = []
        axes_fields = []

        for i, ax in enumerate(spec.get("default_axes", [])):

            ax_path = _resolve(base_path, ax)
            ax_field = root[ax_path]

            ax_key = ax_path.split("/")[-1]

            if ax_key not in nxdata:
                nxdata[ax_key] = NXlink(ax_field)

            ax_field.attrs["axis"] = i + 1

            axes_keys.append(ax_key)
            axes_fields.append(ax_field)

        if axes_keys:
            nxdata.attrs["axes"] = axes_keys

        # scatter detection
        if (
            signal.ndim == 1
            and len(axes_fields) > 1
            and all(len(ax.nxdata) == len(signal) for ax in axes_fields)
        ):
            nxdata.attrs["interpretation"] = "point"
            nxdata.attrs["plot_type"] = "scatter"

        created_paths.append(base_path)


    # -------------------------------------------------
    # Default chain root → entry → last NXdata
    # -------------------------------------------------

    if created_paths:

        deepest = created_paths[-1]

        current = root
        for part in deepest.split("/"):
            current.attrs["default"] = part
            current = current[part]

    root.save(nexus_file, "w")

    if verbose:
        print("Saved:", nexus_file)

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
