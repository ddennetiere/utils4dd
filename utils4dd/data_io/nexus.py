import tables
import plotly.express as px
import numpy as np
import pandas as pd
from pathlib import Path
from nexusformat.nexus import NXdata, NXentry, NXfield, nxopen, NXgroup, NXlink


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
        with tables.open_file(filename) as fin:
            for node in fin.iter_nodes('/' + data_root):
                node_data = np.array(node)
                if node_data.ndim == 1:
                    data[node.get_attr('long_name').decode()] = node_data
                elif node_data.ndim == 3:
                    self.images[node.get_attr('long_name').decode()] = node_data
        self.df = pd.DataFrame(data)
        if x is not None:
            self.df = self.df.set_index(x)
        self._xarray = None

    def __repr__(self):
        return self.df.__repr__()

    def _repr_html_(self):
        return self.df._repr_html_()

    @property
    def xarray(self):
        return self.df.to_xarray()

    @xarray.setter
    def xarray(self, value):
        self.df = pd.DataFrame(value.to_pandas())

    def __getitem__(self, item):
        return self.df[item]

    def __setitem__(self, key, value):
        self.df[key] = value

    def plot(self, axtype="linlin"):
        fig = px.line(self.xarray)
        if axtype[3:] == 'log':
            fig.update_yaxes(type="log")
        if axtype[:3] == 'log':
            fig.update_xaxes(type="log")
        return fig


def write_nexus_file(filename, data_dict, verbose=False):
    """Write a Nexus file from a structured dictionary describing scans.

    The function expects ``data_dict`` to be a mapping where each key is a
    group name (for example ``"Scan1"``) and each value is another mapping
    describing fields for that group. Each scan group should contain the
    main data array (commonly named ``'measurement'``) and its axis arrays
    (commonly ``'x'`` and ``'y'``). The group may also contain additional
    arrays and a ``'metadata'`` dict.

    The function will create a top-level group for each scan (e.g.
    ``/Scan1``), write the arrays as :class:`nexusformat.nexus.NXfield` nodes
    (e.g. ``/Scan1/measurement``, ``/Scan1/x``, ``/Scan1/y``), write
    metadata under ``/Scan1/metadata/*``, and add an ``/Scan1/data``
    :class:`nexusformat.nexus.NXdata` node that links the main data to its
    axes using :class:`nexusformat.nexus.NXlink`.

    Parameters
    ----------
    filename : str or pathlib.Path
        Output Nexus filename (will be created/overwritten).
    data_dict : dict
        Mapping of group_name -> group_spec, where ``group_spec`` is a
        dictionary with keys for arrays and metadata. Expected keys:

        - ``'measurement'`` (or another key specified by ``'data'``) : ndarray or dict
            The main data array (2D for images, 1D for scans) or dict containing data and optional name and units.
        - axis arrays such as ``'x'``, ``'y'`` : 1D ndarray, optionnal
        - ``'data'`` : str, optional
            Name of the key that contains the main data.
        - ``'data_axes'`` : list of str, optional
            List of axis keys (e.g. ``['y', 'x']``) indicating the order of
            axes for NXdata. If omitted, the function will try to infer
            sensible axes (``['y','x']`` for 2D data, ``['x']`` for 1D data).
        - ``'metadata'`` : dict, optional
            Mapping of metadata name -> value. Each becomes an NXfield under
            ``/group/metadata``.
        - other keys : ndarray, dict or scalar to be recorded as fields.

    verbose : bool, optional
        If True, prints the path of the written file.

    Returns
    -------
    pathlib.Path
        Path to the written Nexus file.

    Example
    -------
    >>> data_dict = {
    ...     'Scan1': {  # Example of a complete 2D scan
    ...         'measurement': {"data": np.random.rand(64, 128), "name":"my scan data", "units":"nm"},
    ...         'x': {"data": np.linspace(0, 1, 128), "name":"measurement axis x", "units":"mm"},
    ...         'y': {"data": np.linspace(0, 2, 64), "name":"measurement axis y", "units":"mm"},
    ...         'data': 'measurement',
    ...         'data_axes': ['y', 'x'],
    ...         'metadata': {'operator': 'alice', 'date': '2025-10-31', "scan_type": "scan 2D"}
    ...     },
    ...     'Scan2': {  # Example of a complete 1D scan
    ...         'scan_data': np.random.rand(64),
    ...         'x': np.linspace(0, 1, 64),
    ...         'data': 'scan_data',
    ...         'data_axes': ['x'],
    ...         'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D"}
    ...     },
    ...     'Scan3': {  # Example of a 1D scan with a data field but no axes
    ...         'scan_data': np.random.rand(64),
    ...         'x': np.linspace(0, 1, 64),
    ...         'data': 'scan_data',
    ...         'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D no axes"}
    ...     },
    ...     'Scan4': {  # Example of a 1D scan without a data field
    ...         'scan_data': np.random.rand(64),
    ...         'x': np.linspace(0, 1, 64),
    ...         'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D no main data"}
    ...     },
    ... }
    >>> write_nexus_file('out.nxs', data_dict, verbose=True)
    """

    nexus_file = Path(filename)
    with nxopen(nexus_file, 'w') as f:
        # Create a standard entry (optional but customary)
        f['entry'] = NXentry()

        for group_name, spec in data_dict.items():
            # create a group for this scan under the NXentry so defaults are allowed
            group_path = f"entry/{group_name}"
            f[group_path] = NXgroup()

            # write all arrays (fields) found in the group spec except metadata
            for key, val in spec.items():
                if key == 'metadata' or key == 'data' or key == 'data_axes':
                    continue
                # write numpy arrays or scalars as NXfield
                if isinstance(val, (np.ndarray, int, float)) or np.isscalar(val):
                    f[f"{group_path}/{key}"] = NXfield(val, name=key)
                elif isinstance(val, dict):
                    f[f"{group_path}/{key}"] = NXfield(val["data"], name=val.get("name", key), units=val.get("units", ""))

            # write metadata
            metadata = spec.get('metadata', {}) or {}
            if metadata:
                f[f"{group_path}/metadata"] = NXgroup()
                for mk, mv in metadata.items():
                    f[f"{group_path}/metadata/{mk}"] = NXfield(mv, name=mk)

            # determine which key is the main data
            main_key = spec.get('data', None)
            if main_key is not None:
                main_array = spec.get(main_key, None)
                if isinstance(main_array, dict):
                    main_array = main_array["data"]
                elif isinstance(main_array, list):
                    main_array = np.array(main_array)
                if main_array is None:
                    raise ValueError(f"No array found for group '{group_name}' to use as main data")

                # establish axis links
                axes = spec.get('data_axes', None)
                axis_links = []
                if axes is not None:
                    for ax in axes:
                        try:
                            axis_field = f[f"{group_path}/{ax}"]
                        except Exception:
                            raise KeyError(f"Axis '{ax}' not found in group '{group_name}'")
                        axis_links.append(NXlink(axis_field))
                    assert len(axis_links) == main_array.ndim, \
                        f"Number of axes ({len(axis_links)}) does not match data dimensions ({main_array.ndim})"

                main_field = f[f"{group_path}/{main_key}"]
                # create NXdata linking the main data to its axes
                nxdata = NXdata(NXlink(main_field), axis_links)
                f[f"{group_path}/data"] = nxdata

    if verbose:
        print(f"Saved Nexus file: {nexus_file}")
    return nexus_file


if __name__ == "__main__":
    fin = NexusFile(r'\\ruche-hermes\hermes-soleil\com-hermes\COM-HERMES\2025\2025-10-07\scan_0002.nxs', data_root="scan/scan_data")
    print(fin.df)

    data_dict = {
         'Scan1': {  # Example of a complete 2D scan
             'measurement': {"data": np.random.rand(64, 128), "name":"my scan data", "units":"nm"},
             'x': {"data": np.linspace(0, 1, 128), "name":"measurement axis x", "units":"mm"},
             'y': {"data": np.linspace(0, 2, 64), "name":"measurement axis y", "units":"mm"},
             'data': 'measurement',
             'data_axes': ['y', 'x'],
             'metadata': {'operator': 'alice', 'date': '2025-10-31', "scan_type": "scan 2D"}
         },
         'Scan2': {  # Example of a complete 1D scan
             'scan_data': np.random.rand(64),
             'x': np.linspace(0, 1, 64),
             'data': 'scan_data',
             'data_axes': ['x'],
             'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D"}
         },
         'Scan3': {  # Example of a 1D scan with a data field but no axes
             'scan_data': np.random.rand(64),
             'x': np.linspace(0, 1, 64),
             'data': 'scan_data',
             'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D no axes"}
         },
         'Scan4': {  # Example of a 1D scan without a data field
             'scan_data': np.random.rand(64),
             'x': np.linspace(0, 1, 64),
             'metadata': {'operator': 'paul', 'date': '2025-10-31', "scan_type": "scan 1D no main data"}
         },
     }
    write_nexus_file(r'D:\Dennetiere\Programmes_Python\Sandbox\test_nexus_write_utils4dd.nxs', data_dict, verbose=True)
