import tables
import plotly.express as px
import numpy as np
import pandas as pd


class NexusFile(object):
    """
    Class for opening SOLEIL's nexus files and store the data in a pandas DataFrame (NexusFile.df) and in a synchronized
    xarray (NexusFile.xarray)
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


if __name__ == "__main__":
    fin = NexusFile(r'X:/com-hermes/COM-BiPer/2024/2024-06-04/scan_0051.nxs', data_root="root_spyc_tscan/scan_data")
    print(fin.images)
