import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Union
import plotly.io as pio
import numpy as np
from plotly.subplots import make_subplots
from itertools import cycle
from scipy.interpolate import interp1d
from collections import defaultdict

class Plot():
    def __init__(self, nrows=1, ncols=1, shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.08, 
                 subplot_titles=None, height=1100, width=1500, title="", legend_orientation="h", legend_yanchor="bottom", 
                 legend_y=-0.1, legend_xanchor="left", legend_x=0, data=None, colormap_name="Rainbow", sync_colors=True,
                 sync_legend=True, axtype="linlin"):
        """Plot class to create and manage plotly figures with subplots.
        Parameters
        ----------
        nrows : int, optional
            Number of rows in the subplot grid, by default 1
        ncols : int, optional
            Number of columns in the subplot grid, by default 1
        shared_xaxes : bool, optional
            Whether to share x-axes among subplots, by default True
        shared_yaxes : bool, optional
            Whether to share y-axes among subplots, by default False
        vertical_spacing : float, optional
            Vertical spacing between subplots, by default 0.08  
        subplot_titles : list, optional
            Titles for each subplot, by default None
        height : int, optional
            Height of the figure in pixels, by default 1100
        width : int, optional
            Width of the figure in pixels, by default 1500
        title : str, optional
            Overall title of the figure, by default ""
        legend_orientation : str, optional
            Orientation of the legend ("h" for horizontal, "v" for vertical), by default "h"
        legend_yanchor : str, optional
            Y anchor position of the legend ("top", "middle", "bottom"), by default "bottom"
        legend_y : float, optional  
            Y position of the legend, by default -0.1
        legend_xanchor : str, optional  
            X anchor position of the legend ("left", "center", "right"), by default "left"
        legend_x : float, optional  
            X position of the legend, by default 0  
        data : pd.DataFrame, optional
            DataFrame containing data for plotting, by default None
        colormap_name : str, optional
            Name of the Plotly colormap to use for coloring traces, by default "Rainbow"
        sync_colors : bool, optional
            Whether to synchronize colors across subplots, by default True
        sync_legend : bool, optional
            Whether to synchronize legend entries across subplots, by default True
        axtype : str, optional
            Type of axes to use. Options are 'linlin', 'loglin', 'linlog', and 'loglog'. Default is 'linlin'.
        """
        self.fig = make_subplots(rows=nrows, cols=ncols, shared_xaxes=shared_xaxes, shared_yaxes=shared_yaxes,
                                vertical_spacing=vertical_spacing,
                                subplot_titles=subplot_titles)
        self.fig.update_layout(height=height, width=width)
        self.fig.update_layout(
                title_text=title,
                legend=dict(orientation=legend_orientation, yanchor=legend_yanchor, 
                            y=legend_y, xanchor=legend_xanchor, x=legend_x),
                hovermode="x", hoversubplots="axis",
            )
        if axtype[3:] == 'log':
            for r in range(1, nrows+1):
                for c in range(1, ncols+1):
                    self.fig.update_yaxes(type="log", row=r, col=c)
        if axtype[:3] == 'log':
            for r in range(1, nrows+1):
                for c in range(1, ncols+1):
                    self.fig.update_xaxes(type="log", row=r, col=c)
        self.n_traces = 0
        self._n_rows = nrows
        self._n_cols = ncols
        self.data = data
        self.colormap_name = colormap_name
        self._sync_colors = sync_colors
        self._color_cycle = None
        self._sync_legend = sync_legend
        self.get_color_cycle(colormap_name, 1)
     
    def set_theme(self, theme_name: str= "plotly_dark"):
        """set_theme sets the theme for plots

        Parameters
        ----------
        theme_name : str, optional
            plotly theme to be used, by default "plotly_dark", can be any of the plotly built-in themes such as 
            "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation", 
            "xgridoff", "ygridoff", "gridon", "none".
        """
        self.fig.update_layout(template=theme_name)

    def get_color_cycle(self, colormap_name="Rainbow", n_colors=10):
        """Generate a color cycle from a specified colormap.

        Parameters
        ----------
        colormap_name :str, optional
            Name of the Plotly colormap to use, by default "Rainbow".
        n_colors : int
            Number of colors needed.
        
        Returns
        -------
        cycle: An iterator that cycles through the generated colors.
        """
        colormap = getattr(px.colors.sequential, colormap_name)

        if len(colormap) < n_colors:
            # interpolate to get more colors
            x = np.linspace(0, 1, len(colormap))
            if colormap[0].startswith("rgb"):
                # convert rgb(...) to hex
                colormap = [
                    f"#{int(c.split('(')[1].split(')')[0].split(',')[0]):02x}"
                    f"{int(c.split('(')[1].split(')')[0].split(',')[1]):02x}"
                    f"{int(c.split('(')[1].split(')')[0].split(',')[2]):02x}"
                    for c in colormap
                ]
            r = [int(c[1:3], 16) for c in colormap]
            g = [int(c[3:5], 16) for c in colormap]
            b = [int(c[5:7], 16) for c in colormap]

            r_interp = interp1d(x, r, kind='linear')
            g_interp = interp1d(x, g, kind='linear')
            b_interp = interp1d(x, b, kind='linear')

            x_new = np.linspace(0, 1, n_colors)
            new_colormap = [
                f"#{int(r_interp(xi)):02x}{int(g_interp(xi)):02x}{int(b_interp(xi)):02x}"
                for xi in x_new
            ]
            colormap = new_colormap
        else:
            # sample evenly from the existing colormap
            indices = np.linspace(0, len(colormap) - 1, n_colors).astype(int)
            sampled_colors = [colormap[i] for i in indices]
            colormap =  sampled_colors
        if self._sync_colors:
            self._color_cycle = [[cycle(colormap)]*self._n_cols for _ in range(self._n_rows)]
        else:
            self._color_cycle = [[cycle(colormap)]]

    def add_trace(self, x, y, row=1, col=1, label="data", mode="lines"):
        if self.data is not None and isinstance(x, str):
            x = self.data[x].values
        if self.data is not None and isinstance(y, str):
            y = self.data[y].values
        showlegend = True 
        if row > 1 or col > 1:
            if self._sync_legend:
                showlegend = False # links the legend entry to the first subplot only & synchronizes displayed legends
        self.fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=mode,
                name=label,
                legendgroup=label,
                showlegend=showlegend,
                line=dict(color="blue", width=2), # color is gonna be overwritten by update_colors
                # hovertemplate="X=%{x:.2f} mm<br>H=%{y:.3f} nm<extra></extra>"
            ),
            row=row, col=col
        )
        self.update_colors()

    def get_trace_subplot_position(self, trace):
        """
        Return (row, col) of a given trace inside a plotly subplot figure.
        """
        # xaxis and yaxis names like "x2", "y3", or None -> "x" / "y"
        xa = trace.xaxis or "x"
        ya = trace.yaxis or "y"

        # Strip axis letters to get number: "x" -> 1, "x3" -> 3
        xa_num = 1 if xa == "x" else int(xa[1:])
        ya_num = 1 if ya == "y" else int(ya[1:])

        # The subplot grid associates xaxis and yaxis by number
        # (x2 pairs with y2, x3 with y3, etc.)
        axis_num = xa_num  # or ya_num — they are paired

        # Convert axis number to (row, col)
        # Plotly fills subplots row by row (row-major order)
        ncols = self._n_cols
        row = (axis_num - 1) // ncols + 1
        col = (axis_num - 1) % ncols + 1

        return row, col

    def update_colors(self):
        if self._sync_colors:
            # n_colors is the maximum number of traces in any subplot
            counts = defaultdict(int)
            for trace in self.fig.data:
                r, c = self.get_trace_subplot_position(trace)
                loc = str(r) + "_" + str(c)
                counts[loc] += 1
            n_colors = max(counts.values())
        else:
            # n_colors is the total number of traces
            n_colors = len(self.fig.data)
        self.get_color_cycle(self.colormap_name, n_colors=n_colors)
        for trace in self.fig.data:
            r, c = self.get_trace_subplot_position(trace)
            if self._sync_colors:
                trace.line.color = next(self._color_cycle[r-1][c-1])
            else:
                trace.line.color = next(self._color_cycle[0][0])

    def set_axes_titles(self, row=1, col=1, x_title="", y_title=""):
        self.fig.update_xaxes(title_text=x_title, row=row, col=col)
        self.fig.update_yaxes(title_text=y_title, row=row, col=col)

    def show(self):
        self.fig.show()



def plot_pd(
        dfs: List[pd.DataFrame] = [],
        names: List[str] = [],
        x: Union[None, int, str, List[Union[int, str, None]]] = None,
        y: List[str] = [],
        axtype: str = 'linlin',
        renormalize: Union[bool, float, List[float]] = False,
        **kwargs
) -> px.line:
    """
    Plots data from multiple DataFrames using a common column as x and renaming columns listed in y with names.

    Parameters:
    dfs : List[pd.DataFrame]
        A list of pandas DataFrames containing the data to be plotted.
    names : List[str]
        A list of new names for the columns specified in y.
    x : Union[None, int, str, List[Union[int, str, None]]], optional
        The column name or index to be used as the x-axis. If a list is provided,
        each DataFrame will be expected to have a different x column. If None is provided, the index will be usd as x
    y : List[str]
        A list of column names in the DataFrames to be renamed and plotted.
    axtype : str, optional
        Type of axes to use. Options are 'linlin', 'loglin', 'linlog', and 'loglog'. Default is 'linlin'.
    renormalize : Union[bool, float, List[float]], optional
        If True, each y column will be normalized by its maximum value. If a float, each y column will be normalized by the closest value to this float in the x column. If a list of floats, each y column will be normalized by the corresponding float in the list. Default is False.
    **kwargs
        Additional keyword arguments to pass to plotly.express.line.

    Returns:
    plotly.graph_objects.Figure
        The plotly figure object.

    Raises:
    AssertionError
        If the lengths of dfs, names, and y are not equal.
    """
    assert len(dfs) == len(names)
    assert len(dfs) == len(y)
    if isinstance(x, list):
        assert len(dfs) == len(x)
    else:
        x = [x]*len(dfs)

    ####
    fig = go.Figure()
    for i, df in enumerate(dfs):
        if x[i] is None:
            XX = df.index
        elif isinstance(x[i], int):
            XX = df.iloc[:, x[i]]
        else:
            try:
                XX = df[x[i]]
            except KeyError:
                raise KeyError(str(x[i]) + " not valid. Valid x are " + str(df.columns))
        try:
            YY = df[y[i]]
        except KeyError:
            raise KeyError(str(y[i]) + " not valid. Valid y are " + str(df.columns))
        if renormalize:
            if isinstance(renormalize, bool):
                    YY = YY / YY.max()
            elif isinstance(renormalize, float):
                    dfsort = df.iloc[(XX - renormalize).abs().argsort()[:2]]
                    closest_index = dfsort.index.tolist()[0]
                    YY = YY / YY.iloc[closest_index]
            else:
                    YY = YY / renormalize[i]
        fig.add_trace(go.Scatter(x=XX,
                                 y=YY,
                                 mode='lines', name=names[i]))
    # Update layout
    fig.update_layout(title=kwargs.get('title', ""),
                      xaxis_title=kwargs.get('xaxis_title', "X"),
                      yaxis_title=kwargs.get('yaxis_title', "values"))
    ####

    if axtype[3:] == 'log':
        fig.update_yaxes(type="log")
    if axtype[:3] == 'log':
        fig.update_xaxes(type="log")
    return fig

if __name__ == "__main__":
    # Example usage of plot_pd
    # df1 = pd.DataFrame({
    #     'A': [1, 2, 3, 4, 5],
    #     'B': [10, 20, 30, 40, 50]
    # })
    # df2 = pd.DataFrame({
    #     'A': [1, 2, 3, 4, 5],
    #     'C': [15, 25, 35, 45, 55]
    # })
    # fig = plot_pd(
    #     dfs=[df1, df2],
    #     names=['DataFrame 1', 'DataFrame 2'],
    #     x='A',
    #     y=['B', 'C'],
    #     axtype='linlin',
    #     renormalize=False,
    #     title='Example Plot',
    #     xaxis_title='A values',
    #     yaxis_title='B and C values'
    # )
    # fig.show()

    df3 = pd.DataFrame({
        'X': np.linspace(1, 100, 100),
        'rand': np.random.rand(100) ,
        'Gauss1': np.exp(-0.5*((np.linspace(1, 100, 100)-30)/5)**2),
        'Gauss2': np.exp(-0.5*((np.linspace(1, 100, 100)-70)/10)**2)
    })
    # Example of using the Plot class with linked subplots
    # plt = Plot(nrows=2, ncols=1, data=df3, title="Test Plot Class", width=1500, height=700, 
    #            shared_xaxes=True, sync_colors=True, colormap_name="Jet")
    # plt.add_trace(x='X', y='rand', row=2, col=1, label="Gauss 1") # same label and sync_legend=True -> only one legend entry
    # plt.add_trace(x='X', y='Gauss1', row=1, col=1, label="Gauss 1")
    # plt.add_trace(x='X', y='Gauss2', row=1, col=1, label="Gauss 2")
    # plt.set_axes_titles(row=1, col=1, y_title="Random Values")
    # plt.set_axes_titles(row=2, col=1, x_title="X-axis", y_title="Gaussian Values")
    # plt.show()

    # Example of using the Plot class in simple mode
    plt = Plot(data=df3, title="Test Plot Class", width=1500, height=700, axtype='linlog',colormap_name="Rainbow")
    plt.set_theme("plotly_dark")
    plt.add_trace(x='X', y='rand', label="Rand") 
    plt.add_trace(x='X', y='Gauss1', label="Gauss 1")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 2")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 3")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 4")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 5")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 6")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 7")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 8")
    plt.add_trace(x='X', y='Gauss2', label="Gauss 9")
    plt.set_axes_titles(x_title="X-axis", y_title="Values")
    plt.show()
