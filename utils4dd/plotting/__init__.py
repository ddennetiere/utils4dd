import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Union


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
