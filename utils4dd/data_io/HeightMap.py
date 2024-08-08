import logging

import numpy as np
from plotly.express import line, imshow
from plotly.graph_objects import Figure, Scatter
from numpy.polynomial.legendre import Legendre
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import welch


class LegendreMap(object):
    """
    A class to create and fit a map using Legendre polynomials.

    Attributes:
    ----------
    coeffs : np.ndarray
        Coefficients for the Legendre polynomials. Default is None.
    width : int
        Width of the map (number of columns). Default is 100.
    height : int
        Height of the map (number of lines). Default is 50.
    deg : int
        Degree of the Legendre polynomials. Default is 3.
    map : np.ndarray
        The generated map using Legendre polynomials.
    base : list
        A list of base Legendre polynomial values.
    index : tuple
        A meshgrid of indices for the map.
    """

    def __init__(self, coeffs=None, width=100, height=50, deg=3):
        """
        Initializes the LegendreMap with given coefficients, width, height, and degree.

        Parameters:
        ----------
        coeffs : np.ndarray, optional
            Coefficients for the Legendre polynomials. Default is None. If None coefficients are assumed from degree
        width : int, optional
            Width of the map. Default is 100.
        height : int, optional
            Height of the map. Default is 50.
        deg : int, optional
            Degree of the Legendre polynomials. Default is 3.
        """
        if coeffs is None:
            self.coeffs = np.triu(np.ones((deg, deg)))[:, ::-1]
            self.deg = deg
        else:
            self.coeffs = np.array(coeffs)
            self.deg = max(*self.coeffs.shape)
        self.map = np.zeros((height, width))
        self.base = []
        self.index = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        assert self.map.shape == self.index[0].shape
        self.make_base()
        self.get_map()

    def make_base(self):
        """
        Generates the base Legendre polynomial values for the map.
        """
        x = self.index[0][0, :]
        y = self.index[1][:, 0]
        self.base = []
        for i in range(self.coeffs.shape[0]):
            base_row = []
            degx = np.zeros(i + 1)
            degx[-1] = 1
            lx = Legendre(degx)(x)
            for j in range(self.coeffs.shape[1]):
                degy = np.zeros(j + 1)
                degy[-1] = 1
                ly = Legendre(degy)(y)
                baseij = np.tile(lx, (y.shape[0], 1)) * np.tile(ly.reshape(ly.shape[0], 1), x.shape[0])
                baseij /= np.sqrt((baseij ** 2).mean())  # normalization
                base_row.append(baseij)
            self.base.append(base_row)

    def show_base(self):
        """
        Displays the base Legendre polynomial values as a heatmaps.
        """
        fig = make_subplots(rows=self.coeffs.shape[0], cols=self.coeffs.shape[1])
        for i in range(self.coeffs.shape[0]):
            for j in range(self.coeffs.shape[1]):
                fig.add_trace(go.Heatmap(z=self.base[i][j], coloraxis="coloraxis",
                                         name=f"L<sub>{i},{j}</sub>"),
                              row=i + 1, col=j + 1)
        fig.update_layout(coloraxis={'colorscale': 'Plasma', 'cmin': -1, "cmax": 1})
        fig.show()

    def show_decomp(self):
        """
        Displays the decomposition of the map into its Legendre polynomial components.
        """
        fig = make_subplots(rows=self.coeffs.shape[0], cols=self.coeffs.shape[1])
        for i in range(self.coeffs.shape[0]):
            for j in range(self.coeffs.shape[1]):
                fig.add_trace(go.Heatmap(z=self.base[i][j] * self.coeffs[i][j],
                                         name=f"L<sub>{i},{j}</sub> * {self.coeffs[i][j]}"),
                              row=i + 1, col=j + 1)
        fig.show()

    def get_map(self):
        """
        Generates the map using the base Legendre polynomial values and coefficients.

        Returns:
        -------
        np.ndarray
            The generated map.
        """
        self.map *= 0
        for i in range(self.coeffs.shape[0]):
            for j in range(self.coeffs.shape[1]):
                self.map += self.base[i][j] * self.coeffs[i][j]
        return self.map

    def show_map(self):
        """
        Displays the generated map.
        """
        fig = px.imshow(self.map)
        fig.show()

    def fit(self, z, mask=None):
        """
        Fits the Legendre map to the given data.

        Parameters:
        ----------
        z : np.ndarray or HeightMap
            Data to fit the map to.
        mask : np.ndarray, optional
            Mask to apply to the fitting process. Default is None.

        Returns:
        -------
        np.ndarray
            The residual between the data and the fitted map.
        """
        if isinstance(z, HeightMap):
            z = z.height.values
        else:
            assert isinstance(z, np.ndarray)
        assert z.shape == self.map.shape
        if mask is None:
            mask = np.ones(self.coeffs.shape)
        for i in range(self.coeffs.shape[0]):
            for j in range(self.coeffs.shape[1]):
                self.coeffs[i, j] = (self.base[i][j] * z * mask[i, j]).mean()

        return z - self.get_map()

    def check_solution_unicity(self, z, i=1, j=1):
        """
        Checks the uniqueness of the solution for the given data and indices.

        This method asserts that the mean value of the product of the residual and
        the base Legendre polynomial is less than a small threshold, indicating
        that the solution is unique.

        Parameters:
        ----------
        z : np.ndarray
            Data to check the solution against.
        i : int, optional
            Row index for the coefficient. Default is 1.
        j : int, optional
            Column index for the coefficient. Default is 1.

        Raises:
        ------
        AssertionError
            If the solution is not unique for the given base indices.
        """
        assert ((z - self.coeffs[i, j] * self.base[i][j]) * self.base[i][j]).mean() < 1e-15, \
            "Solution on base {i},{j} is not unique"

    def check_norms(self):
        """
        Checks the normalization of the base Legendre polynomial values.

        This method calculates the norms of the base Legendre polynomial values and
        asserts that each norm is equal to 1, indicating proper normalization.

        Returns:
        -------
        np.ndarray
            The norms of the base Legendre polynomial values.

        Raises:
        ------
        AssertionError
            If any base is not properly normalized.
        """
        norms = np.zeros(self.coeffs.shape)
        for i in range(self.coeffs.shape[0]):
            for j in range(self.coeffs.shape[1]):
                norms[i, j] = (self.base[i][j] ** 2).mean()
                assert norms[i, j] - 1 < 1e-12, f"normalisation error for base {i},{j}"
        return norms


class HeightMap(object):
    """
    A class to represent and manipulate a height map.

    Attributes:
    ----------
    height : np.ndarray
        The height values of the map.
    psd : list
        The power spectral density (PSD) values for each axis.
    psd_fit : list
        The fitted PSD values.
    filepath : str
        The file path to the height map data.
    """
    def __init__(self):
        """
        Initializes the HeightMap with default values.
        """
        self.height = None
        self.psd = [None, None]
        self.psd_fit = []
        self.filepath = None

    def getHeight(self):
        """
        Abstract method to be implemented in a child class to retrieve height data.

        Raises:
        ------
        NotImplementedError
            If the method is not implemented in a child class.
        """
        raise NotImplementedError("Method to be implemented in child class")

    def getPixel(self):
        """
        Abstract method to be implemented in a child class to retrieve pixel size.

        Raises:
        ------
        NotImplementedError
            If the method is not implemented in a child class.
        """
        raise NotImplementedError("Method to be implemented in child class")

    def get_psd(self, axis: int = 0, show_psd: bool = False, mean: str = "psd"):
        """
        Calculates the power spectral density (PSD) of the height map.

        Parameters:
        ----------
        axis : int, optional
            The axis along which to compute the PSD. Default is 0.
        show_psd : bool, optional
            Whether to display the PSD plot. Default is False.
        mean : str, optional
            The method to compute the mean PSD. Options are 'line' or 'psd'. Default is 'psd'. 'line' returns the PSD
            of the mean of the lines along 'axis', 'psd' returns the mean of the PSDs of each line along 'axis'

        Returns:
        -------
        tuple
            The frequencies and corresponding PSD values.

        Raises:
        ------
        ValueError
            If the mean parameter is not 'line' or 'psd'.
        """
        if self.height is None:
            self.getHeight()
        if mean == "line":
            fft = np.fft.fft(self.height.values.mean(axis))
            freq = np.fft.fftfreq(self.height.values.mean(axis).shape[0], d=self.getPixel())
            psd = np.abs(fft) ** 2
        elif mean == "psd":
            # psds = []
            # if axis == 0:
            #     heights = self.height
            # else:
            #     heights = self.height.T
            # for height_line in heights:
            #     fft = np.fft.fft(height_line)
            #     freq = np.fft.fftfreq(self.height.values.mean(axis).shape[0], d=self.getPixel())
            #     psds.append(np.abs(fft) ** 2)
            # psd = np.array(psds).mean(axis=0)
            freq, psd = welch(self.height.values.T, fs=self.getPixel(), axis=axis, scaling="density")
            psd = psd.T.mean(axis=axis)
            freq = 1/np.sqrt(freq)
            # logging.debug(psd)
            logging.debug(f"Height shape {self.height.values.shape}")
            logging.debug(f"PSD shape {psd.shape}")
            logging.debug(freq)
            logging.debug(f"freq shape {freq.shape}")
            # raise RuntimeError("")
        else:
            raise ValueError("mean parameter must be either 'line' or 'psd'")
        self.psd[axis] = (freq[freq > 0], psd[freq > 0])
        logging.debug(psd.shape)
        if show_psd:
            fig_psd = line(x=freq[freq > 0], y=psd[freq > 0], log_y=True, log_x=True,
                           title=f"Power Spectral Density<br><sup>{self.filepath}, mean method = {mean}</sup>")
            fig_psd.update_xaxes(
                title=r"$\text{Spatial frequency }(m^{-1})$",
                mirror=True,
                ticks='outside',
                showline=True,
            )
            fig_psd.update_yaxes(
                title=r"$\text{Power}$",
                mirror=True,
                ticks='outside',
                showline=True,
            )
            fig_psd.show()
        return freq[freq > 0], psd[freq > 0]

    def fit_psd(self, axis=0, show_fit=False, cutoffs=None, clear_show=False):
        """
        Fits the power spectral density (PSD) of the height map.

        Parameters:
        ----------
        axis : int, optional
            The axis along which to fit the PSD. Default is 0.
        show_fit : bool, optional
            Whether to display the fitted PSD plot. Default is False.
        cutoffs : list, optional
            The cutoff frequencies for fitting. Default is None.
        clear_show : bool, optional
            Whether to clear the background of the plot. Default is False.

        Returns:
        -------
        list
            A list of dictionaries containing the fit ranges and coefficients.
        """
        if cutoffs is None:
            cutoffs = [10, ]
        if self.psd[axis] is None:
            self.get_psd(axis=axis)
        freq, psd = self.psd[axis]
        fits = []
        if show_fit:
            fig_psd = Figure()
            fig_psd.add_trace(Scatter(x=freq[freq > 0], y=psd[freq > 0], name="PSD"))
            # fig_psd.add_trace(Scatter(x=np.log10(freq[freq > 0]), y=np.log10(psd[freq > 0]), name="PSD"))
        for i, cutoff in enumerate(cutoffs):
            if i != len(cutoffs) - 1:
                sub_freq = freq[(freq > cutoff) & (freq < cutoffs[i + 1])]
                sub_psd = psd[(freq > cutoff) & (freq < cutoffs[i + 1])]
            else:
                sub_freq = freq[(freq > cutoff)]
                sub_psd = psd[(freq > cutoff)]
            weights = np.concatenate((np.diff(np.log10(sub_freq)), [np.diff(np.log10(sub_freq))[-1]]))
            a, b = np.polyfit(x=np.log10(sub_freq), y=np.log10(sub_psd), deg=1, w=weights)
            fits.append(dict(range=(sub_freq.min(), sub_freq.max()), coeff=(a, b)))
            if show_fit:
                fig_psd.add_trace(Scatter(x=sub_freq, y=10**(a * np.log10(sub_freq) + b),
                                          name=f"fit {sub_freq.min():.2f}-{sub_freq.max():.2f} (coeff {a:.2f})"))
        if show_fit:
            fig_psd.update_layout(dict(title=f"Mirror PSD and fits<br><sup>{self.filepath}</sup>"))
            fig_psd.update_xaxes(
                title=r"$\text{Frequency }(m^{-1})$",
                mirror=True,
                ticks='outside',
                showline=True,
                type="log",
                minor=dict(ticks="inside",  showgrid=True)
            )
            fig_psd.update_yaxes(
                title=r"$\text{Power}$",
                mirror=True,
                ticks='outside',
                showline=True,
                type="log",
                minor=dict(ticks="inside",  showgrid=True)
            )
            if clear_show:
                fig_psd.update_layout(
                    plot_bgcolor='white'
                )
                fig_psd.update_xaxes(
                    linecolor='black',
                    gridcolor='lightgrey'
                )
                fig_psd.update_yaxes(
                    linecolor='black',
                    gridcolor='lightgrey'
                )
            fig_psd.show()
        return fits

    def show_height(self):
        """
        Displays the height map.

        This method generates a plot of the height map and displays it with proper
        axis labels.

        """
        fig = imshow(self.height, title=f"Height map<br><sup>{self.filepath}</sup>")
        fig.update_xaxes(
            title=r"X (m)", )
        fig.update_yaxes(
            title=r"Y (m)", )
        fig.show()


if __name__ == "__main__":
    m = LegendreMap([[0, 0, 1e-8, 5e-9],
                     [0, 2e-9, 0, 0],
                     [5e-9, 1e-9, 2e-9, 0]], 100, 50)
    # m = LegendreMap(deg=5, width=100, height=50)

    m.show_base()
    m.show_map()
