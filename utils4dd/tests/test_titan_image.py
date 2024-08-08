import logging
from utils4dd.data_io.ZygoDat import DatFile
import numpy as np
import sys
import plotly.io as pio
from utils4dd.data_io.HeightMap import LegendreMap
import plotly.express as px

pio.renderers.default = "browser"
logger = logging.getLogger(__name__)
root = logging.getLogger()
root.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
root.addHandler(handler)
# test = "read"
test = "psd"

if test == "read":
    datfile = DatFile()
    datfile.openFile(r"Y:\Mesures-2024\DI-61020_SWING\A Stitched Scans\Stitched_Scan_207_topStripe.dat")
    _ = datfile.getHeight()
    logger.info(f"Height RMS : {datfile.height.std()*1e9} nm")
    datfile.show_height()

if test == "psd":
    datfile = DatFile()
    datfile.openFile(r"D:\Dennetiere\Biblio\Moi\SRI 2024\Residu_Scan221_A10_q6.91416_300mm.dat")
    datfile.info()
    _ = datfile.getHeight()
    logger.info(f"Height RMS : {datfile.height.std()*1e9} nm")
    # datfile.show_height()
    Legendre_model = LegendreMap(deg=6, width=datfile.height.shape[1], height=datfile.height.shape[0])
    residual = Legendre_model.fit(datfile, mask=np.triu(np.ones((6, 6)))[:, ::-1])
    logger.info(f"Detrended height RMS : {residual.std()*1e9} nm")
    logging.debug(f"norms: \n{Legendre_model.check_norms()}")
    # Legendre_model.show_decomp()
    logger.info(f"Result of fit : \n{Legendre_model.coeffs}")
    logger.debug(f"Check : {Legendre_model.check_solution_unicity(datfile.height.values) is None}")
    # Legendre_model.show_map()
    corrected_height = datfile.height.copy()
    corrected_height.values = residual
    # f = px.imshow(corrected_height, title="Residual of fit")
    # f.show()
    datfile.height = corrected_height
    # psdh, freqh = datfile.get_psd(axis=0, show_psd=True, mean="psd")
    _ = datfile.fit_psd(axis=0, cutoffs=[10, 350, 2300], show_fit=True, clear_show=True)
    # psdv, freqv = datfile.get_psd(axis=1, show_psd=True, mean="psd")
    _ = datfile.fit_psd(axis=1, cutoffs=[100, 4000], show_fit=True, clear_show=True)
    # psd, freq = datfile.get_PSD(axis=0, show_psd=True, mean="line")
    # _ = datfile.fit_PSD(axis=0, cutoffs=[50, 10 ** 2.65], show_fit=True, clear_show=True)
