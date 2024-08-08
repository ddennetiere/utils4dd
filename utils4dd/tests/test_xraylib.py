from data_io.xraylib_func import reflectivity_bulk, element_ratios
import xraylib
import timeit
import xarray as xr
import plotly.express as px
import numpy as np
from scipy.constants import degree


if __name__ == "__main__":
    def test():
        E = np.linspace(30, 4500, 1000)
        theta = np.linspace(1, 10, 100) * degree
        R = []
        for th in theta:
            R.append(reflectivity_bulk(E, th, "Si"))
        array = xr.DataArray(R, (("Theta", theta / degree), ("E", E)))
        return array


    def test_mesh():
        E = np.linspace(30, 4500, 1000)
        theta = np.linspace(1, 10, 100) * degree
        R = (reflectivity_bulk(E, theta, "Si", mesh=True))
        array = xr.DataArray(R, (("Theta", theta / degree), ("E", E)))
        return array


    fig = px.line(x=np.linspace(30, 1000, 100),
                  y=reflectivity_bulk(np.linspace(30, 1000, 100), 2 * degree, "Si", density=-1))
    fig.show()
    print(timeit.timeit(test, number=10))
    print(timeit.timeit(test_mesh, number=10))
    nistCompounds = xraylib.GetCompoundDataNISTList()
    for i in range(len(nistCompounds)):
        cdn = xraylib.GetCompoundDataNISTByIndex(i)
        if "Boron Carbide" in nistCompounds[i]:
            print("\tCompound {}: {}".format(i, nistCompounds[i]))
            print("\tDensity: {}".format(cdn['density']))
            for i in range(cdn['nElements']):
                print(
                    f"\t\tElement {xraylib.AtomicNumberToSymbol(cdn['Elements'][i])}: {cdn['massFractions'][i] * 100.0} %")
    formula = 'B4C'
    ratios = element_ratios(formula)
    for element, ratio in ratios.items():
        print(f"{element}: {ratio}")
