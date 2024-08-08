import pandas as pd


class CarpemFile(pd.DataFrame):
    """
    Class for opening a Carpem file as a pandas.DataFrame
    header is dismissed
    angles are in degrees
    lambda is in nanometer
    E is in eV
    Columns are renamed in an orderly fashion :
    [alpha, E, lambda, [beta<order>, eff<order>, phi<order>]*(2*n_order+1), [eff<order>, rapp<order>]*(2*n_order+1)*n_harm]
    """
    def __init__(self, filename, n_order, n_harm):
        colnames = ["alpha", "E", "lambda"]
        for order in range(-n_order, n_order + 1):
            colnames.append(f"beta{order}")
            colnames.append(f"eff{order}_h1")
            colnames.append(f"phi{order}")
        for harm in range(2, n_harm + 1):
            for order in range(-n_order, n_order + 1):
                colnames.append(f"eff{order}_h{harm}")
                colnames.append(f"rapp{order}_h{harm}")
        data = pd.read_fwf(filename, widths=[14] * len(colnames), header=None, skiprows=35, names=colnames)
        super().__init__(data)

