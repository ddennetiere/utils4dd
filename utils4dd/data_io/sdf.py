# sdf_reader.py
import struct
import numpy as np
from pathlib import Path


def data_type_info(dtype_code):
    """Map C++ SdfDataType code to numpy dtype and byte size"""
    mapping = {
        0: ('<f8', 8, 'double'),
        1: ('<f4', 4, 'float'),
        2: ('<i4', 4, 'int32'),
        3: ('<i2', 2, 'int16'),
        4: ('<i1', 1, 'int8'),
        5: ('<u4', 4, 'uint32'),
        6: ('<u2', 2, 'uint16'),
        7: ('<u1', 1, 'uint8'),
    }
    return mapping.get(dtype_code, ('<u1', 1, 'unknown'))

def load_sdf(path, verbose=0, instrument="WLI"):
    """Reads SDF header + data block"""
    assert instrument in ["WLI", "MINT"], "Currently only WLI and MINT instrument SDF files are supported"
    header_reader = {"WLI": read_WLI_sdf_header, "MINT":read_MINT_sdf_header}[instrument]
    print(f"Loading SDF file from {path} for instrument {instrument}")
    with open(path, "rb") as f:
        hdr, hdr_size = header_reader(f, verbose=verbose)
        np_dtype, typelen, name = data_type_info(hdr["DataType"])
        nvals = hdr["X"] * hdr["Y"]
        if verbose:
            print(f"Header size ≈ {hdr_size=} bytes")
            print(f"File size = {path.stat().st_size} bytes")
            print(f"Data type = {name=} ({np_dtype=}), pixels = {hdr['X']}×{hdr['Y']} = {nvals=} {name}")
            print(f"Expected data block = {nvals * typelen} bytes")
    # Find possible data start offsets that fit cleanly in file
    np_dtype, type_len, dtype_name = data_type_info(hdr["DataType"])
    nx, ny = hdr["X"], hdr["Y"]
    data_bytes = nx * ny * type_len
    
     # compute ideal header size so data fits exactly
    ideal_hdr_size = path.stat().st_size - data_bytes
    if ideal_hdr_size > hdr_size:
        print(f"Adjusting header size from {hdr_size} → {ideal_hdr_size} bytes (fits data perfectly)")
        hdr_size = ideal_hdr_size

    with open(path, "rb") as f:
        f.seek(hdr_size)
        if verbose:
            print(f"Seeking to data offset {f.tell()} bytes")

        arr = np.fromfile(f, dtype=np_dtype, count=nx * ny) * hdr["Zscale"]

    arr = arr.reshape((ny, nx))
    if verbose:
        print(f"Loaded {nx}×{ny} {dtype_name} array from offset {hdr_size}")
    return hdr, arr, hdr_size


def read_WLI_sdf_header(f, verbose=0):
    """
    Reads SDF header according to Optocat-like format (bBCR-1.1)
    These sdf files are the ouput of the white light interferometer of SOLEIL's LMO
    Returns (header_dict, header_size_bytes)
    """
    header_fmt = (
        "8s"     # fileID
        + "9s"  # manufID
        + "13s"  # timeCreate
        + "12s"  # timeModified
        + "HH"   # X, Y (unsigned short)
        + "4d"   # Xscale, Yscale, Zscale, Zresolution
        + "i"    # CheckType,  
        + "H"    # DataType
    )
    header_size = struct.calcsize("<" + header_fmt)
    print(f"---header format size: {header_size} bytes---")

    raw = f.read(header_size)

    fields = struct.unpack("<" + header_fmt, raw[:header_size])

    hdr = {
        "fileID": fields[0].decode('ascii', errors='ignore').strip('\x00'),
        "manufID": fields[1].decode('ascii', errors='ignore').strip('\x00'),
        "timeCreate": fields[2].decode('ascii', errors='ignore').strip('\x00'),
        "timeModified": fields[3].decode('ascii', errors='ignore').strip('\x00'),
        "Xscale": fields[6],
        "Yscale": fields[7],
        "Zscale": fields[8],
        "Zresolution": fields[9],
        "X": fields[4],
        "Y": fields[5],
        "CheckType": fields[10],
        "DataType": fields[11],
        "raw_header_bytes": header_size
    }
    if verbose:
        print("SDF Header Info:")
        for k, v in hdr.items():
            print(f"  {k}: {v}")

    return hdr, len(raw)

def read_MINT_sdf_header(f,  verbose=0):
    """
    Reads SDF header according to Optocat-like format (bBCR-1.1)
    These sdf files are the ouput of the MINT interferometer of SOLEIL's LMO
    Returns (header_dict, header_size_bytes)
    """
    header_fmt = (
    "8s"     # fileID
    + "10s"  # manufID
    + "12s"  # timeCreate
    + "12s"  # timeModified
    + "HH"   # X, Y (unsigned short)
    + "4d"   # Xscale, Yscale, Zscale, Zresolution
    + "3B"   # CheckType, Compression, DataType, 
    + "H"     # NumDataSet, 
    + "B"     # NanPresent
    )
    header_size = struct.calcsize("<" + header_fmt)
    raw = f.read(header_size)
    fields = struct.unpack_from("<" + header_fmt, raw[:header_size])
    hdr = {
        "fileID": fields[0].decode("ascii", errors="ignore").strip("\x00"),
        "manufID": fields[1].decode("ascii", errors="ignore").strip("\x00"),
        "timeCreate": fields[2].decode("ascii", errors="ignore").strip("\x00"),
        "timeModified": fields[3].decode("ascii", errors="ignore").strip("\x00"),
        "X": fields[4],
        "Y": fields[5],
        "Xscale": fields[6],
        "Yscale": fields[7],
        "Zscale": fields[8],
        "Zresolution": fields[9],
        "Compression": fields[10],
        "DataType": 1,#fields[11], # TODO: no idea why MINT SDF always shows DataType=3 in header but data is actually stored as float
        "CheckType": fields[12],
        "NumDataSet": fields[13],
        "NanPresent": fields[14],
        "raw_header_bytes": header_size
    }
    if verbose:
        print("SDF Header Info:")
        for k, v in hdr.items():
            print(f"  {k}: {v}")
    return hdr, header_size

# Example usage
if __name__ == "__main__":
    path = Path(r"C:\Users\dennetiere\Downloads\SESO_LC_pt6.sdf")
    hdr, data, offset = load_sdf(path, verbose=1, instrument="WLI")
    print(f"Data stats: min={np.nanmin(data):.2g} m, max={np.nanmax(data):.2g} m, mean={np.nanmean(data):.2g} m, std={np.nanstd(data):.2g} m")

    import matplotlib.pyplot as plt
    # threshold the data so as to limit the colorscale to the center 80% values of pixels
    vmin = np.nanpercentile(data, 2)
    vmax = np.nanpercentile(data, 98)
    print(f"Displaying data with vmin={vmin}, vmax={vmax}")
    
    plt.imshow(data, cmap='viridis',vmin=vmin, vmax=vmax)
    plt.title(f"SDF data preview from {path.name}")
    plt.colorbar()

    path2 = Path(r"E:\MINT\sdf data\image_008.sdf")
    
    hdr, data, offset = load_sdf(path2, verbose=1, instrument="MINT")
    print(f"Data stats: min={np.nanmin(data):.2g} m, max={np.nanmax(data):.2g} m, mean={np.nanmean(data):.2g} m, std={np.nanstd(data):.2g} m")

    import matplotlib.pyplot as plt
    # threshold the data so as to limit the colorscale to the center 80% values of pixels
    vmin = np.nanpercentile(data, 1)
    vmax = np.nanpercentile(data, 99)
    print(f"Displaying data with vmin={vmin}, vmax={vmax}")
    
    plt.figure()
    plt.imshow(data, cmap='viridis',vmin=vmin, vmax=vmax)
    plt.title(f"SDF data preview from {path2.name}")
    plt.colorbar()
    plt.show()
