import xarray
from utils4dd.data_io.HeightMap import HeightMap


class XarrayFile(HeightMap):
    def __init__(self, filename):
        super().__init__()
        self.filepath = filename

    def getHeight(self):
        self.height = xarray.open_dataarray(self.filepath)
        return self.height

    def getPixel(self):
        self.getHeight()
        return float(self.height.x[1] - self.height.x[0])