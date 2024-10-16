import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

DATA_PATH = 'data/data_stream-oper.nc'

data = xr.open_dataset(DATA_PATH)
data