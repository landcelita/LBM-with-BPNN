import os
from typing import Tuple
import pygrib
import numpy as np
import datetime as dt
from dotenv import load_dotenv

load_dotenv(override=True)
DATA_DIR = os.getenv("DATA_DIR")

def fetchdata(time: dt.datetime) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grbs = pygrib.open(DATA_DIR + time.strftime("%Y") + "/" + time.strftime("%Y%m%d%H") + ".grib2")
    pressure = grbs.select()[0].values
    u_hori = grbs.select()[1].values
    u_vert = grbs.select()[2].values
    
    return u_vert, u_hori, pressure
