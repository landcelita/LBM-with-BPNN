from typing import List, Tuple
import numpy as np
import pygrib
import datetime as dt
import os
from dotenv import load_dotenv

load_dotenv(override=True)
DATA_DIR = os.getenv("DATA_DIR")

def generate_dates() -> Tuple[List[dt.datetime], List[dt.datetime]]:
    since_year = 2011
    since_month = 7
    since_day = 1
    years = 10
    days_per_year = 92
    x_hour = 0
    forcast_period_hour = 3

    res_x, res_y = [], []

    for yr in range(since_year, since_year + years):
        x_datetime_from = dt.datetime(yr, since_month, since_day, x_hour)
        res_x += [x_datetime_from + dt.timedelta(days=i) for i in range(days_per_year)]
        
        y_datetime_from = dt.datetime(yr, since_month, since_day, x_hour) + dt.timedelta(hours=forcast_period_hour)
        res_y += [y_datetime_from + dt.timedelta(days=i) for i in range(days_per_year)]

    return res_x, res_y

res_x, _ = generate_dates()
m = 0.0
for date in res_x:
    grbs = pygrib.open(DATA_DIR + date.strftime("%Y") + "/" + date.strftime("%Y%m%d%H") + ".grib2")
    u_hori = grbs.select()[1].values
    u_vert = grbs.select()[2].values
    m = max(np.max(np.abs(u_hori)), np.max(u_vert), m)

print(m) # 47くらいが最高風速だったので、無次元化する入力速度としては100くらいで割れば良さそう