from re import T
from typing import List, Tuple
import repository.repo as repo
import datetime as dt
import learnableLBM.learnableLBM as LBM
import numpy as np

# 入出力の取り回しは分離するべき?
def generate_train_date() -> Tuple[List[dt.datetime], List[dt.datetime]]:
    since_year = 2011
    since_month = 7
    since_day = 1
    years = 9
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

def generate_test_date() -> Tuple[List[dt.datetime], List[dt.datetime]]:
    since_year = 2020
    since_month = 7
    since_day = 1
    years = 1
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

def main():
    train_datetimes_x, train_datetimes_y = generate_train_date()
    test_datetimes_x, test_datetimes_y = generate_test_date()

    u_vert, _, _ = repo.fetchdata(train_datetimes_x[0])
    lbm = LBM.LearnableLBM(u_vert.shape[0], u_vert.shape[1])

    err_u_vert = err_u_hori = du_vert = du_hori = 0.0
    # このへんのLBMに対する処理はもしかしたら後で切り出すかも
    for i in range(10):
        for j in range(len(train_datetimes_x)):
            u_vert_x_nparr, u_hori_x_nparr, pressure_x_nparr = repo.fetchdata(train_datetimes_x[j])
            u_vert_y_nparr, u_hori_y_nparr, _ = repo.fetchdata(train_datetimes_y[j])
            u_vert_x_nparr *= -0.01 # 下向き正
            u_hori_x_nparr *= 0.01 # この定数の根拠はmax_wind_speed.py参照
            u_vert_y_nparr *= -0.01
            u_hori_y_nparr *= 0.01
            u_vert_y = LBM.pyarr2d(u_vert_y_nparr, 2, 2) # この2がマジックナンバーでわかりずらそう
            u_hori_y = LBM.pyarr2d(u_hori_y_nparr, 2, 2)
            
            u_vert_got, u_hori_got = lbm.forward(u_vert_x_nparr, u_hori_x_nparr, pressure_x_nparr)
            lbm.backward(0.25, u_vert_y, u_hori_y)

            forb = u_vert_got.forbidden_at
            err_u_vert += 100.0 * np.mean(np.abs(u_vert_got.arr[forb[0]:-forb[0], forb[1]:-forb[1]] - u_vert_y_nparr[forb[0]:-forb[0], forb[1]:-forb[1]]))
            err_u_hori += 100.0 * np.mean(np.abs(u_hori_got.arr[forb[0]:-forb[0], forb[1]:-forb[1]] - u_hori_y_nparr[forb[0]:-forb[0], forb[1]:-forb[1]]))
            du_vert += 100.0 * np.mean(np.abs(u_vert_x_nparr[forb[0]:-forb[0], forb[1]:-forb[1]] - u_vert_y_nparr[forb[0]:-forb[0], forb[1]:-forb[1]]))
            du_hori += 100.0 * np.mean(np.abs(u_hori_x_nparr[forb[0]:-forb[0], forb[1]:-forb[1]] - u_hori_y_nparr[forb[0]:-forb[0], forb[1]:-forb[1]]))

            if j % 10 == 9:
                print(f"epoch {i + 1}, {j + 1}", flush=True)
                print(f"ave diff vert: {'{:.5f}'.format(err_u_vert / 10.0)} m/s, ave diff hori: {'{:.5f}'.format(err_u_hori / 10.0)} m/s", flush=True)
                print(f"actual diff vert: {'{:.5f}'.format(du_vert / 10.0)} m/s, actual diff hori: {'{:.5f}'.format(du_hori / 10.0)} m/s", flush=True)
                print(f"1st / 2nd: {'{:.5f}'.format(err_u_vert / du_vert)}, {'{:.5f}'.format(err_u_hori / du_hori)}", flush=True)
                

                err_u_vert = err_u_hori = du_vert = du_hori = 0.0


    
if __name__ == "__main__":
    main()