import datetime as dt

# 入出力の取り回しは分離するべき?
def generate_train_date():
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

def generate_test_date():
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
    dt = 10
    dx = 5600
    
    train_datetimes_x, train_datetimes_y = generate_train_date()
    test_datetimes_x, test_datetimes_y = generate_test_date()

    
    
if __name__ == "__main__":
    main()