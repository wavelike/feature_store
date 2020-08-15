import os
from pathlib import Path

import pandas as pd

# In the challenge, you are predicting item sales at stores in various locations for two 28-day time periods.
#
# Target: 'sales' column
#
# Time period: 2011-01-29 to 2016-06-19
#
# 3 data sources:
# - data_calendar: date related features
#     - Daily resolution
#     - Keys: date, wm_yr_wk
# - data_sales_train: historical daily unit sales per product and store
#     - Daily resolution
#     - Keys: store_id, item_id
# - data_sell_prices: Price of the products sold per store and date
#     - Weekly averaged resolution
#     - Keys: store_id, item_id, wm_yr_wk


def get_data(keep_day_ratio, n_item_ids):
    # Download data from: https://www.kaggle.com/c/m5-forecasting-accuracy/data and store under the 'data' directory

    data_dir = Path(os.path.join(os.getcwd(), 'data', 'm5-forecasting-accuracy'))

    ####
    # Load 'calendar.csv'
    file_calendar = os.path.join(data_dir, 'calendar.csv')
    data_calendar = pd.read_csv(file_calendar)
    data_calendar['date'] = pd.to_datetime(data_calendar['date'])

    ####
    # Load 'sales_train_validation.csv'
    # Contains the historical daily unit sales data per product and store [d_1 - d_1913]
    # row_id | item_id | department_id | category_id | store_id | state_id | unit_sale_data d_1 - d_1913
    # Contains the historical daily unit sales data per product and store
    file_sales_train_validation = os.path.join(data_dir, 'sales_train_validation.csv')
    data_sales = pd.read_csv(file_sales_train_validation)

    ####
    # Load 'sell_prices.csv'
    #Contains information about the price of the products sold per store and date.
    file_sell_prices = os.path.join(data_dir, 'sell_prices.csv')
    data_sell_prices = pd.read_csv(file_sell_prices)

    # filter out days in order to reduce dataset so it fits into a single laptops RAM
    if keep_day_ratio is not None:
        non_day_columns = [col for col in data_sales.columns if 'd_' not in col]
        day_columns = [col for col in data_sales.columns if 'd_' in col]
        day_columns = day_columns[:int(len(day_columns) * keep_day_ratio)]
        data_sales = data_sales[non_day_columns + day_columns]

    # filter out items in order to reduce dataset so it fits into a single laptops RAM
    if n_item_ids is not None:
        keep_item_ids = data_sales['item_id'].unique()[:n_item_ids]
        data_sales = data_sales[data_sales['item_id'].isin(keep_item_ids)]

    # Transform sales data from wide in a time-series like long format via 'pd.melt'
    keep_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    data_sales_ts = pd.melt(data_sales,
                            id_vars=keep_cols,
                            var_name='day',
                            value_name='sales'
                            )

    # Convert the integer 'day' column into a proper datetime type 'date' column
    data_sales_ts['day_since_start_date'] = data_sales_ts['day'].apply(lambda x: int(x[2:])) - 1
    data_sales_ts['day_since_start_date_timedelta'] = pd.to_timedelta(data_sales_ts['day_since_start_date'], unit='d')
    data_sales_ts['date'] = pd.Timestamp("2011-01-29") + data_sales_ts['day_since_start_date_timedelta']
    data_sales_ts = data_sales_ts.drop(columns=['day', 'day_since_start_date', 'day_since_start_date_timedelta'])

    # merge sales data and date features
    data_full = data_sales_ts.merge(data_calendar.drop(columns=['d']), how='left', on=['date'])

    # merge data_full and sell_prices per week
    data_full = data_full.merge(data_sell_prices, how='left', on=['item_id', 'store_id', 'wm_yr_wk'])
    data_full = data_full.rename(columns={'sell_price': 'sell_price_per_week'})

    return data_full

