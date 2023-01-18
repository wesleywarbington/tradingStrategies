import pandas as pd
from util import get_data


def compute_portvals(orders_df=None, start_val=100000, commission=9.95, impact=0.005):
    # Step 1
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]
    symbol = orders_df.columns.values[0]

    prices = get_data([symbol], pd.date_range(start_date, end_date), addSPY=True, colname="Adj Close")
    prices.fillna(method="ffill", inplace=True)
    prices.fillna(method="bfill", inplace=True)
    prices = prices.assign(Cash=1.00)

    # Step 2
    trades = prices.copy()
    trades[trades.columns] = 0
    orders_list = orders_df[symbol].tolist()

    for index, order in enumerate(orders_list):
        trades.iloc[index][symbol] += order
        trades.iloc[index]['Cash'] += prices.iloc[index][symbol] * -order

        # Commission
        try:
            trades.iloc[index]['Cash'] -= commission
        except:
            pass
        # Impact
        try:
            impact_fee = abs(prices.iloc[index][symbol] * order * impact)
            trades.iloc[index]['Cash'] -= impact_fee
        except:
            pass

    # Step 3
    holdings = trades.copy()
    holdings[holdings.columns] = 0
    holdings.iloc[0]['Cash'] = start_val

    holdings += trades
    holdings = holdings.cumsum()

    # Step 4
    values = prices * holdings

    # Step 5
    portvals = values.sum(axis=1)
    return portvals