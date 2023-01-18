import numpy as np


# Bollinger Bands
def get_bbp(price=None, lookback=20):
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2*rolling_std)
    bottom_band = sma - (2*rolling_std)
    bbp = (price-bottom_band)/(top_band-bottom_band)
    return bbp

# RSI
def get_rsi(price=None, lookback=20):
    daily_rets = price.copy()
    daily_rets.values[1:,:] = price.values[1:,:] - price.values[:-1,:]
    daily_rets.values[0,:] = np.nan

    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1*daily_rets[daily_rets < 0].fillna(0).cumsum()
    up_gain = price.copy()
    up_gain.ix[:,:] = 0
    up_gain.values[lookback:,:] = up_rets.values[lookback:,:] - up_rets.values[:-lookback,:]
    down_loss = price.copy()
    down_loss.ix[:,:] = 0
    down_loss.values[lookback:, :] = down_rets.values[lookback:,:] - down_rets.values[:-lookback,:]
    rs = (up_gain/lookback)/(down_loss/lookback)
    rsi = 100-(100/(1+rs))
    rsi.ix[:lookback,:] = np.nan
    rsi[rsi==np.inf] = 100
    return rsi

# EMA, 13 and 21 day cross? (Buy when ema_cross crosses above 1, sell when crosses below 1)
def get_emaCross(price=None):
    ema_13 = price.ewm(span=13).mean()
    ema_21 = price.ewm(span=21).mean()
    ema_cross = ema_13/ema_21
    return ema_cross
