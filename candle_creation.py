import pandas as pd
import datetime
import numpy as np

market_start_time = datetime.time(9,15,0)
market_end_time = datetime.time(15,29,0)

def create_discrete_candles(data,date,freq):
    start = str(date)+" "+"09:15:00"
    end = str(date)+" "+"15:29:00"
    candle_ts = pd.date_range(start = start, end = end, freq = freq)
    date_candles = []
    for i in range(len(candle_ts)):
        try:
            sub_df = data[data['time_stamp'].between(candle_ts[i],candle_ts[i+1],inclusive="left")]
        except IndexError:
            sub_df = data[data['time_stamp'].between(candle_ts[i],end)]
        if not sub_df.empty:
            candle={}
            candle['Date'] = sub_df.iloc[0].name
            candle['Time'] = sub_df.iloc[0]['Time']
            candle['Open'] = sub_df.iloc[0]['Open']
            candle['High'] = sub_df['High'].max()
            candle['Low'] = sub_df['Low'].min()
            candle['Close'] = sub_df.iloc[-1]['Close']
            candle['time_stamp'] = sub_df.iloc[0]['time_stamp']
            date_candles.append(candle)
    return pd.DataFrame(date_candles).set_index('Date')


def get_Discrete_candles(df,freq):
    date_wise_df = []
    for datewise_data in df.groupby(by='Date'):
        date = datewise_data[0].date()
        data = datewise_data[1]
        date_wise_df.append(create_discrete_candles(data,date,freq))
    return pd.concat(date_wise_df,axis=0)
