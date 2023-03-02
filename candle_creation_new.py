import pandas as pd
import numpy as np
import datetime
import os

#global variables

market_start_time = datetime.time(9,15,0)
market_end_time = datetime.time(15,29,0)

def get_all_file_paths(index_folder_path ,data_type = "SPOT",start_year = 2016, end_year=2022):
    """
    This function returns a list of particular data_type file_paths over a period of time 
    """

    date_wise_files = []                                            ## In this list we will append all the file_paths which will be used for continuos dataset creation (a DATAFRAME)

    for year in range(start_year,end_year+1):                       #Iterating over all the year folders
        year_dir = index_folder_path+'\\'+str(year)+'\\'        
        for path_i in os.listdir(year_dir):                         #Iterating over all month folders  for a particular year
            month_dir = os.path.join(year_dir,path_i)
            for path_j in os.listdir(month_dir):                     #Iterating over SPOT,FUT,OPT folders in a month depending on the data_type input
                if path_j == data_type:
                    spot_dir = os.path.join(month_dir,path_j)
                    for file_name in os.listdir(spot_dir):                         #Iterating over date wise csv files available over a month
                        date_wise_files.append(os.path.join(spot_dir,file_name))

    return date_wise_files

def get_dataset(files,data_type="SPOT"):
    df_list = []

    if data_type == "SPOT":
        columns = ['Date','Time','Open','High','Low','Close']
    else:
        columns = ['Date','Time','Open','High','Low','Close','Volume']
        
    for file_path in files:
        df = pd.read_csv(file_path,index_col=0,parse_dates=True,usecols=columns)
        df_list.append(df)

    total_data = pd.concat(df_list,axis=0,ignore_index=False)

    total_data['Time'] = total_data['Time'].apply(lambda x:(pd.to_datetime(x)-pd.to_timedelta("59S")).time())
    time_vals = total_data['Time'].between(market_start_time,market_end_time)
    total_data = total_data[time_vals]

    total_data['time_stamp'] = total_data.apply(lambda x: x.name.replace(hour = x.Time.hour, minute = x.Time.minute),axis=1)
    total_data.sort_values(by='time_stamp',inplace=True)
    total_data.set_index('time_stamp',inplace=True)
    
    return total_data

def get_resampled_data(total_data,time_frame="60T",data_type="SPOT"):
    """
    Generate candle-stick of passed input time-frame over a  minute-wise dataframe 
    """

    if data_type == "SPOT":
        ohlc = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
                }
    else:
        ohlc = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume' : 'sum'

        }

    total_data['day'] = total_data.apply(lambda x: x.name.date(),axis=1)
    grouped_data = total_data.groupby('day')                                                #Grouping dataset on a date wise basis
    df_list = []

    for day,day_wise_data in grouped_data:                                                  #Iterating over grouped data and creating candles on a date-wise basis
        resampled_data = day_wise_data.resample(time_frame,origin = 'start').agg(ohlc)      # resampled data for a day
        df_list.append(resampled_data)
        
    data = pd.concat(df_list,ignore_index=False)                                            # Merging all the date-wise candles across the main - dataset

    return data 
