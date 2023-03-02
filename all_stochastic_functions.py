import numpy as np
import pandas as pd
import datetime 

def interval_diff(data,base_col,op_col,interval = 1):
    data[op_col]=data[base_col].diff(interval)
    return data

def classification_1(data,base_col,op_col = 'classification_1'):
    data.loc[data[base_col].between(float("-inf"),-1,'right'), op_col] = '<-1'
    data.loc[data[base_col].between(-1,-0.5,'right'),op_col] = 'between -0.99 and -0.5'
    data.loc[data[base_col].between(-0.5,0,'both'),op_col] = 'between -0.5 and 0'
    data.loc[data[base_col].between(0,0.5,'both'),op_col] = 'between 0 and 0.5'
    data.loc[data[base_col].between(0.5,1,'left'),op_col] = 'between 0.5 and 0.99'
    data.loc[data[base_col].between(1,float("inf"),'left'),op_col] = '>1'
    return data

def classification_2(data,base_col,op_col='classification_2'):
    data[op_col] = np.where(data[base_col]<0,'Red','Green')
    data.loc[data[base_col].isna(),op_col] = np.NaN
    return data

def classification_3(data,base_col, op_col = 'classification_3'):
    data[op_col] = np.where(data[base_col] > data[base_col].shift(1),'Increasing','Decreasing')
    data.loc[data[base_col].isna(),op_col] = np.NaN
    return data

def classification_4(data,base_col,op_col='classification_4'):
    data[op_col] = np.where(data[base_col] > data[base_col].shift(1),'Increasing','Decreasing')
    data.loc[data[base_col].isna(),op_col] = np.NaN
    return data

def classification_6(data,col_1,col_2,na_col,op_col='classification_6'):
    data[op_col] =  np.where(data[col_1]>data[col_2],'Green','Red')
    data.loc[data[na_col].isna(),op_col] = np.NaN
    return data

def binning_EMA(data,base_col,bin_length = 20,op_col = 'EMA_bins_20'):
    step = bin_length
    bins = [i for i in range(0,100+step,step)]
    labels = [f"{bins[i-1]+1}-{bins[i]}" if i>1 else f"<{bins[i]}" for i in range(1,len(bins)) ]
    data[op_col] = pd.cut(x=data[base_col],bins=bins,labels=labels)
    return data

def classification_2_bin_wise(data,base_col,bin_col,new_col='Classification-7'):
    
    data[new_col] = data[base_col]

    first = data[new_col].first_valid_index()
    start_row_num = data.index.get_loc(first)+1
    start_index = data.index[start_row_num]

    for index,row in data[start_index:].iterrows():

        row_num = data.index.get_loc(index)
        prev_index = data.index[row_num-1]


        if data.loc[index][bin_col] == data.loc[prev_index][bin_col]:      ##Checking if bin_value is same as prev row
            if data.loc[index][new_col] != data.loc[prev_index][new_col]: ##Changing current val to prev if bins are matching
                data.at[index, new_col] = data.at[prev_index, new_col]

    return data

def moment_on_close(cycle,base_col = None):
    last_close = cycle.iloc[-1]['Close']
    first_close = cycle.iloc[0]['Close']
    moment = (last_close/first_close)-1
    no_of_bars = cycle.shape[0]
    return moment*100,no_of_bars

def moment_on_High_Low(cycle,base_col):
    
    first_close = cycle.iloc[0]['Close']
    no_of_bars = cycle.shape[0]
    color = cycle.iloc[0][base_col]
    
    if color == 'Red':
        low = cycle['Low'].min()
        moment = (low/first_close) - 1
    else:
        high = cycle['High'].max()
        moment = (high/first_close) - 1
    
    return moment*100,no_of_bars

def percentage_moment(data,base_col,moment_basis='Close',op_col='percent_moment',no_of_bar_col='no_of_bars'):
    
    if moment_basis == 'Close':
        Moment = moment_on_close
    elif moment_basis == 'H/L':
        Moment = moment_on_High_Low
    
    cycle_change = abs(data[base_col].replace(['Red','Green'],[0,1]).diff())
    cycle_change[0] = 0
    data = data.assign(cycle_change = cycle_change)
    moment = pd.Series(dtype='float64',index=data.index)
    no_of_bars = pd.Series(dtype='int64',index=data.index)
    
    cycle_start_index = data.index[0]
    for index,row in data.iterrows():
        if row['cycle_change'] == 1:
            prev_row_num = data.index.get_loc(index)-1
            cycle_end_index = data.iloc[prev_row_num].name
            cycle = data.loc[cycle_start_index:cycle_end_index]
            moment.loc[cycle_end_index], no_of_bars.loc[cycle_end_index]  = Moment(cycle,base_col)
            cycle_start_index = index
    
    if moment_basis == 'Close':
        #data = data.assign(moment_on_close = moment)
        data[op_col] = moment
    elif moment_basis == 'H/L':
        # data = data.assign(moment_on_HL = moment)
        data[op_col] = moment
        
    # data = data.assign(no_of_bars = no_of_bars)
    data[no_of_bar_col] = no_of_bars
    data = data.drop(columns=['cycle_change'])   ## dropping the cycle_change 
    return data

def generate_green_fractal_average(data,cross_limit,thresh_cross_indices,op_col='fractal_avg_green'):
    
    start_indices = data[data['level_change']==1].index
    end_indices = data[data['level_change']==-1].index

    bin_start_time = pd.Series(dtype="datetime64[ns]",index=data.index)
    bin_end_time = pd.Series(dtype="datetime64[ns]",index=data.index)

    for i,j in zip(start_indices,end_indices):
        bin_start_time[i] = i
        bin_end_time[i] = j
        #print(i,'|',j)

    data['bin_start_time'] = bin_start_time
    data['bin_end_time'] = bin_end_time
    
    
    for i in range(0,len(thresh_cross_indices)-1):
        current_bin_start = thresh_cross_indices[i]
        next_bin_start = thresh_cross_indices[i+1]

        between_data = data.loc[current_bin_start:next_bin_start]

        if (between_data['%_K_EMA']<cross_limit).any():
            pass
        else:
            data.at[next_bin_start,'bin_start_time'] = data.loc[current_bin_start]['bin_start_time']
            
    unique_bin_start_ts = data['bin_start_time'].unique()
    unique_bin_start_ts = unique_bin_start_ts[~np.isnan(unique_bin_start_ts)]
    
    fractal_avg = pd.Series(dtype="float64",index=data.index)

    for time_stamp in unique_bin_start_ts:
        bin_starts_df = data[data['bin_start_time']==time_stamp]
        if bin_starts_df.shape[0]==1:
            ##Creating a sub_df for non_continuos bins to take fractal avg for that bin period only
            sub_df_start = data.loc[time_stamp]['bin_start_time']
            sub_df_end = data.loc[time_stamp]['bin_end_time']
            sub_df = data.loc[sub_df_start:sub_df_end]

            if len(sub_df)>1:            ##Same condition as sub_df_start != sub_df_end
                sub_df = sub_df.iloc[:-1]
            else:
                #For candles where single EMA-bin occurs whith no surrounding similar bins
                pass

            sub_df_fractal_avg = sub_df[sub_df['is_high']==1]['High'].mean()
            if np.isnan(sub_df_fractal_avg):
                sub_df_fractal_avg = 0

            fractal_avg[time_stamp] = sub_df_fractal_avg

        else:
            fractal_values = []
            for index,row in bin_starts_df.iterrows():
                sub_df_start = index
                sub_df_end = row['bin_end_time']
                sub_df = data.loc[sub_df_start:sub_df_end]

                if len(sub_df)>1:         ##Same condition as sub_df_start != sub_df_end
                    sub_df = sub_df.iloc[:-1]
                else:
                    pass

                sub_df_fractal_values = sub_df[sub_df['is_high']==1]['High'].to_list()
                fractal_values.extend(sub_df_fractal_values)

            if len(fractal_values)>0:
                fractal_avg[time_stamp] = sum(fractal_values)/len(fractal_values)
            else:
                ##If we dont have any high or low fractals in the data
                fractal_avg[time_stamp] = 0

            print(time_stamp)

    # data['fractal_avg'] = fractal_avg
    data[op_col] = fractal_avg
    data = data.drop(columns=['bin_start_time','bin_end_time'])
    return data

def generate_red_fractal_average(data,cross_limit,thresh_cross_indices,op_col='fractal_avg_red'):
    
    start_indices = data[data['level_change']==1].index
    end_indices = data[data['level_change']==-1].index

    bin_start_time = pd.Series(dtype="datetime64[ns]",index=data.index)
    bin_end_time = pd.Series(dtype="datetime64[ns]",index=data.index)

    for i,j in zip(start_indices,end_indices):
        bin_start_time[i] = i
        bin_end_time[i] = j
        #print(i,'|',j)

    data['bin_start_time_red'] = bin_start_time
    data['bin_end_time_red'] = bin_end_time
    
    for i in range(0,len(thresh_cross_indices)-1):
        current_bin_start = thresh_cross_indices[i]
        next_bin_start = thresh_cross_indices[i+1]

        between_data = data.loc[current_bin_start:next_bin_start]

        if (between_data['%_K_EMA']>cross_limit).any():
            pass
        else:
            data.at[next_bin_start,'bin_start_time_red'] = data.loc[current_bin_start]['bin_start_time_red']
            
    unique_bin_start_ts = data['bin_start_time_red'].unique()
    unique_bin_start_ts = unique_bin_start_ts[~np.isnan(unique_bin_start_ts)]
    
    fractal_avg = pd.Series(dtype="float64",index=data.index)

    for time_stamp in unique_bin_start_ts:
        bin_starts_df = data[data['bin_start_time_red']==time_stamp]
        if bin_starts_df.shape[0]==1:
            ##Creating a sub_df for non_continuos bins to take fractal avg for that bin period only
            sub_df_start = data.loc[time_stamp]['bin_start_time_red']
            sub_df_end = data.loc[time_stamp]['bin_end_time_red']
            sub_df = data.loc[sub_df_start:sub_df_end]

            if len(sub_df)>1:            ##Same condition as sub_df_start != sub_df_end
                sub_df = sub_df.iloc[:-1]
            else:
                #For candles where single EMA-bin occurs whith no surrounding similar bins
                pass

            sub_df_fractal_avg = sub_df[sub_df['is_low']==1]['Low'].mean()
            if np.isnan(sub_df_fractal_avg):
                sub_df_fractal_avg = 0

            fractal_avg[time_stamp] = sub_df_fractal_avg

        else:
            fractal_values = []
            for index,row in bin_starts_df.iterrows():
                sub_df_start = index
                sub_df_end = row['bin_end_time_red']
                sub_df = data.loc[sub_df_start:sub_df_end]

                if len(sub_df)>1:         ##Same condition as sub_df_start != sub_df_end
                    sub_df = sub_df.iloc[:-1]
                else:
                    pass

                sub_df_fractal_values = sub_df[sub_df['is_low']==1]['Low'].to_list()
                fractal_values.extend(sub_df_fractal_values)

            if len(fractal_values)>0:
                fractal_avg[time_stamp] = sum(fractal_values)/len(fractal_values)
            else:
                ##If we dont have any high or low fractals in the data
                fractal_avg[time_stamp] = 0

            print(time_stamp)

    # data['fractal_avg_red'] = fractal_avg
    data[op_col] = fractal_avg
    data = data.drop(columns=['bin_start_time_red','bin_end_time_red'])
    return data
    
def ema_bin_wise_fractal_avg(data,bin_col,bin_val,green_fractal_col_name = 'fractal_avg_green',red_fractal_col_name='fractal_avg_red'):
    
    # data['level'] = np.where(data['EMA_bins']==bin_val,1,0)   ##2
    data['level'] = np.where(data[bin_col]==bin_val,1,0)
    # data.loc[data['EMA_bins'].isna(),'level'] = np.NaN
    data.loc[data[bin_col].isna(),'level'] = np.NaN
    
    data['level_change'] = data['level'] - data['level'].shift()

    first_row = data[data[bin_col]==bin_val]
    if not first_row.empty:
        time_stamp = data[data[bin_col]==bin_val].iloc[0].name
        data.at[time_stamp,'level_change'] = 1

    thresh_cross_indices = data[data['level_change']==1].index 
    cross_limit = 50
    
    data = generate_green_fractal_average(data,cross_limit,thresh_cross_indices,green_fractal_col_name)
    data = generate_red_fractal_average(data,cross_limit,thresh_cross_indices,red_fractal_col_name)

    data = data.drop(columns=['level','level_change'])  ##fropping level and level_change columns
    
    return data

def rolling_z_score(window,moment_col):
    x = window.iloc[-1][moment_col]
    mu = window[moment_col].mean()
    std = window[moment_col].std(ddof=0)
    z_score = (x-mu)/std
    return z_score

def Calc_Z_score(data,base_col,op_col_name):
    Z_Score = pd.Series(dtype="float64",index = data.index)
    z_score_indices = data[data[base_col].notna()].index
    
    start = z_score_indices[0]
    for index in z_score_indices:
        window = data.loc[start:index]
        z_score = rolling_z_score(window,base_col)
        Z_Score.loc[index] = z_score
    #data = data.assign(Z_score = Z_Score)
    data[op_col_name] = Z_Score
    return data


def z_score_2(data,moment_col,bars_col,op_col='z_score_2'):
    
    data['product'] = data[moment_col]*data[bars_col]
    z_score_indices = data[~data['product'].isnull()].index
    data[op_col] = pd.Series(dtype='float64',index=data.index)
    
    numerators,denominators = [],[]
    for index in z_score_indices:
        numerators.append(data.loc[index]['product'])
        denominators.append(data.loc[index][bars_col])
        z_score_2 = sum(numerators)/sum(denominators)
        data.at[index,op_col] = z_score_2
    data=data.drop(columns = 'product')
    return data
    
