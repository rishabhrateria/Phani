import numpy as np
import pandas as pd
import datetime
import operator

def get_first_n_previous_indices(data,win_len):
    prev_n_red = []
    prev_n_green = []
    
    for index,row in data.iterrows():
        if row.is_high and len(prev_n_green)<win_len:  #Checking if we got previous_n_highs so we can start computation
            prev_n_green.append(index)

        if row.is_low and len(prev_n_red)<win_len:   #Checking if we got previous_n_lows
            prev_n_red.append(index)

        if len(prev_n_green) == win_len and len(prev_n_red) == win_len: #Start index i.es we
            break
    
    return prev_n_red,prev_n_green


def get_n_trailing_averages_new(data,win_len=5):
    
    prev_n_red, prev_n_green = get_first_n_previous_indices(data,win_len)

    if prev_n_red[-1]<prev_n_green[-1]:  ## Start index of calculation
        current_index = prev_n_red[-1]
    else:
        current_index = prev_n_green[-1]
    
    start_row_num = data.index.get_loc(prev_n_green[-1]) + 1
    red_start_index = data.iloc[start_row_num].name

    start_row_num = data.index.get_loc(prev_n_red[-1]) + 1
    green_start_index = data.iloc[start_row_num].name
    
    prev_n_op_avg = pd.Series(dtype="float64", index=data.index) #Initializing a Pandas Series to store trailing avg
    
    prev_n_op_red_avg = pd.Series(dtype="float64", index=data.index)
    prev_n_op_green_avg = pd.Series(dtype="float64", index=data.index)

    for index,row in data.loc[green_start_index:].iterrows():
        if row.is_high:
            prev_n_low_avg = data.loc[prev_n_red]['Low'].mean()
            prev_n_op_red_avg.loc[index] = prev_n_low_avg

        if row.is_low:
            prev_n_red.pop(0)
            prev_n_red.append(row.name)
            
    for index,row in data.loc[red_start_index:].iterrows():
        if row.is_low:
            prev_n_high_avg = data.loc[prev_n_green]['High'].mean()
            prev_n_op_green_avg.loc[index] = prev_n_high_avg

        if row.is_high:
            prev_n_green.pop(0)
            prev_n_green.append(row.name)
    
            
    data = data.assign(prev_n_op_red_avg = prev_n_op_red_avg)
    data = data.assign(prev_n_op_green_avg = prev_n_op_green_avg)
    
    return data

def get_fractal_val_green(is_high,high,low):
    if is_high:
        return high
    else:
        return np.NaN

def get_fractal_val_red(is_low,high,low):
    if is_low:
        return low
    else:
        return np.NaN
    

def compute_abs_diff(is_low,is_high,prev_n_op_red_avg,prev_n_op_green_avg,fractal_val_green,fractal_val_red):
    
    if is_high ==1 and is_low==0:
        return fractal_val_green - prev_n_op_red_avg

    elif is_high == 0 and is_low == 1:
        return fractal_val_red - prev_n_op_green_avg

    elif is_high == 1 and is_low == 1:
        return np.NaN


def compute_percent_growth(is_low,is_high,prev_n_op_red_avg,prev_n_op_green_avg,fractal_val_green,fractal_val_red):
    
    if is_high ==1 and is_low==0:
        return (fractal_val_green/prev_n_op_red_avg -1)*100
    elif is_high == 0 and is_low == 1:
        return (fractal_val_red/prev_n_op_green_avg -1)*100
    elif is_high == 1 and is_low == 1:
            return np.NaN
    

def change_label(label):
    if label == 'A':
        return 'B'
    else:
        return 'A'
    
def generate_levels(data,base_col):
    
    level = pd.Series(dtype='str',index=data.index)
    current_label = 'A'                                 #Assigning label A to the first type of fractal that occurs
    op_map={'is_high':operator.lt,'is_low':operator.gt} #mapping the type of operation depending on the type of fractal
    
    for index,row in data.iterrows():
        if row.is_low ==1  and row.is_high==0 and  row.abs_diff>0:
            start_index = index
            label_map = {'A':'is_high','B':'is_low'}
            level[index] = 'A'
            break
        if row.is_high == 1  and row.is_low == 0 and  row.abs_diff<0:
            start_index = index
            label_map = {'A':'is_low', 'B':'is_high'}
            level[index] = 'A'
            break

        if row.is_high ==1 and row.is_low ==1 :
            if row['neutral_index_is_high'] == 0 and row.abs_diff>0:
                start_index = index
                label_map = {'A':'is_high','B':'is_low'}
                level[index] = 'A'
                break
            if row['neutral_index_is_high'] == 1 and row.abs_diff<0:
                start_index = index
                label_map = {'A':'is_low', 'B':'is_high'}
                level[index] = 'A'
                break

    temp = {'is_high':1,'is_low':0}
    neutral_map = {i:temp[label_map[i]] for i in label_map}



            
    start_loc = data.index.get_loc(start_index)+1
    start_index = data.index[start_loc]
    
    for index,row in data.loc[start_index:].iterrows():
        if row[label_map[current_label]] == 1:                          #We doing the check depending on the previous row level
            opposite_label = change_label(current_label)

            if row[label_map[opposite_label]] == 0:                 # Not a neutral fractal
                if op_map[label_map[current_label]](row[base_col],0):  # checking if the next oppsoite fractal meets the criteria 
                    current_label = change_label(current_label)          #Switching the current label to opposite
                    level[index] = current_label
                else:
                    level[index] = current_label

            else:                                               #Is as neutral fractal
                if row['neutral_index_is_high'] == neutral_map[current_label]:      #Checking for a neutral fractal if it should be computed or not
                    
                    if op_map[label_map[current_label]](row[base_col],0):  # checking if the next oppsoite fractal meets the criteria 
                        current_label = change_label(current_label)          #Switching the current label to opposite
                        level[index] = current_label
                    else:
                        level[index] = current_label

                else:
                    level[index] = current_label

        else:
            level[index] = current_label
            
    data = data.assign(level=level)
    return data,label_map,op_map

def classification_fractal(data,base_col,thresh,op_col = 'CL-1'):
    data.loc[data[base_col].between(float("-inf"),-1*thresh,'right'), op_col] = 'Declining'
    data.loc[data[base_col].between(-1*thresh,thresh,'both'),op_col] = 'Same'
    data.loc[data[base_col].between(thresh,float("inf"),'left'),op_col] = 'Increasing'
    return data

def bin_abs_diff(data,label_map,thresh=0.5):

    new_map = {'A':label_map['B'],'B':label_map['A']}  ##now we are tracking exactly as per labels
    temp = {1:'is_high',0:'is_low'}
    reverse_map =  {new_map[i]:i for i in new_map} 
    neutral_index_map = {reverse_map[temp[i]]:i  for i in temp}

    data['level_change'] = data['level'].apply(lambda x : 1 if x=='A' else 0) 
    data['level_change'] = abs(data['level_change'].diff()) #Finding where cycle change happened
    data=classification_fractal(data,'%_growth',thresh)
    data['CL-1'] = data.apply(lambda row : 'First' if row['level_change'] else row['CL-1'],axis=1)

    data.loc[(data[label_map['A']]==1) & (data[label_map['B']]==0) & (data['level'] == 'A'),'CL-1'] = np.NaN
    data.loc[(data[label_map['B']]==1) & (data[label_map['A']]==0) & (data['level'] == 'B'),'CL-1'] = np.NaN

    data.loc[(data[label_map['A']]==1) & (data[label_map['B']]==1) & (data['level'] == 'B') & (data['neutral_index_is_high']==neutral_index_map['A']),'CL-1'] = np.NaN
    data.loc[(data[label_map['A']]==1) & (data[label_map['B']]==1) & (data['level'] == 'A') & (data['neutral_index_is_high']==neutral_index_map['B']),'CL-1'] = np.NaN

    data.at[data.index[0],'CL-1'] = np.NaN
    
    return data


def is_low_operation(fractal_val,CTA):
    op = (fractal_val/CTA) -1
    return op*100

def is_high_operation(fractal_val,CTA):
    op = (CTA/fractal_val) -1
    return op*100

def step_5_new(data,label_map):
    
    op_map={'is_high':is_high_operation,'is_low':is_low_operation}   #Storing the compute functions of High and Low Fractals
    new_map = {'A':label_map['B'],'B':label_map['A']}                # This mapping is just the opposite of 
    cycles_first_avg = pd.Series(dtype="float64",index=data.index)
    first_avg = np.NaN

    temp = {1:'is_high',0:'is_low'}
    reverse_map =  {new_map[i]:i for i in new_map} 
    neutral_index_map = {i:reverse_map[temp[i]] for i in temp}   ## To check whether step_5 has to to be calculated for 1,1 fractsl

    for index,row in data.iterrows():           ##LOCKING THE FIRST TRAILING AVERAGES OF A LEVEL
        if row['level_change'] ==1:
            if row.is_high == 1 and row.is_low == 0:                     #changed
                first_avg = row['prev_n_op_red_avg']
            elif row.is_low == 1 and row.is_high == 0:
                first_avg = row['prev_n_op_green_avg']
            elif row.is_high == 1 and row.is_low == 1:
                if row['neutral_index_is_high']:
                    first_avg = row['prev_n_op_red_avg']
                else:
                    first_avg = row['prev_n_op_green_avg']
                
            cycles_first_avg[index] = first_avg
        else:
            pass
        
    data = data.assign(CTA=cycles_first_avg)
    
    step_5 = pd.Series(dtype="float64",index=data.index)
    current_label = 'A'
    ## this variable tracks previous trailing avg for fractal
    prev_cycle_avg_map = {'A':data[data['level']=='A'].iloc[0]['CTA'] ,'B':data[data['level']=='B'].iloc[0]['CTA']}  #changed
    
    start_index = data[data['level_change']==1].iloc[0].name

    for index,row in data[start_index:].iterrows():

        if row.level_change == 1:                    ## This is for tracking which value to use for changing CTA used fro computation
            
            if row.is_high == 1 and row.is_low == 0:                           #changed
                prev_cycle_avg_map[row.level] = row['prev_n_op_red_avg']
            elif row.is_low == 1 and row.is_high == 0:
                prev_cycle_avg_map[row.level] = row['prev_n_op_green_avg']
            elif row.is_high == 1 and row.is_low == 1:
                if row['neutral_index_is_high']:
                    prev_cycle_avg_map[row.level] = row['prev_n_op_red_avg']
                else:
                    prev_cycle_avg_map[row.level] = row['prev_n_op_green_avg']
                      

        if row[new_map[current_label]]:

            compute = op_map[new_map[current_label]]
            if row.is_high == 1 and row.is_low == 0:
                fractal_val = row['fractal_val_green']
            elif row.is_high == 0 and row.is_low == 1:
                fractal_val = row['fractal_val_red']
            elif row.is_high == 1 and row.is_low ==1:
                fractal_val = np.NaN                         ## Initializing fractal_val to Nan So that if 1,1  doesnt match with current label we would get step5 NaN
                if current_label == neutral_index_map[row['neutral_index_is_high']]:
                    if row['neutral_index_is_high'] == 1:
                        fractal_val = row['fractal_val_green']
                    else:
                        fractal_val = row['fractal_val_red']
                
            val = compute(fractal_val,prev_cycle_avg_map[current_label])

            # if row.is_high ==1 and row.is_low ==1:
            #     print(index,'|',val,'|',current_label)

            if val<0:
                step_5[index] = val
                current_label = change_label(current_label)  
            else:
                step_5[index] = val
                
    data = data.assign(step_5 = step_5)
    return data,new_map

def step_6(data,new_map):
    
    step_6_map = {new_map[i]:i for i in new_map} 
    
    data.loc[(data['is_high']==1) & (data['is_low']==0) & (data['step_5']>0) ,'step_6'] = step_6_map['is_high']
    data.loc[(data['is_high']==1) & (data['is_low']==0) & (data['step_5']<0),'step_6'] = step_6_map['is_low']

    data.loc[(data['is_low']==1) & (data['is_high']==0) & (data['step_5']>0) ,'step_6'] = step_6_map['is_low']
    data.loc[(data['is_low']==1) & (data['is_high']==0) & (data['step_5']<0),'step_6'] = step_6_map['is_high']

    data.loc[(data['is_high']==1) & (data['is_low']==1) & (data['neutral_index_is_high']==1) & (data['step_5']>0),'step_6'] = step_6_map['is_high']
    data.loc[(data['is_high']==1) & (data['is_low']==1) & (data['neutral_index_is_high']==0) & (data['step_5']>0),'step_6'] = step_6_map['is_low']
    
    data.loc[(data['is_high']==1) & (data['is_low']==1) & (data['neutral_index_is_high']==1) & (data['step_5']<0),'step_6'] = step_6_map['is_low']
    data.loc[(data['is_high']==1) & (data['is_low']==1) & (data['neutral_index_is_high']==0) & (data['step_5']<0),'step_6'] = step_6_map['is_high']
   
    return data

def calc_step_7(data,new_map,step = 1):
    step_7 = pd.Series(dtype="float64",index=data.index)
    data = data.assign(step_7 = step_7)

    step_7_map = {'is_high':'prev_n_op_red_avg','is_low':'prev_n_op_green_avg'}
    
    only_A_firsts = data[(data['level_change']==1)&(data['level']=='A')]  #Fetching Only the StartRows of a partcular cycle 
    only_B_firsts = data[(data['level_change']==1)&(data['level']=='B')]
    
    only_A_firsts['shift'] = only_A_firsts[step_7_map[new_map['A']]].shift(step)   #Shifting trailing average
    only_B_firsts['shift'] = only_B_firsts[step_7_map[new_map['B']]].shift(step)
    
    only_A_firsts['step_7'] = only_A_firsts[step_7_map[new_map['A']]]/only_A_firsts['shift']  #Calc growrh ratio
    only_B_firsts['step_7'] = only_B_firsts[step_7_map[new_map['B']]]/only_B_firsts['shift']
    
    for index,row in only_A_firsts.iterrows():
        data.at[index,'step_7'] = row['step_7']
    
    for index,row in only_B_firsts.iterrows():
        data.at[index,'step_7'] = row['step_7']
        
    return data

def step_9(data,n=5,k=2):
    
    prev_n_green = []
    prev_n_red = []

    for index,row in data.iterrows():

        if row.is_high and len(prev_n_green)<n:
            prev_n_green.append(index)
        elif row.is_high and len(prev_n_green)==n:
            prev_n_green.pop(0)
            prev_n_green.append(index)

        if row.is_low and len(prev_n_red)<n:
            prev_n_red.append(index)
        elif row.is_low and len(prev_n_red)==n:
            prev_n_red.pop(0)
            prev_n_red.append(index)

        if len(prev_n_green) == n and len(prev_n_red)==n:
            break  
    
    
    if prev_n_red[-1]<prev_n_green[-1]:  ## Start index of calculation
        start_index = prev_n_green[-1]
        #prev_n_green.pop()   ## removing the latest index so loop can have continuity of logic from start row and we'll appen this index at the begining of loop

    elif prev_n_green[-1]<prev_n_red[-1]:
        start_index = prev_n_red[-1]
        #prev_n_red.pop

    elif prev_n_green[-1] == prev_n_red[-1]:
        start_index = prev_n_red[-1]
        
    n_green_avg = pd.Series(dtype="float64",index=data.index)
    n_red_avg = pd.Series(dtype="float64",index=data.index)

    for index,row in data[start_index:].iterrows():

        if index == start_index:        #Not popping the prev_red and prev_green indices 
            green_val = data.loc[prev_n_green]['High'].mean()
            red_val = data.loc[prev_n_red]['Low'].mean()

            n_green_avg.loc[index] = green_val
            n_red_avg.loc[index] = red_val

        else:

            if row.is_high :                    #When Green fractal comes pop the first element and add current index
                prev_n_green.pop(0)
                prev_n_green.append(index)

            if row.is_low:                      #When Red fractal comes pop the first element and add current index    
                prev_n_red.pop(0)
                prev_n_red.append(index)        #When both Red and Green comes we will any way have latest n using above logic


            n_green_avg.loc[index] = data.loc[prev_n_green]['High'].mean()
            n_red_avg.loc[index] = data.loc[prev_n_red]['Low'].mean()

    data['green_avg'] = n_green_avg
    data['red_avg'] = n_red_avg
    
    
    strided_green_avg = pd.Series(dtype="float64",index=data.index)
    strided_red_avg = pd.Series(dtype="float64",index=data.index)
    min_prev_candles = (k+1)*n

    for index,row in data.iterrows():
        sub_df = data.loc[:index]
        sub_df_high = sub_df[sub_df['is_high']==1]
        sub_df_low = sub_df[sub_df['is_low']==1]
        print(index)

        if sub_df_high.shape[0]>= min_prev_candles:

            last_index = sub_df_high.index[-1]
            row_num = sub_df_high.index.get_loc(last_index)

            end_index = sub_df_high.index[row_num-n*k]
            start_index = sub_df_high.index[row_num-n*(k+1)+1]

            prev_strided_green_avg = sub_df_high.loc[start_index:end_index]['High'].mean()  

            strided_green_avg.loc[index] = prev_strided_green_avg

        if sub_df_low.shape[0]>=min_prev_candles:

            last_index = sub_df_low.index[-1]
            row_num = sub_df_low.index.get_loc(last_index)

            end_index = sub_df_low.index[row_num-n*k]
            start_index = sub_df_low.index[row_num-n*(k+1)+1]

            prev_strided_red_avg = sub_df_low.loc[start_index:end_index]['Low'].mean()

            strided_red_avg.loc[index] = prev_strided_red_avg
            

    data['green_op'] = strided_green_avg
    data['red_op'] = strided_red_avg
        
    data['avg_green_growth'] = data['green_avg']/data['green_op']-1
    data['avg_red_growth'] = data['red_avg']/data['red_op']-1
    
    return data


def step_11(data,n=5,k=2):
    
    min_prev_candles = (n+k)
    
    step_10 = pd.Series(dtype="float64",index=data.index)

    red_op = pd.Series(dtype="float64",index=data.index)
    green_op = pd.Series(dtype="float64",index=data.index)

    for index,row in data.iterrows():

        sub_df = data.loc[:index]

        if row.is_high ==1 :

            sub_df_low = sub_df[sub_df['is_low']==1]
            if sub_df_low.shape[0]>=min_prev_candles:

                last_index = sub_df_low.index[-1]
                row_num = sub_df_low.index.get_loc(last_index)

                end_index = sub_df_low.index[row_num-n]
                start_index = sub_df_low.index[row_num-n-k+1]

                output_val = sub_df_low.loc[start_index:end_index]['Low'].mean()
                red_op.loc[index] = output_val

        elif row.is_low == 1 :

            sub_df_high = sub_df[sub_df['is_high']==1]
            if sub_df_high.shape[0]>=min_prev_candles:

                last_index = sub_df_high.index[-1]
                row_num = sub_df_high.index.get_loc(last_index)

                end_index = sub_df_high.index[row_num-n]
                start_index = sub_df_high.index[row_num-n-k+1]

                output_val = sub_df_high.loc[start_index:end_index]['High'].mean()
                green_op.loc[index] = output_val
                
    data['red_op_continuos'] = red_op
    data['green_op_continuos'] = green_op
    return data


def fractal_all(data, trailing_avg_length = 5, bin_threshold = 0.5, trailing_length = 10, trailing_stride = 2, trailing_continuos = 5):

    data = get_n_trailing_averages_new(data,trailing_avg_length)

    data['fractal_val_green'] = data.apply(lambda row : get_fractal_val_green(row.is_high,row.High,row.Low),axis=1)
    data['fractal_val_red'] = data.apply(lambda row : get_fractal_val_red(row.is_low,row.High,row.Low),axis=1)

    data['abs_diff'] = data.apply(lambda row: compute_abs_diff(row.is_low,row.is_high,row.prev_n_op_red_avg,row.prev_n_op_green_avg,row.fractal_val_green,row.fractal_val_red),axis=1)
    data['%_growth'] = data.apply(lambda row: compute_percent_growth(row.is_low,row.is_high,row.prev_n_op_red_avg,row.prev_n_op_green_avg,row.fractal_val_green,row.fractal_val_red),axis=1)

    ## modifying the values for 1,1 fractals
    neutral_indices = data[(data['is_high']==1) & (data['is_low']==1)].index

    green_start = data['prev_n_op_red_avg'].first_valid_index()
    red_start = data['prev_n_op_green_avg'].first_valid_index()

    if green_start<red_start:
        start_index = green_start
    else:
        start_index = red_start


    neutral_index_is_high = pd.Series(dtype="float64",index=data.index)

    for index in neutral_indices:
        prev_total_data = data.loc[start_index:index]
        prev_normal_data = prev_total_data[~((prev_total_data['is_high']==1) & (prev_total_data['is_low']==1))]

        if not prev_normal_data.empty:
            prev_last_normal_rec = prev_normal_data.iloc[-1]

            if prev_last_normal_rec.is_high:
                data.at[index,'abs_diff'] = data.loc[index]['fractal_val_green'] - data.loc[index]['prev_n_op_red_avg']
                data.at[index,'%_growth'] = (data.loc[index]['fractal_val_green']/data.loc[index]['prev_n_op_red_avg'] - 1)*100
                neutral_index_is_high[index] = 1
            else:
                data.at[index,'abs_diff'] = data.loc[index]['fractal_val_red'] - data.loc[index]['prev_n_op_green_avg']
                data.at[index,'%_growth'] = (data.loc[index]['fractal_val_red']/data.loc[index]['prev_n_op_green_avg'] - 1)*100
                neutral_index_is_high[index] = 0
        
        
        else:
            if (data.loc[index]['fractal_val_green'] - data.loc[index]['prev_n_op_red_avg']) < 0:
                data.at[index,'abs_diff'] = (data.loc[index]['fractal_val_green'] - data.loc[index]['prev_n_op_red_avg'])
                data.at[index,'%_growth'] = (data.loc[index]['fractal_val_green']/data.loc[index]['prev_n_op_red_avg'] - 1)*100
                neutral_index_is_high[index] = 1

            elif (data.loc[index]['fractal_val_red'] - data.loc[index]['prev_n_op_green_avg']) > 0:
                data.at[index,'abs_diff'] = (data.loc[index]['fractal_val_red'] - data.loc[index]['prev_n_op_green_avg'])
                data.at[index,'%_growth'] = (data.loc[index]['fractal_val_red']/data.loc[index]['prev_n_op_green_avg'] - 1)*100
                neutral_index_is_high[index] = 0
            else:
                data.at[index,'abs_diff'] = np.NaN

    data = data.assign(neutral_index_is_high = neutral_index_is_high)

    data,label_map,op_map = generate_levels(data,'abs_diff')
    data = bin_abs_diff(data,label_map,bin_threshold)
    data,new_map = step_5_new(data,label_map)
    data = step_6(data,new_map)
    data = calc_step_7(data,new_map)
    data = step_9(data, trailing_length, trailing_stride)
    data = step_11(data, trailing_length, trailing_continuos)
    return data


        
