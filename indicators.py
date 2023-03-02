import numpy as np

def stochastic(data, k_length=14, d_length=3, ewm=9):
    """
    Takes K_length , D_length, EWM span as inputs
    """
    data['14-H'] = data['High'].rolling(k_length).max()
    data['14-L'] = data['Low'].rolling(k_length).min()
    data['%_K'] = (data['Close']-data['14-L'])/(data['14-H']-data['14-L'])*100
    data['%_D'] = data['%_K'].rolling(d_length).mean()
    data['%_K_EMA'] = data['%_K'].ewm(span=ewm,min_periods=0,adjust=False,ignore_na=False).mean()
    data.drop(columns=['14-H','14-L'],inplace=True)
    return data


def high_fractal_appeared(current_val,prev_2_max,next_val):
    if current_val == prev_2_max and current_val>next_val:
        return 1
    else:
        return 0
    
def low_fractal_appeared(current_val,prev_2_low,next_val):
    if current_val == prev_2_low and current_val<next_val:
        return 1
    else:
        return 0
    
def high_fractal_broken(fractal_appeared,current_val,last_val):
    if fractal_appeared == 0:
        return 0
    else:
        if last_val > current_val:
            return 0
        else:
            return 1
        
def low_fractal_broken(fractal_appeared,current_val,last_val):
    if fractal_appeared == 0:
        return 0
    else:
        if last_val < current_val:
            return 0
        else:
            return 1

def fractals(data):
    
    data['prev_2_high']=data['High'].rolling(3).max()
    data['prev_2_low'] = data['Low'].rolling(3).min()
    data['next_high'] = data['High'].shift(-1)
    data['next_low'] = data['Low'].shift(-1)
    data['last_high'] = data['High'].shift(-2)
    data['last_low'] = data['Low'].shift(-2)

    data['HF_appeared'] = data.apply(lambda x: high_fractal_appeared(x['High'],x['prev_2_high'],x['next_high']),axis=1)
    data['LF_appeared'] = data.apply(lambda x: low_fractal_appeared(x['Low'],x['prev_2_low'],x['next_low']),axis=1)

    data['is_high'] = data.apply(lambda x: high_fractal_broken(x['HF_appeared'],x['High'],x['last_high']),axis=1)
    data['is_low'] = data.apply(lambda x : low_fractal_broken(x['LF_appeared'],x['Low'],x['last_low']),axis=1)

    data['fractal_time'] = data.index
    data['fractal_time'] = data['fractal_time'].shift(-2)
    data.loc[(data['is_high']==0) & (data['is_low']==0),'fractal_time'] = np.NaN
    
    data.drop(columns=['prev_2_high','prev_2_low','next_high','next_low','last_high','last_low'],axis=1,inplace=True)
    return data


def rsi(data):
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up/ema_down
    data['RSI'] = 100 - (100/(1 + rs))
    data['RSI_EMA'] = data['RSI'].ewm(span=9,min_periods=0,adjust=False,ignore_na=False).mean()
    return data


def bollinger_bands(data, length = 20, n=2):
    TP = (data['Close'] + data['Low'] + data['High'])/3
    std = TP.rolling(length).std()
    MA_TP = TP.rolling(length).mean()
    data['BOLU'] = MA_TP + n*std
    data['Basis'] = MA_TP
    data['BOLD'] = MA_TP - n*std
    return data


def gain(x):
    return ((x > 0) * x).sum()

def loss(x):
    return ((x < 0) * x).sum()

def mfi(data,n=14):
    typical_price = (data['High'] + data['Low']+ data['Close'])/3
    money_flow = typical_price*data['Total Volume']
    mf_sign = np.where(typical_price > typical_price.shift(1),1,-1)
    signed_mf = mf_sign*money_flow
    mf_avg_gain = signed_mf.rolling(n).apply(gain, raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(loss, raw=True)
    mfi=(100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))).to_numpy()
    data['MF'] = mfi
    data['MF_EMA'] = data['MF'].ewm(span=9,min_periods=0,adjust=False,ignore_na=False).mean()
    return data

