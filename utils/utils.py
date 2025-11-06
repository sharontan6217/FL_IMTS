import pandas as pd
import numpy as np
from numpy import log10, log2, exp2

def logarithm(x_orig,y_orig):
    diff =y_orig-x_orig
    orig_log=[]
    for i in range(len(x_orig)):
        diff[i]=y_orig[i]-x_orig[i]
        if diff[i]<0:
            orig_log.append((-1)*log2((-1)*diff[i]))
        if diff[i]==0:
            orig_log.append(0)
        if diff[i]>0:
            orig_log.append(log2(diff[i]))
        i+=1
    return orig_log
def reverse_logarithm(orig_log):
    diff=np.zeros(shape=(orig_log.shape[0],1))
    for i in range(orig_log.shape[0]):
        if orig_log[i, :]>0:
            diff[i]=(-1)*np.exp2((-1)*orig_log[i, :])
        if orig_log[i, :]==0:
            diff[i]=0
        if orig_log[i, :]<0:
            diff[i]=np.exp2(orig_log[i, :])
    return diff
def reverse_normalization(scaler_data, x,cols_orig):
    X = pd.DataFrame(scaler_data)
    min_scaler = X.min(axis=1)
    max_scaler = X.max(axis=1)
    print(min_scaler)
    print(max_scaler)
    df_x = pd.DataFrame(data=x,columns=cols_orig)
    for i in range(len(cols_orig)):
        print(cols_orig[i])
        col = cols_orig[i]
        col_temp = col+'_temp'
        df_x[col_temp]=df_x[col]
        df_x[col]=df_x[col_temp]*(max_scaler[i]-min_scaler[i])+min_scaler[i]
        df_x=df_x.drop(col_temp,axis=1)
    x_std = df_x.values
    print(x)
    print(x_std)
    return x_std
def reverse_standardation(scaler_data, x,cols_orig):
    X = pd.DataFrame(scaler_data)
    mean_scaler = X.mean()
    stdev_scaler = X.std()
    print(mean_scaler)
    print(stdev_scaler)
    df_x = pd.DataFrame(data=x,columns=cols_orig)
    for i in range(len(cols_orig)):
        print(cols_orig[i])
        col = cols_orig[i]
        col_temp = col+'_temp'
        df_x[col_temp]=df_x[col]
        df_x[col]=df_x[col_temp]*stdev_scaler[i]+mean_scaler[i]
        df_x=df_x.drop(col_temp,axis=1)
    x_std = df_x.values
    print(x)
    print(x_std)
    return x_std

def fl_convertion(data_fl):
    df_fl = pd.DataFrame(data=data_fl)
    data_converted = []
    for col in df_fl.columns:
        data_converted.append(df_fl[col])
    data_converted = np.asarray(data_converted).reshape(-1,1,1)
    return data_converted