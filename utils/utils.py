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
def fl_convertion(data_fl):
    df_fl = pd.DataFrame(data=data_fl)
    data_converted = []
    for col in df_fl.columns:
        data_converted.append(df_fl[col])
    data_converted = np.asarray(data_converted).reshape(-1,1,1)
    return data_converted