import pandas as pd
import numpy as np
from numpy import log10, log2, exp2
import torch
from torch import Tensor
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import sqeuclidean,jaccard,canberra,cdist,euclidean,jensenshannon
import tensorflow as tf
#import model
#from model import Config, brnn
#from model.Config_ablation_gru import brnn_config,fl_config
#from model.brnn import neuralNetwork
#config = fl_config()

#model_config = brnn_config()


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

def reverse_normalization(x, missing,cols_orig,na_replacement):
    X = pd.DataFrame(x,columns=cols_orig).reset_index().drop('index',axis=1)
    X_ = pd.DataFrame(x.replace(na_replacement,np.nan).dropna().astype(np.float32))
    min_scaler = X_.min()
    max_scaler = X_.max()
    print(min_scaler)
    print(max_scaler)
    df_x = pd.DataFrame(data=missing,columns=cols_orig)
    for i in range(len(cols_orig)):
        print(cols_orig[i])
        col = cols_orig[i]
        col_temp = col+'_temp'
        df_x[col_temp]=df_x[col]
        df_x[col]=df_x[col_temp]*(max_scaler[i]-min_scaler[i])+min_scaler[i]
        df_x=df_x.drop(col_temp,axis=1)
        for j in range(len(df_x)):
            if X.loc[j,col]==na_replacement:
                X.loc[j,col]=df_x.loc[j,col]
    #x_std = df_x.values
    x_std = X.values
    return x_std
def reverse_standardation(x, missing,cols_orig,na_replacement):
    X = pd.DataFrame(x,columns=cols_orig).reset_index().drop('index',axis=1)
    X_ = pd.DataFrame(x.replace(na_replacement,0).dropna().astype(np.float32))
    mean_scaler = X_.mean()
    stdev_scaler = X_.std()
    print(mean_scaler)
    print(stdev_scaler)
    df_x = pd.DataFrame(data=missing,columns=cols_orig)
    for i in range(len(cols_orig)):
        print(cols_orig[i])
        col = cols_orig[i]
        col_temp = col+'_temp'
        df_x[col_temp]=df_x[col]
        df_x[col]=df_x[col_temp]*stdev_scaler[i]+mean_scaler[i]
        df_x=df_x.drop(col_temp,axis=1)

        for j in range(len(X)):
            if int(X.loc[j,col])==na_replacement:
                X.loc[j,col]=df_x.loc[j,col]
    #x_std = df_x.values
    x_std = X.values
    return x_std

def fl_convertion(data_fl):
    df_fl = pd.DataFrame(data=data_fl)
    data_converted = []
    for col in df_fl.columns:
        data_converted.append(df_fl[col])
    data_converted = np.asarray(data_converted).reshape(-1,1,1)
    return data_converted


def federated_timegap(dataset):
	gaps=[0]
	for col in dataset.columns:
		if ('date' in col.lower() or 'time' in col.lower())==True:
			ts = pd.to_datetime(dataset[col]).reset_index()
			if len(ts)<2:
				return 0.0
			
			for i in range(len(ts)-1):
				gap = ts['DATE'].values[i+1]-ts['DATE'].values[i]
				gap = gap.astype('timedelta64[s]').item()
				gap = float(gap.total_seconds()/(60*60*24))
				gaps.append(gap)
	return gaps
def max_min_value(batchdataset):
	min_values=[]
	max_values=[]
	#timegaps = timegaps.numpy()
	for values in batchdataset:
		if len(values)>1:
			for value in values:
				value = np.array(value).reshape(-1,)
				max_value_ = max(value)
				min_value_ = min(value)
				min_values.append(min_value_)
				max_values.append(max_value_)
		else:
			values = np.array(values).reshape(-1,)
			max_value_ = max(values)
			min_value_ = min(values)
			min_values.append(min_value_)
			max_values.append(max_value_)
	max_value = max(max_values)
	min_value = min(min_values)
    #print(max_timegaps,min_timegaps)
	return max_value,min_value
class lower_bound_minimum_decay():
    def lower_bound_minimum_decay(time_length, max_gap, gamma, gamma_change_per_step=0.01 ):

        gatting_bound = gamma**time_length
        decay_bound = np.exp(-gamma_change_per_step*time_length*max_gap)
        theoretical_lower_bound = gatting_bound*decay_bound

        return theoretical_lower_bound
    def visualize_t(max_gaps,preset_gamma,time_length,timeSequence,start,brnn_graph_dir):

        gamma = preset_gamma
        bounds =[]
        for max_gap in max_gaps:
            theoretical_lower_bound = lower_bound_minimum_decay.lower_bound_minimum_decay(time_length, max_gap, gamma)
            bounds.append(theoretical_lower_bound)
        # Plotting the results
        fig=plt.figure()
        plt.figure(figsize=(10, 6))
        plt.plot(max_gaps, bounds, label='Theoretical Lower Bound ')
        plt.xlabel('Maximum Time Gap')
        plt.ylabel('Theoretical Lower Bound')
        plt.title('Theoretical Lower Bound For Different Maximum Time Gap When Minimum Decay Facotr is 0.9')
        plt.legend()
        plt.grid(True)
        #plt.show()
        fig_name='Theoretical Lower Bound For Different Maximum Time Gap_'+timeSequence+'_'+str(start)+'.png'
        plt.savefig(brnn_graph_dir+fig_name)
        return fig

