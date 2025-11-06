
import pandas as pd
import numpy as np
from numpy import log10, log2, exp2
import os
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import train_test_split
import model
from model import brnn
from model.brnn import neuralNetwork
import keras
import math
import argparse
import datetime
import random
import time
import matplotlib.pyplot as plt
from framework import federated_learning_nn
from model.Config import brnn_config,fl_config
import gc

scaler = StandardScaler()
#scaler = MinMaxScaler()
fl_config = fl_config()
trainSize = fl_config.trainSize
testSize = fl_config.testSize
predictSize = fl_config.predictSize

def visualize(actual,predict,timeSequence,start,cols_orig,opt):
    for n in range(len(cols_orig)):
        time.sleep(5)
        print(cols_orig[n])
        col_name = cols_orig[n]
        if n==0:
            predict_value=predict[:predictSize]
            actual_value=actual[:predictSize]
        else:
            predict_value=predict[predictSize*n:predictSize*(n+1)]
            actual_value=actual[predictSize*n:predictSize*(n+1)]
        col_name = col_name.replace(' ','').replace('|','')
        fig_name='test_scenario_'+col_name+'_'+timeSequence+'_'+str(start)+'_brnn.png'
        fig=plt.figure()
        print(actual_value)
        print(predict_value)
        plt.plot(actual_value,color='blue',label='Actual')
        plt.plot(predict_value,color='red',label='Prediction')
        plt.xlabel('Time')
        plt.ylabel(col_name)
        plt.title('Plot Graph of Actual and Predicted {}'.format(col_name))
        plt.legend(loc='best')
        plt.savefig(opt.graph_dir+fig_name)
        plt.close()   
    fig=plt.figure()
    for n in range(len(cols_orig)):
        #print(cols_orig[n])
        col_name = cols_orig[n]
        if n==0:
            predict_value=predict[:predictSize]
            actual_value=actual[:predictSize]
        else:
            predict_value=predict[predictSize*n:predictSize*(n+1)]
            actual_value=actual[predictSize*n:predictSize*(n+1)]
        
        plt.plot(actual_value,label=col_name+'_Actual')
        plt.plot(predict_value,label=col_name+'_Prediction')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend(bbox_to_anchor=(0.5,1.15),fontsize='xx-small',ncol=4,loc='upper center')
    fig_name='test_scenario_all'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(opt.graph_dir+fig_name)
    plt.close()  
    return fig
def output(actual,predict,timeSequence,start,opt):
    df_result_actual=pd.DataFrame(data=actual)
    df_result_predict=pd.DataFrame(data=predict)
    df_result = pd.concat((df_result_actual,df_result_predict),axis=1)
    output_name = 'output_'+timeSequence+'_'+str(start)+'.csv'
    df_result.to_csv(opt.output_dir+output_name)
    return df_result