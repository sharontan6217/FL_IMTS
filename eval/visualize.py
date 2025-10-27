
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
import data
from data import airquality_dataLoad,mimicicu_dataLoad,ecg_dataLoad,uci_dataLoad,eeg_dataLoad
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
import gc

#scaler = StandardScaler()
scaler = MinMaxScaler()


def visualize(actual,predict):
    
    predict_0=predict[:predictSize]
    predict_1=predict[predictSize:predictSize*2]
    predict_2=predict[predictSize*2:predictSize*3]
    predict_3=predict[predictSize*3:]
    actual_0=actual[:predictSize]
    actual_1=actual[predictSize:predictSize*2]
    actual_2=actual[predictSize*2:predictSize*3]
    actual_3=actual[predictSize*3:]

    fig=plt.figure(1)
    plt.plot(actual_0,color='blue',label='Actual')
    plt.plot(predict_0,color='red',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('P4')
    plt.title('Plot Graph of Actual and Predicted EEG-P4')
    plt.legend(loc='best')
    fig_name='test_scenario_P4'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close()      
    fig=plt.figure(2)
    plt.plot(actual_1,color='blue',label='Actual')
    plt.plot(predict_1,color='red',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('Cz')
    plt.title('Plot Graph of Actual and Predicted EEG-Cz')
    plt.legend(loc='best')
    fig_name='test_scenario_Cz'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close() 
    fig=plt.figure(3)
    plt.plot(actual_2,color='blue',label='Actual')
    plt.plot(predict_2,color='red',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('F8')
    plt.title('Plot Graph of Actual and Predicted EEG-F8')
    plt.legend(loc='best')
    fig_name='test_scenario_F8'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close() 
    fig=plt.figure(4)
    plt.plot(actual_3,color='blue',label='Actual')
    plt.plot(predict_3,color='red',label='Prediction')
    plt.xlabel('Time')
    plt.ylabel('T7')
    plt.title('Plot Graph of Actual and Predicted EEG-T7')
    plt.legend(loc='best')
    fig_name='test_scenario_T7'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close() 
    fig=plt.figure(0)
    plt.plot(actual_0,color='blue',label='P4_Actual')
    plt.plot(predict_0,color='red',label='P4_Prediction')
    plt.plot(actual_1,color='grey',label='Cz_Actual')
    plt.plot(predict_1,color='pink',label='Cz_Prediction')
    plt.plot(actual_2,color='green',label='F8_Actual')
    plt.plot(predict_2,color='yellow',label='F8_Prediction')
    plt.plot(actual_3,color='black',label='T7_Actual')
    plt.plot(predict_3,color='orange',label='T7_Prediction')
    plt.xlabel('Time')
    plt.ylabel('Values')
    plt.legend(bbox_to_anchor=(0.5,1.15),fontsize='xx-small',ncol=4,loc='upper center')
    fig_name='test_scenario_all'+timeSequence+'_'+str(start)+'_brnn.png'
    plt.savefig(graph_dir+fig_name)
    plt.close()  
    return fig
def output(actual,predict):
    df_result_actual=pd.DataFrame(data=actual)
    df_result_predict=pd.DataFrame(data=predict)
    df_result = pd.concat((df_result_actual,df_result_predict),axis=1)
    output_name = 'output_'+timeSequence+'_'+str(start)+'.csv'
    df_result.to_csv(output_dir+output_name)
    return df_result