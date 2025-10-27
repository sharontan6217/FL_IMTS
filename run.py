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

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_metalearning',type=bool,default=False, help = 'Defult to be False, True if adding meta-learning method.' )
    parser.add_argument('--metalearning_name',type=str,default='None', help = 'learning method is one of the list ["None", "reptile","MAML"], reptile for gradient decent algorithms and Model Agonistic Meta Learning (MAML) for ML and DL algorithms' )
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)
    #data_path = './data/climate/data/ushcn_daily/pub12/ushcn_daily/state08_FL.csv'
    #data_path = 'C:/Users/sharo/Documents/fl_imts/data/PhysioNet/ecg-id-database-1.0.0/ecg-id-database-1.0.0/Person_01/output.csv'
    data_path = 'C:/Users/sharo/Documents/fl_imts/data/PhysioNet/auditory-evoked-potential-eeg-biometric-dataset-1.0.0/auditory-evoked-potential-eeg-biometric-dataset-1.0.0/Filtered_Data/s01_ex01_s01.csv'
    #data_path = 'C:/Users/sharo/Documents/fl_imts/data/mimic_icu/LABEVENTS.csv'
    graph_dir = './graph/5000-1000/eeg/'
    log_dir = './log/5000-1000/eeg/'
    output_dir = './output/5000-1000/eeg/'




    #df,orig = climate_dataLoad()
    orig = eeg_dataLoad()
    #data,orig = mimicicu_dataLoad()
    print(len(orig))

    totalSize = trainSize+testSize+predictSize
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    start = random.randint(totalSize+1,len(orig)-1)
    #start = 0
    timeSequence = str(datetime.datetime.now())[20:26]
    x,y,x_impute,x_train,y_train,x_test,y_test,y_actual = dataSplit(orig)
    y_predict, y_actual = FL_train_nn(x_train,y_train,x_test,y_test,y_actual,x_impute)
    #y_actual,y_predict = FL_train_gan(x_train,y_train,x_test,y_test,y_actual)
    #y_actual,y_predict = FL_train_predict_window(x_train,y_train,x_test,y_test,y_actual)
    y_predict_fl = fl_convertion(y_predict).reshape(-1,1)
    y_actual_fl = fl_convertion(y_actual).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    f1score,accuracy,mse,mae = evaluation(y_actual_fl,y_predict_fl)
    print(f1score,accuracy,mse,mae )
    #fig = visualize(x_actual_,x_predict_)
    fig = visualize(y_actual_fl,y_predict_fl)   
    df_result = output(y_actual_fl,y_predict_fl)   
    #df_result = output(y_actual_fl,y_predict_fl)

