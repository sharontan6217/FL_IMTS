import pandas as pd
import numpy as np
from numpy import log10, log2, exp2
import os

import experiments 

from experiments.data import activity_dataLoad,physionet_dataLoad,airquality_dataLoad,mimicicu_dataLoad,ecg_dataLoad,uci_dataLoad,eeg_dataLoad,climate_dataLoad,climate_dataLoad,test_dataLoad,finance_dataLoad
import model
from model import brnn
from model.brnn import neuralNetwork
import framework
from framework import federated_learning_nn
import eval
from eval import evaluation, visualize
import train
from train import impute, predict
import utils
from utils import preprocess, utils
import keras
import math
import argparse
import datetime
import random
import time
import matplotlib.pyplot as plt
from framework import federated_learning_nn
import gc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='./data/activity/processed/data.pt', help = 'directory of the original data.' )
    parser.add_argument('--graph_dir',type=str,default='./graph/', help = 'directory of graphs' )
    parser.add_argument('--output_dir',type=str,default='./output/', help = 'directory of outputs')
    parser.add_argument('--log_dir',type=str,default='./log/', help = 'directory of the transaction logs.' )
    parser.add_argument('--save_dir',type=str,default='./train/', help = 'directory of the weights.' )
    parser.add_argument('--pretrain',type=bool,default=False, help = 'Whether to use pretrained models.' )

    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    start_time = datetime.datetime.now()
    gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)

    #data_dir = './data/activity/processed/data.pt'
    #data_dir = './data/physionet/processed/'
    #data_dir = './data/climate/data/processed/cleaned_df.csv'
    #data_dir = './data/climate/data/uci/'
    #data_dir = 'C:/Users/sharo/Documents/fl_imts/data/mimic_icu/mimic_preprocessed.csv'
    opt = get_parser()
    data_dir = opt.data_dir
    graph_dir = opt.graph_dir
    log_dir =opt.log_dir
    output_dir =opt.output_dir
    train_data_dir=data_dir+'X_train.csv'
    test_data_dir=data_dir+'X_test.csv'
    save_dir = opt.save_dir
    if 'uci' in data_dir:
        orig,cols_orig = uci_dataLoad(train_data_dir,test_data_dir)
    elif 'climate' in data_dir.lower():
        orig,cols_orig = climate_dataLoad(data_dir)
    elif 'eeg' in data_dir.lower():
        orig,cols_orig = eeg_dataLoad(data_dir)
    elif 'ecg' in data_dir.lower():
        orig,cols_orig = ecg_dataLoad(data_dir)
    elif 'mimic' in data_dir.lower():
        orig,cols_orig = mimicicu_dataLoad(data_dir)
    elif 'physionet' in data_dir.lower():
        orig,cols_orig = physionet_dataLoad(data_dir)
    elif 'activity' in data_dir.lower():
        orig,cols_orig = activity_dataLoad(data_dir)

    print(len(orig))


    
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)

    #start = 0
    timeSequence = str(datetime.datetime.now())[20:26]
    x,y,x_impute,x_train,y_train,x_test,y_test,y_actual,start = preprocess.dataSplit(orig,timeSequence,opt,cols_orig)
    #-----------------------Predict with daily refreshed data, e.g.: predict 30 or 100 days consecutively based on Day_t-1 data---------------------
    y_predict, y_actual = predict.FL_train_nn(x_train,y_train,x_test,y_test,y_actual,x_impute,cols_orig,timeSequence,start,opt)
    #-----------------------Predict in a time window: e.g.: predict 30 days or 100 days based on Day_0 data----------------------
    #y_predict, y_actual = predict.FL_train_predict_window(x_train,y_train,x_test,y_test,y_actual,x_impute,cols_orig,timeSequence,start,opt)
    y_predict_fl = utils.fl_convertion(y_predict).reshape(-1,1)
    y_actual_fl = utils.fl_convertion(y_actual).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    f1score,accuracy,mse,mae = eval.evaluation.evaluation(y_actual_fl,y_predict_fl,opt)
    print(f1score,accuracy,mse,mae )
    fig = eval.visualize.visualize(y_actual_fl,y_predict_fl,timeSequence,start,cols_orig,opt)   
    df_result = eval.visualize.output(y_actual_fl,y_predict_fl,timeSequence,start,opt)   
    
    end_time = datetime.datetime.now()
    print("start time is {}, and end time is {}".format(str(start_time),str(end_time)))
    with open('time.log','a') as f:
        f.write(str([start_time,end_time, end_time-start_time]))
        f.close()


