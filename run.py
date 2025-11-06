import pandas as pd
import numpy as np
from numpy import log10, log2, exp2
import os

import experiments 

from experiments.data import airquality_dataLoad,mimicicu_dataLoad,ecg_dataLoad,uci_dataLoad,eeg_dataLoad
import model
from model import brnn
from model.brnn import neuralNetwork
import framework
from framework import federated_learning_nn
import eval
from eval import evaluation, visualize
import train
from train import imputate, predict
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
    parser.add_argument('--data_dir',type=str,default='./data/eeg/s01_ex01_s01.csv', help = 'directory of the original data.' )
    parser.add_argument('--graph_dir',type=str,default='./graph/eeg/', help = 'directory of graphs' )
    parser.add_argument('--output_dir',type=str,default='./output/eeg/', help = 'directory of outputs')
    parser.add_argument('--log_dir',type=str,default='./log/eeg/', help = 'directory of the transaction logs.' )
    parser.add_argument('--with_metalearning',type=bool,default=False, help = 'Defult to be False, True if adding meta-learning method.' )
    parser.add_argument('--metalearning_name',type=str,default='None', help = 'learning method is one of the list ["None", "reptile","MAML"], reptile for gradient decent algorithms and Model Agonistic Meta Learning (MAML) for ML and DL algorithms' )
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    gc.collect()
    project_dir = os.getcwd()
    os.chdir(project_dir)
    #data_dir = './data/climate/data/ushcn_daily/pub12/ushcn_daily/state08_FL.csv'
    #data_dir = 'C:/Users/sharo/Documents/fl_imts/data/PhysioNet/ecg-id-database-1.0.0/ecg-id-database-1.0.0/Person_01/output.csv'
    #data_dir = 'C:/Users/sharo/Documents/fl_imts/data/PhysioNet/auditory-evoked-potential-eeg-biometric-dataset-1.0.0/auditory-evoked-potential-eeg-biometric-dataset-1.0.0/Filtered_Data/s01_ex01_s01.csv'
    #data_dir = 'C:/Users/sharo/Documents/fl_imts/data/mimic_icu/LABEVENTS.csv'
    #graph_dir = './graph/5000-1000/eeg/'
    #log_dir = './log/5000-1000/eeg/'
    #output_dir = './output/5000-1000/eeg/'
    opt = get_parser()
    data_dir = opt.data_dir
    graph_dir = opt.graph_dir
    log_dir =opt.log_dir
    output_dir =opt.output_dir




    #df,orig = climate_dataLoad()
    orig = eeg_dataLoad(data_dir)
    #data,orig = mimicicu_dataLoad()
    print(len(orig))
    cols_orig = orig.columns

    
    if os.path.exists(graph_dir)==False:
        os.makedirs(graph_dir)
    if os.path.exists(log_dir)==False:
        os.makedirs(log_dir)
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    #start = 0
    timeSequence = str(datetime.datetime.now())[20:26]
    x,y,x_imputate,x_train,y_train,x_test,y_test,y_actual,start = preprocess.dataSplit(orig,timeSequence,opt,cols_orig)
    y_predict, y_actual = predict.FL_train_nn(x_train,y_train,x_test,y_test,y_actual,x_imputate,cols_orig,timeSequence,start,opt)
    #y_actual,y_predict = FL_train_gan(x_train,y_train,x_test,y_test,y_actual)
    #y_actual,y_predict = FL_train_predict_window(x,y,x_train,y_train,x_test,y_test,y_actual,start)
    y_predict_fl = utils.fl_convertion(y_predict).reshape(-1,1)
    y_actual_fl = utils.fl_convertion(y_actual).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    f1score,accuracy,mse,mae = eval.evaluation.evaluation(y_actual_fl,y_predict_fl,opt)
    print(f1score,accuracy,mse,mae )
    #fig = visualize(x_actual_,x_predict_)
    fig = eval.visualize.visualize(y_actual_fl,y_predict_fl,timeSequence,start,cols_orig,opt)   
    df_result = eval.visualize.output(y_actual_fl,y_predict_fl,timeSequence,start,opt)   
    #df_result = output(y_actual_fl,y_predict_fl)

