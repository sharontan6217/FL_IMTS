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
import utils
from utils.utils import fl_convertion,reverse_normalization,reverse_standardation
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
from model.Config import brnn_config,fl_config
from framework import federated_learning_nn
import gc


scaler = MinMaxScaler()
#scaler = StandardScaler()
config = fl_config()
model_config = brnn_config()
trainSize = config.trainSize
testSize = config.testSize
predictSize = config.predictSize



def FL_train_nn(x_train,y_train,x_test,y_test,y_actual,x_imputate,cols_orig,timeSequence,start,opt):
    gc.collect()
    len_cols = len(cols_orig)

    y_train_orig=y_train
    #x_train_orig = x_train
    print('-----------------------------------------y_origin is--------------------------------------')
    print(y_train_orig)
    print('-----------------------------------------y_actual is--------------------------------------')
    print(y_actual)
    
    #y_actual=scaler_y.transform(y_actual)
    x_total = np.concatenate([x_train,x_test],axis=0)
    scaler_x = scaler.fit(x_total)
    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)
    #x_imputate = scaler_x.transform(x_imputate)
    #y_actual=scaler.transform(y_actual)
    y_total = np.concatenate([y_train,y_test],axis=0)
    scaler_y=scaler.fit(y_total)
    y_train = scaler_y.transform(y_train)
    y_test=scaler_y.transform(y_test)
    y_actual = scaler_y.transform(y_actual)
    #x_imputate = scaler.transform(x_imputate)
    #x_actual = x_imputate[trainSize:trainSize+testSize]
    #x_actual = scaler.transform(x_actual)
    #x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    #x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    with open ('train_predict.log','w') as f:
        f.write('-----------------x_train is-------------\n')
        f.write(str(x_train))
        f.write('-----------------y_train is-------------\n')
        f.write(str(y_train))
        f.write('-----------------x_test is-------------\n')
        f.write(str(x_test))
        f.write('-----------------y_test is-------------\n')
        f.write(str(y_test))
        f.close()
    graph_dir = opt.graph_dir
    brnn_graph_dir=graph_dir+'accuracy/'
    if os.path.exists(brnn_graph_dir)==False:
        os.makedirs(brnn_graph_dir)
    client_datasets,test_datasets=federated_learning_nn.dataProcess(x_train,y_train,x_test,y_test)
    state,metrics,loss,mae = federated_learning_nn.train(client_datasets)
    model_predict,test_metrics=federated_learning_nn.eval(test_datasets,state,metrics)
    fig = federated_learning_nn.fl_visualize(loss,mae,timeSequence,start,brnn_graph_dir)
    x_test_fl = fl_convertion(x_test)

    
    #meta_optimizer = optim.Adam(param_dict, lr=0.001)
    # Dummy tasks for demonstration
    for j in range(predictSize):
        print(j)
        gc.collect()
        if j==0:
            #y_predict,model_predict = selfTrainingClassifier.selfTrainingClassifier(model_predict,x_train,y_train,x_test,y_test)
            y_predict_fl = model_predict.predict(x_test_fl)
            y_predict_fl = y_predict_fl.astype(np.float32)
            print('---------------------predicted y is-------------------')
            print(y_predict_fl)
        else:
            #client_datasets,test_datasets=federated_learning_nn.dataProcess(x_train,y_train,x_actual,y_predict_fl)
            #state,metrics = federated_learning_nn.train(client_datasets)
            #model_predict=federated_learning_nn.eval(test_datasets,state,metrics)
            #y_predict,model_predict = selfTrainingClassifier.selfTrainingClassifier(model_predict,x_train,y_train,x_actual,y_predict)
            x_actual_fl = fl_convertion(x_actual)
            y_predict_fl = model_predict.predict(x_actual_fl)
            print('---------------------predicted y is-------------------')
            print(y_predict_fl)
            y_predict_fl = y_predict_fl.astype(np.float32)

        df_predict = pd.DataFrame()

        for n in range(len(cols_orig)):
            #print(cols_orig[n])
            col_name = cols_orig[n]
            if n==0:
                df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[:testSize]
            else:
                df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[testSize*n:testSize*(n+1)]

        print(df_predict)
        x_new=df_predict.values[-1:].reshape(1,-1)
        #x_new=df_predict.values[min(0,-j-1):]
        #x_new = np.reshape(np.array(x_new),((j+1),len_cols))
        print(x_new)

        x_actual1 = x_imputate[trainSize+j+1:trainSize+testSize+j]
        print(x_actual1)
        #print(x_actual1.shape)

        x_actual1=scaler_x.transform(x_actual1)
        x_actual = np.append(x_actual1,x_new,axis=0).astype(np.float32)
        #x_actual = scaler.transform(x_actual)
        #x_actual=fl_convertion(x_actual)
        print(x_actual)
        #x_new=x_predict[-1].reshape(-1,1)
        #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
        #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
        #y_predict.append(x_new)
        j=j+1
    print(len(y_predict_fl))
    df_predict = pd.DataFrame()

    for n in range(len(cols_orig)):
        print(cols_orig[n])
        col_name = cols_orig[n]
        if n==0:
            df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[:testSize]
        else:
            df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[testSize*n:testSize*(n+1)]


    print(len(df_predict))
    print(df_predict)
    
    y_predict_ = df_predict.values
    #y_predict_ = reverse_normalization(y_train_orig,y_predict_,cols_orig)
    #y_predict_ = reverse_standardation(x_total,y_predict_,cols_orig)
    y_predict = y_predict_[-predictSize:]
    
    print(len(y_predict))
    #y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    del x_new
    #scale_y= scaler.fit(y_imputate)
    #y_actual = scale_y.transform(y_actual)
    #y_predict = scale_y.transform(y_predict)
    
    print('start is: ',start)

    return y_predict, y_actual


def FL_train_predict_window(x_train,y_train,x_test,y_test,y_actual,x_imputate,cols_orig,timeSequence,start,opt):
    

    gc.collect()
    len_cols = len(cols_orig)

    y_train_orig=y_train
    #x_train_orig = x_train
    print('-----------------------------------------y_origin is--------------------------------------')
    print(y_train_orig)
    print('-----------------------------------------y_actual is--------------------------------------')
    print(y_actual)
    
    #y_actual=scaler_y.transform(y_actual)
    x_total = np.concatenate([x_train,x_test],axis=0)
    scaler_x = scaler.fit(x_total)
    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)
    x_total_imputate = scaler_x.transform(x_total)
    #x_imputate = scaler_x.transform(x_imputate)
    #y_actual=scaler.transform(y_actual)
    y_total = np.concatenate([y_train,y_test],axis=0)
    scaler_y=scaler.fit(y_total)
    y_train = scaler_y.transform(y_train)
    y_test=scaler_y.transform(y_test)
    y_actual = scaler_y.transform(y_actual)
    #x_imputate = scaler.transform(x_imputate)
    #x_actual = x_imputate[trainSize:trainSize+testSize]
    #x_actual = scaler.transform(x_actual)
    #x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    #x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
    
    with open ('train_predict.log','w') as f:
        f.write('-----------------x_train is-------------\n')
        f.write(str(x_train))
        f.write('-----------------y_train is-------------\n')
        f.write(str(y_train))
        f.write('-----------------x_test is-------------\n')
        f.write(str(x_test))
        f.write('-----------------y_test is-------------\n')
        f.write(str(y_test))
        f.close()
    graph_dir = opt.graph_dir
    brnn_graph_dir=graph_dir+'accuracy/'
    if os.path.exists(brnn_graph_dir)==False:
        os.makedirs(brnn_graph_dir)
    client_datasets,test_datasets=federated_learning_nn.dataProcess(x_train,y_train,x_test,y_test)
    state,metrics,loss,mae = federated_learning_nn.train(client_datasets)
    model_predict,test_metrics=federated_learning_nn.eval(test_datasets,state,metrics)
    fig = federated_learning_nn.fl_visualize(loss,mae,timeSequence,start,brnn_graph_dir)
    x_test_fl = fl_convertion(x_test)
    x_actual1 = x_total_imputate[trainSize:trainSize+testSize]
    
    #meta_optimizer = optim.Adam(param_dict, lr=0.001)
    # Dummy tasks for demonstration
    for j in range(predictSize):
        print(j)
        gc.collect()
        if j==0:
            #y_predict,model_predict = selfTrainingClassifier.selfTrainingClassifier(model_predict,x_train,y_train,x_test,y_test)
            y_predict_fl = model_predict.predict(x_test_fl)
            y_predict_fl = y_predict_fl.astype(np.float32)
            print('---------------------predicted y is-------------------')
            print(y_predict_fl)
        else:
            #client_datasets,test_datasets=federated_learning_nn.dataProcess(x_train,y_train,x_actual,y_predict_fl)
            #state,metrics = federated_learning_nn.train(client_datasets)
            #model_predict=federated_learning_nn.eval(test_datasets,state,metrics)
            #y_predict,model_predict = selfTrainingClassifier.selfTrainingClassifier(model_predict,x_train,y_train,x_actual,y_predict)
            x_actual_fl = fl_convertion(x_actual)
            y_predict_fl = model_predict.predict(x_actual_fl)
            print('---------------------predicted y is-------------------')
            print(y_predict_fl)
            y_predict_fl = y_predict_fl.astype(np.float32)

        df_predict = pd.DataFrame()

        for n in range(len(cols_orig)):
            #print(cols_orig[n])
            col_name = cols_orig[n]
            if n==0:
                df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[:testSize]
            else:
                df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[testSize*n:testSize*(n+1)]

        print(df_predict)
        x_new=df_predict.values[-1:].reshape(1,-1)
        #x_new=df_predict.values[min(0,-j-1):]
        #x_new = np.reshape(np.array(x_new),((j+1),len_cols))
        print(x_new)


        x_actual = np.append(x_actual1,x_new,axis=0).astype(np.float32)
        #x_actual = scaler.transform(x_actual)
        #x_actual=fl_convertion(x_actual)
        print(x_actual)
        #x_new=x_predict[-1].reshape(-1,1)
        #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
        #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)
        #y_predict.append(x_new)
        j=j+1
    print(len(y_predict_fl))
    df_predict = pd.DataFrame()

    for n in range(len(cols_orig)):
        print(cols_orig[n])
        col_name = cols_orig[n]
        if n==0:
            df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[:testSize]
        else:
            df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[testSize*n:testSize*(n+1)]


    print(len(df_predict))
    print(df_predict)
    
    y_predict_ = df_predict.values
    #y_predict_ = reverse_normalization(y_train_orig,y_predict_,cols_orig)
    #y_predict_ = reverse_standardation(x_total,y_predict_,cols_orig)
    y_predict = y_predict_[-predictSize:]
    
    print(len(y_predict))
    #y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    del x_new
    #scale_y= scaler.fit(y_imputate)
    #y_actual = scale_y.transform(y_actual)
    #y_predict = scale_y.transform(y_predict)
    
    print('start is: ',start)

    return y_predict, y_actual
