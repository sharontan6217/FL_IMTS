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
from utils.utils import fl_convertion
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
config = fl_config()
model_config = brnn_config()
trainSize = config.trainSize
testSize = config.testSize
predictSize = config.predictSize



def FL_train_nn(x_train,y_train,x_test,y_test,y_actual,x_imputate,cols_orig,timeSequence,start,opt):
    gc.collect()
    len_cols = len(cols_orig)


    scaler_y=scaler.fit(y_train)
    y_train = scaler_y.transform(y_train)
    y_test=scaler_y.transform(y_test)
    y_actual=scaler_y.transform(y_actual)
    
    scaler_x = scaler.fit(x_train)
    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)
    #y_actual=scaler.transform(y_actual)

    #x_imputate = scaler.transform(x_imputate)
    #x_actual = x_imputate[trainSize:trainSize+testSize]
    #x_actual = scaler.transform(x_actual)
    #x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    #x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
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
        '''
        df_predict['P4']=np.array(y_predict_fl).reshape(-1,)[:testSize]
        df_predict['Cz']=np.array(y_predict_fl).reshape(-1,)[testSize:testSize*2]
        df_predict['F8']=np.array(y_predict_fl).reshape(-1,)[testSize*2:testSize*3]
        df_predict['T7']=np.array(y_predict_fl).reshape(-1,)[testSize*3:]
        '''
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
    '''
    df_predict['P4']=np.array(y_predict_fl).reshape(-1,)[:testSize]
    df_predict['Cz']=np.array(y_predict_fl).reshape(-1,)[testSize:testSize*2]
    df_predict['F8']=np.array(y_predict_fl).reshape(-1,)[testSize*2:testSize*3]
    df_predict['T7']=np.array(y_predict_fl).reshape(-1,)[testSize*3:]
    '''
    for n in range(len(cols_orig)):
        print(cols_orig[n])
        col_name = cols_orig[n]
        if n==0:
            df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[:testSize]
        else:
            df_predict[col_name]=np.array(y_predict_fl).reshape(-1,)[testSize*n:testSize*(n+1)]


    print(len(df_predict))
    print(df_predict)
    y_predict = df_predict.values
    y_predict = y_predict[-predictSize:]
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


def FL_train_predict_window(x,y,x_train,y_train,x_test,y_test,y_actual,start):
    
    x_train=scaler.fit_transform(x_train).astype(np.float32)
    x_test=scaler.transform(x_test).astype(np.float32)
    y_train=scaler.fit_transform(y_train).astype(np.float32)
    y_test=scaler.transform(y_test).astype(np.float32)
    y_actual=scaler.transform(y_actual).astype(np.float32)
    x_train=np.reshape(x_train,(len(x_train),1,1))
    x_test=np.reshape(x_test,(len(x_test),1,1))
    client_datasets,test_datasets=federated_learning_nn.dataProcess(x_train,y_train,x_test,y_test)
    state,metrics,loss,mae = federated_learning_nn.train(client_datasets)
    model_predict=federated_learning_nn.eval(test_datasets,state,metrics)
    #meta_optimizer = optim.Adam(param_dict, lr=0.001)
    y_predict = []
    # Dummy tasks for demonstration
    for j in range(predictSize):
        print(j)
        gc.collect()
        if j==0:
            y_predict_ = model_predict.predict(x_test)
            y_predict_ = y_predict_.astype(np.float32)
        else:
            client_datasets,test_datasets=federated_learning_nn.dataProcess(x_train,y_train,x_actual,y_predict_)
            model_predict=federated_learning_nn.eval(test_datasets,state,metrics)
            y_predict_ = model_predict.predict(x_actual)
            y_predict_ = y_predict_.astype(np.float32)
        #print(y_predict_)
        #print(y_predict_[-1])
        x_new=y_predict_[-1].reshape(-1,1)
        x_actual1 = x[trainSize+j+1:trainSize+testSize+j]
        #print(x_actual1)
        x_actual1=scaler.transform(x_actual1)
        x_actual = np.append(x_actual1,x_new,axis=0).astype(np.float32)
        x_actual=np.reshape(x_actual,(len(x_actual),1,1))
        y_predict.append(x_new)
        #x_new=x_predict[-1].reshape(-1,1)
        #x_actual1 = x[-start+j:-start+trainSize+testSize+j-1]
        #x_actual = np.append(x_actual1,x_new,axis=0).reshape(-1,1)

        j=j+1

    #y_predict = y_predict[-predictSize-1:-1]
    y_predict = np.array(y_predict).reshape(-1,1)
    print('original data is: ')
    print(y_actual)
    print('prediction is: ')
    print(y_predict)
    #del x_new
    print('start is: ',start)
    return y_predict, y_actual
def train(model_predict,x,x_train,y_train,x_test,y_test,y_actual,start):

    #data load
    print("#data load:")
    print(np.count_nonzero(x_train),np.count_nonzero(y_train),np.count_nonzero(x_test),np.count_nonzero(y_test))
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    x_train=np.reshape(x_train,(len(x_train),1,1))
    x_test=np.reshape(x_test,(len(x_test),1,1))
    #train

    trainPredict=model_predict.predict(x_train)
    trainScore=math.sqrt(mean_squared_error(y_train,trainPredict)) 
    print('Train Score: %.5f RMSE' % (trainScore))
        
    testPredict=model_predict.predict(x_test)

    testScore=math.sqrt(mean_squared_error(y_test,testPredict)) 
    print('Test Score: %.5f RMSE' % (testScore))
    y_predict = testPredict  
    for j in range(predictSize):
        print(j)
        if j == 0:
            loss_history=model_predict.fit(x_train,y_train,batch_size=model_config.batch_size,epochs=model_config.epochs,verbose=2, validation_data=[x_test,y_test])            
        else:
            loss_history=model_predict.fit(x_train,y_train,batch_size=model_config.batch_size,epochs=model_config.epochs,verbose=2, validation_data=[x_actual,y_predict]) 
        x_new=y_predict[-1].reshape(-1,1)
        x_actual1 = x[-start+trainSize:-start+trainSize+testSize+j]
        x_actual = np.append(x_actual1,x_new,axis=0)
        x_actual=scaler.transform(x_actual1)
        x_actual=np.reshape(x_actual,(len(x_actual),1,1))
        y_predict=model_predict.predict(x_actual)
        j=j+1

    #y_predict = y_predict[-predictSize-1:-1]

        
    return y_actual,y_predict