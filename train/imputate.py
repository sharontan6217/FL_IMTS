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
from model import Config, brnn
from model.Config import brnn_config,fl_config
import matplotlib.pyplot as plt
import random
from framework import federated_learning_nn
import gc

#scaler = StandardScaler()
scaler = MinMaxScaler()

config = fl_config()

def brnn_imputate(x,y,start,timeSequence,opt,cols_orig):
    gc.collect()
    trainSize = config.trainSize
    testSize = config.testSize
    predictSize = config.predictSize
    graph_dir = opt.graph_dir
    

    x_estimate = imputate(x,imputated_value=-1)
    estimate = np.array(x_estimate)
    #estimate = x_estimate
    print(x)
    print(estimate)
    
    print(len(y.replace(-1,np.nan).dropna()))
    x_ = x.replace(-1,np.nan).dropna().astype(np.float32)
    y_ = y.replace(-1,np.nan).dropna().astype(np.float32)
    x_.to_csv('filtered.csv')
    print(len(x_))
    print(len(y_))

    x_train_ = x_[:trainSize]
    x_test_ = x_[trainSize:trainSize+testSize]
    y_train_ = y_[:trainSize]
    y_test_ = y_[trainSize:trainSize+testSize]

    df_missing = x.replace(-1,np.nan)
    df_missing_y= y.replace(-1,np.nan)
    missing = df_missing.copy()
    missing_y = df_missing_y.copy()


    for col in missing.columns:
        non_null_mask = missing[col].notna()
        missing.loc[non_null_mask,col]=scaler.fit_transform(missing.loc[non_null_mask,col].values.reshape(-1,1)).flatten()
    
    for col in missing_y.columns:
        non_null_mask = missing_y[col].notna()
        missing_y.loc[non_null_mask,col]=scaler.fit_transform(missing_y.loc[non_null_mask,col].values.reshape(-1,1)).flatten()
    

    x_ = scaler.fit_transform(np.array(x_))
    estimate = scaler.transform(x_estimate)
    x_train_ = x_[:trainSize]
    x_test_ = x_[trainSize:trainSize+testSize]

    y_ = scaler.transform(np.array(y_))
    y_train_ = y_[:trainSize]
    y_test_ = y_[trainSize:trainSize+testSize]

    y_actual_ = missing_y[trainSize+testSize:trainSize+testSize+predictSize]
    
    #y_actual_ = scaler.transform(y_actual_)
    print(len(x_train_),len(y_train_),len(x_test_),len(y_test_),len(y_actual_))
    print(y_actual_)
    if 'index' in x.columns:
        x = x.drop(('index'),axis=1)
    brnn_graph_dir=graph_dir+'accuracy/'
    if os.path.exists(brnn_graph_dir)==False:
        os.makedirs(brnn_graph_dir)
    missing.to_csv('missing.csv')
    if any(missing.isnull())==True:
        print('-------------------start imputation-------------------')
        client_datasets,test_datasets=federated_learning_nn.dataProcess(x_train_,y_train_,x_test_,y_test_)
        state,metrics,loss,mae = federated_learning_nn.train(client_datasets)
        model_imputate,test_metrics=federated_learning_nn.eval(test_datasets,state,metrics)
        fig = federated_learning_nn.fl_visualize(loss,mae,timeSequence,start,brnn_graph_dir)
        print(len(estimate),len(missing))
        missing = np.array(missing)

        for i in range(len(missing)):
            for j in range(len(missing[i])):
                print(i,j,missing[i][j])
                if np.isnan([missing[i][j]]) == True:
                
                    print(i,missing[i][j])
                    diff=[]
                    imputated_value=[]
                    if i<8:
                        missing[i][j]=estimate[i][j]
                    else:
                        for iteration in range(5):  # Run 5 iterations
                            pseudo_x_ = missing[:i-1]
                            length = len(pseudo_x_)
                            x_train_fl_ = fl_convertion(pseudo_x_).astype(np.float32)    
                            pseudo_y_ = model_imputate.predict(x_train_fl_).astype(np.float32) # Generate pseudo-labels
                            df_predict = pd.DataFrame()
                            #print(len(cols_orig))
                            '''
                            df_predict['P4']=np.array(pseudo_y_ ).reshape(-1,)[:length]
                            df_predict['Cz']=np.array(pseudo_y_ ).reshape(-1,)[length:length*2]
                            df_predict['F8']=np.array(pseudo_y_ ).reshape(-1,)[length*2:length*3]
                            df_predict['T7']=np.array(pseudo_y_ ).reshape(-1,)[length*3:]
                            '''
                            for n in range(len(cols_orig)):
                                #print(cols_orig[n])
                                col_name = cols_orig[n]
                                if n==0:
                                    df_predict[col_name]=np.array(pseudo_y_ ).reshape(-1,)[:length]
                                else:
                                    df_predict[col_name]=np.array(pseudo_y_ ).reshape(-1,)[length*n:length*(n+1)]

                            print(df_predict)
                            missing_value=df_predict.values[-1][j]
                            diff_ = abs(estimate[i][j]-missing_value)      
                            diff.append(diff_)
                            imputated_value.append(missing_value)
                        with open ('imputate.txt','a') as f:
                            f.write(str(imputated_value))
                            f.close()
                        with open ('diff.txt','a') as f:
                            f.write(str(diff))
                            f.close()
                        missing[i][j] = imputated_value[np.argmin(diff)]
                    print(i,missing[i][j])
        del model_imputate
    x_train = missing[:trainSize].astype(np.float32)
    y_train = missing[1:trainSize+1].astype(np.float32)
    x_test = missing[trainSize:trainSize+testSize].astype(np.float32)
    y_test = missing[trainSize+1:trainSize+testSize+1].astype(np.float32)
    y_actual = y_actual_.astype(np.float32)
    print(len(x_train),len(y_train),len(x_test),len(y_test),len(y_actual))
    x_imputate = missing
    print(x_imputate)
    df_imputate = pd.DataFrame(data=missing)
    df_imputate.to_csv('imputate.csv')
    
    

    return x_train,y_train,x_test,y_test,y_actual,x_imputate
def imputate(df,imputated_value):
    if 'index' in df.columns:
        df = df.drop(('index'),axis=1)
    df = df.reset_index()
    for col in df.columns:
        for i in range(len(df)):
            try:
                if df.loc[i,col]==imputated_value:
                    if df.loc[i-1,col]!=imputated_value and df.loc[i+1,col]!=imputated_value:
                        matrix_before =  [item for item in df.loc[:i-1,col] if item !=imputated_value ][-3:]
                        matrix_after =  [item for item in df.loc[i+1:,col] if item !=imputated_value ][:3]
                        df.loc[i,col] = 0.8*np.mean(matrix_before)+1.2*np.mean(matrix_after)
                    else:
                        estimate_matrix = [item for item in df.loc[max(0,i-1):,col] if item!=imputated_value ][:7]
                        df.loc[i,col]  =np.mean(estimate_matrix)     
                else:
                    continue
            except Exception as e:
                print(e)              
 
    df = df.drop(('index'),axis=1)
    return df

