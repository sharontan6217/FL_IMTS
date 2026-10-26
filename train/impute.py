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
import tensorflow as tf
import model
from model import Config_ablation_gru, brnn
from model.Config_ablation_gru import brnn_config,fl_config
import matplotlib.pyplot as plt
import random
import torch
from framework import federated_learning_nn
import gc

#scaler = StandardScaler()
scaler = MinMaxScaler()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def brnn_impute(x,y,df_imts,start,trainSize,testSize,timeSequence,opt,cols_orig,na_replacement):
    gc.collect()
    timegaps_x = x['TIMEGAP']
    timegaps_y = y['TIMEGAP']
    x = x.drop(['TIMEGAP'],axis=1)
    y = y.drop(['TIMEGAP'],axis=1)
    df_imts = df_imts.drop(['TIMEGAP'],axis=1)
    #cols_orig = df_imts.columns
    graph_dir = opt.graph_dir
    save_dir = opt.save_dir
    data_category = opt.data_dir.split('/')[2]
    data_dir = './data/'+data_category+'/GRU/medium dataset(2000-1000)/'
    print(data_dir)
    if os.path.exists(data_dir)==False:
        os.makedirs(data_dir)



    impute_dir = data_dir+data_category+'_'+'impute.csv'
    print(impute_dir)
    if os.path.exists(impute_dir)==True:
        missing_std = pd.read_csv(impute_dir)
    else:

        x_estimate = impute(x,na_replacement)
        estimate = np.array(x_estimate)
        #estimate = x_estimate
        print(x)
        print(estimate)
        #df_imts_ = df_imts.replace(na_replacement,np.nan).dropna()
        #imputed = np.nan
        x.to_csv(data_dir+data_category+'_'+'orig_before.csv')
        df_imts_ = df_imts.replace(na_replacement,np.nan).dropna()
        print('length of imts after dropna: ', len(df_imts_))
        print(df_imts_)
        print(len(df_imts))
        #df_imts_ = df_imts.replace(na_replacement,0).dropna()
        imputed = np.nan
        scaler = MinMaxScaler()
        if len(df_imts_)<0.2*len(df_imts):
            df_imts_ = df_imts.replace(na_replacement,0).dropna()
            imputed = 0
            scaler=StandardScaler()
        print(scaler)
        print(imputed)
        #df_imts_ = df_imts.replace(na_replacement,0).dropna()
        #df_imts_ = df_imts.replace(na_replacement,np.nan).dropna()
        print(len(y.replace(na_replacement,np.nan).dropna()))
        x_ = df_imts_[-start-1:-1]
        y_ = df_imts_[-start:]

        x_.to_csv(data_dir+data_category+'_'+'filtered.csv')
        print(len(x_))
        print(len(y_))
        trainSize_real = int(len(x_)*0.8)
        testSize_real = len(x_)-trainSize_real

        
        df_missing = x.replace(na_replacement,np.nan)
        df_missing_y= y.replace(na_replacement,np.nan)
        df_missing.to_csv('missing.csv')

        missing = df_missing.copy()
        missing_y = df_missing_y.copy()

        if any(df_missing.isnull())==True:

            for col in missing.columns:
                non_null_mask = missing[col].notna()
                missing.loc[non_null_mask,col]=scaler.fit_transform(missing.loc[non_null_mask,col].values.reshape(-1,1)).flatten()
                
            for col in missing_y.columns:
                non_null_mask = missing_y[col].notna()
                missing_y.loc[non_null_mask,col]=scaler.fit_transform(missing_y.loc[non_null_mask,col].values.reshape(-1,1)).flatten()



            x_ = scaler.fit_transform(np.array(x_))
            estimate = scaler.transform(x_estimate)
            x_train_ = x_[:trainSize_real]
            x_test_ = x_[trainSize_real:trainSize_real+testSize_real]
            timegaps_x_train = timegaps_x[:trainSize_real]
            timegaps_y_train = timegaps_y[:trainSize_real]



            y_ = scaler.transform(np.array(y_))
            y_train_ = y_[:trainSize_real]
            y_test_ = y_[trainSize_real:trainSize_real+testSize_real]


            print(len(x_train_),len(y_train_),len(x_test_),len(y_test_))

            with open (data_dir+'train_impute.log','w') as f:
                f.write('-----------------x_train is-------------\n')
                f.write(str(x_train_))
                f.write('-----------------y_train is-------------\n')
                f.write(str(y_train_))
                f.write('-----------------x_test is-------------\n')
                f.write(str(x_test_))
                f.write('-----------------y_test is-------------\n')
                f.write(str(y_test_))
                f.close()
            if 'index' in x.columns:
                x = x.drop(('index'),axis=1)
            brnn_graph_dir=graph_dir+'accuracy/impute/'
            if os.path.exists(brnn_graph_dir)==False:
                os.makedirs(brnn_graph_dir)
            missing.to_csv(data_dir+data_category+'_'+'missing.csv')

            print('-------------------start imputation-------------------')
            impute_model_dir = save_dir+'impute.keras'
            if os.path.exists(impute_model_dir)==True:
                model_impute = tf.keras.models.load_model(impute_model_dir)
            else:
                client_datasets,test_datasets=federated_learning_nn.dataProcess_impute(x_train_,y_train_,x_test_,y_test_,timegaps_x_train,timegaps_y_train)
                state,metrics,loss,mae = federated_learning_nn.train(client_datasets)
                model_impute,test_metrics=federated_learning_nn.eval(test_datasets,state)
                #model_impute.save(save_dir+'impute.keras')
                fig1 = federated_learning_nn.fl_visualize(loss,mae,timeSequence,start,brnn_graph_dir)
            print(len(estimate),len(missing))
            missing = np.array(missing)

            for i in range(len(missing)):
                gc.collect()
                for j in range(len(missing[i])):
                    print(i,j,missing[i][j])
                    if np.isnan([missing[i][j]]) == True:
                        print('missing cell is:')
                        print(i,missing[i][j])
                        
                        diff=[]
                        imputed_value=[]
                        if i<8:
                            missing[i][j]=estimate[i][j]
                        else:
                            pseudo_x_ = missing[:i-1]
                            length = len(pseudo_x_)
                            #print(length)
                            x_train_fl_ = fl_convertion(pseudo_x_).astype(np.float32)    
                            pseudo_y_ = model_impute.predict(x_train_fl_).astype(np.float32) # Generate pseudo-labels

                            #print(len(pseudo_y_))
                            df_predict = pd.DataFrame()

                            for n in range(len(cols_orig)):
                                col_name = cols_orig[n]
                                if n==0:
                                    df_predict[col_name]=np.array(pseudo_y_ ).reshape(-1,)[:length]
                                else:
                                    df_predict[col_name]=np.array(pseudo_y_ ).reshape(-1,)[length*n:length*(n+1)]
                                
                            #print(df_predict)
                            missing[i][j]=df_predict.values[-1][j]

                            
                            for iteration in range(5):  # Run 5 iterations
                                pseudo_x_ = missing[:i-1]
                                length = len(pseudo_x_)
                                #print(length)
                                x_train_fl_ = fl_convertion(pseudo_x_).astype(np.float32)    
                                pseudo_y_ = model_impute.predict(x_train_fl_).astype(np.float32) # Generate pseudo-labels
                                #print(len(pseudo_y_))
                                df_predict = pd.DataFrame()
                                #print(len(cols_orig))
                                
                                #df_predict['P4']=np.array(pseudo_y_ ).reshape(-1,)[:length]
                                #df_predict['Cz']=np.array(pseudo_y_ ).reshape(-1,)[length:length*2]
                                #df_predict['F8']=np.array(pseudo_y_ ).reshape(-1,)[length*2:length*3]
                                #df_predict['T7']=np.array(pseudo_y_ ).reshape(-1,)[length*3:]
                                
                                for n in range(len(cols_orig)):
                                    #print(cols_orig[n])
                                    col_name = cols_orig[n]
                                    if n==0:
                                        df_predict[col_name]=np.array(pseudo_y_ ).reshape(-1,)[:length]
                                    else:
                                        #print(df_predict[col_name])
                                        #print(pseudo_y_)
                                        #print(np.array(pseudo_y_ ).reshape(-1,)[length*n:length*(n+1)])
                                        df_predict[col_name]=np.array(pseudo_y_ ).reshape(-1,)[length*n:length*(n+1)]
                                
                                #print(df_predict)
                                missing_value=df_predict.values[-1][j]

                                diff_ = abs(estimate[i][j]-missing_value)      
                                diff.append(diff_)
                                imputed_value.append(missing_value)
                                iteration+=1
                            with open (data_dir+data_category+'_'+'impute.txt','a') as f:
                                f.write(str(imputed_value))
                                f.close()
                            with open (data_dir+data_category+'_'+'diff.txt','a') as f:
                                f.write(str(diff))
                                f.close()
                            
                            missing[i][j] = imputed_value[np.argmin(diff)]
                            
                        print(i,missing[i][j])
            del model_impute  
        
        if imputed == 0:   
        
            missing_std = reverse_standardation(x, missing,cols_orig,na_replacement)
        else:
            missing_std = reverse_normalization(x, missing,cols_orig,na_replacement)
        
        x.to_csv(data_dir+data_category+'_'+'orig_after.csv')
        #missing_std = reverse_normalization(x, missing,cols_orig)        
        #missing_std = reverse_standardation(x, missing,cols_orig)
        #missing_std = missing
        #missing_std = estimate
        print(missing_std)
        df_impute = pd.DataFrame(data=missing_std)
        df_impute.to_csv(impute_dir,index=False)
        
    #missing_std = estimate
    x_train = missing_std[:trainSize].astype(np.float32)
    y_train = missing_std[1:trainSize+1].astype(np.float32)
    x_test = missing_std[trainSize:trainSize+testSize].astype(np.float32)
    y_test = missing_std[trainSize+1:trainSize+testSize+1].astype(np.float32)

    print(len(x_train),len(y_train),len(x_test),len(y_test))
    x_impute = missing_std
    print(x_impute)

    

    return x_train,y_train,x_test,y_test,x_impute

def impute(df,na_replacement):
    if 'index' in df.columns:
        df = df.drop(('index'),axis=1)
    df = df.reset_index()
    for col in df.columns:
        for i in range(len(df)):
            try:
                if i==0:
                    estimate_matrix = [item for item in df.loc[i:,col] if item!=na_replacement ][:7]
                    df.loc[i,col]  =np.mean(estimate_matrix)   
                else:  
                    if df.loc[i,col]==na_replacement:
                        if df.loc[i-1,col]!=na_replacement and df.loc[i+1,col]!=na_replacement:
                            matrix_before =  [item for item in df.loc[:i-1,col] if item !=na_replacement ][-3:]
                            matrix_after =  [item for item in df.loc[i+1:,col] if item !=na_replacement ][:3]
                            df.loc[i,col] = 0.8*np.mean(matrix_before)+1.2*np.mean(matrix_after)
                        else:
                            estimate_matrix = [item for item in df.loc[max(0,i-1):,col] if item!=na_replacement ][:7]
                            df.loc[i,col]  =np.mean(estimate_matrix)     
                    else:
                        continue
            except Exception as e:
                print(e)              
 
    df = df.drop(('index'),axis=1)
    return df

