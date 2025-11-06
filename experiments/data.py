import pandas as pd
import numpy as np
import model
from model.Config import fl_config
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random
import time

import gc

#scaler = StandardScaler()
scaler = MinMaxScaler()
config = fl_config()
poolSize = config.poolSize
def airquality_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)    
    #df_merged = df_merged.replace('-9999',np.nan)
    #df_merged = df_merged.dropna()
    df = df_orig.drop(['stationName','longitude','latitude','utc_time'],axis=1)

    print(len(df))
    '''
    unimputated = df[-randint-poolSize:-randint]
    for col in df.columns:
        for i in range(len(df)):
            try:
                if df.loc[i,col]==-9999:
                    if df.loc[i-1,col]!= -9999 and df.loc[i+1,col]!=-9999:
                        df.loc[i,col] = np.mean([0.8*df.loc[i-1,col],1.2*df.loc[i+1,col]])
                    else:
                        df.loc[i,col]  =np.mean(df.loc[i-8:i-1,col])
                else:
                    continue
            except Exception as e:
                    #print(e)
                    df.loc[i,col]  =np.mean(df.loc[i-8:i-1,col])
    '''
    randint = random.randint(0,len(df)-poolSize+1)
    
    orig = df[-randint-poolSize:-randint]
    print(orig)
    cols_orig = df.columns
    print(cols_orig)
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig
def mimicicu_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)    
    df_orig = df_orig.drop(['hadm_id','flag'],axis=1)
    df_50971 = df_orig[df_orig['itemid']==50971]
    df_50868 = df_orig[df_orig['itemid']==50868]
    df_50882 = df_orig[df_orig['itemid']==50882]
    df_50902 = df_orig[df_orig['itemid']==50902]
    df_50912 = df_orig[df_orig['itemid']==50912]
    df_50931 = df_orig[df_orig['itemid']==50931]
    df_50983 = df_orig[df_orig['itemid']==50983]
    df_51006 = df_orig[df_orig['itemid']==51006]
    df_51221 = df_orig[df_orig['itemid']==51221]
    df_merged = df_50971.merge(df_50868,how='outer',on=['subject_id','charttime'],suffixes=('', '_50868')).merge(df_50882,how='outer',on=['subject_id','charttime'],suffixes=('', '_50882')).merge(df_50902,how='outer',on=['subject_id','charttime'],suffixes=('', '_50902')).merge(df_50912,how='outer',on=['subject_id','charttime'],suffixes=('', '_50912')).merge(df_50931,how='outer',on=['subject_id','charttime'],suffixes=('', '_50931')).merge(df_50983,how='outer',on=['subject_id','charttime'],suffixes=('', '_50983')).merge(df_51006,how='outer',on=['subject_id','charttime'],suffixes=('', '_51006')).merge(df_51221,how='outer',on=['subject_id','charttime'],suffixes=('', '_51221'))

    df_merged = df_merged.reset_index()
    df_merged = df_merged.drop(('index'),axis=1)
    print(df_merged)
    data_50971 = []
    data_50868 = []
    data_50882 = []
    data_50902 = []
    data_50912 = []
    data_50931 = []
    data_50983 = []
    data_51006 = []
    data_51221 = []
    for i in range(len(df_merged)):
        for j in range(len(df_merged.columns)):
            col = df_merged.columns[j]
            if 'valuenum' in col.lower():
                value = df_merged.loc[i,col]
                if '_50868' in col.lower():
                    data_50868.append(value)
                elif '_50882' in col.lower():
                    data_50882.append(value)
                elif '_50902' in col.lower():
                    data_50902.append(value)
                elif '_50912' in col.lower():
                    data_50912.append(value)
                elif '_50931' in col.lower():
                    data_50931.append(value)
                elif '_50983' in col.lower():
                    data_50983.append(value)
                elif '_51006' in col.lower():
                    data_51006.append(value)
                elif '_51221' in col.lower():
                    data_51221.append(value)
                else:
                    data_50971.append(value)

    df = pd.DataFrame()
    df['Potassium'] = data_50971
    df['Anion Gap'] = data_50868
    df['Bicarbonate'] = data_50882
    df['Chloride'] = data_50902
    df['Creatinine'] = data_50912
    df['Glucose'] = data_50931

    df['Sex Hormone Binding Globulin'] = data_50983
    df['Urea Nitrogen'] = data_51006
    df['Hematocrit'] = data_51221
    '''
    df_patched = df
    for col in df_patched.columns:
        for i in range(len(df_patched)):
            try:
                if np.isnan(df_patched.loc[i,col])==True:
                    if np.isnan(df_patched.loc[i-1,col])==False and np.isnan(df_patched.loc[i+1,col])==False:
                        df_patched.loc[i,col] = np.mean([0.8*df_patched.loc[i-1,col],1.2*df_patched.loc[i+1,col]])
                    else:
                        df_patched.loc[i,col]  =np.mean(df_patched.loc[i-8:i-1,col])
                else:
                    continue
            except Exception as e:
                    #print(e)
                    df_patched.loc[i,col]  =np.mean(df_patched.loc[i-8:i-1,col])
    
    orig = df_patched[-poolSize-1:-1]
    '''
    print(len(df))
    randint = random.randint(0,len(df)-poolSize+1)
    
    orig = df[-randint-poolSize:-randint]
    print(orig)
    cols_orig = df.columns
    print(cols_orig)
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig
def ecg_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0,na_filter=True)  
    print(df_orig.columns) 

    df_ = df_orig.replace(-9999,np.nan)
    df_ = df_.dropna()
    df_ = df_.reset_index()
    data_01 = []
    data_02 = []
    data_03 = []
    data_04 = []
    for i in range(len(df_)):
        for j in range(len(df_.columns)):
            col = df_.columns[j]
            if 'filtered' in col.lower():
                value = df_.loc[i,col]
                if '_01' in col.lower():
                    data_01.append(value)
                elif '_02' in col.lower():
                    data_02.append(value)
                elif '_03' in col.lower():
                    data_03.append(value)
                elif '_04' in col.lower():
                    data_04.append(value)
    df = pd.DataFrame()
    df['Person_01'] = data_01
    df['Person_02'] = data_02
    df['Person_03'] = data_03
    df['Person_04'] = data_04 
    print(len(df))
    randint = random.randint(0,len(df)-poolSize+1)
    orig = df[-randint-poolSize:-randint]
    cols_orig = df.columns
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig
def uci_dataLoad(train_data_dir,test_data_dir):
    global cols_orig
    df_train = pd.read_csv(train_data_dir,header=0,na_filter=True)  
    print(df_train.columns) 
    
    df_test = pd.read_csv(test_data_dir,header=0,na_filter=True)  
    df = pd.concat((df_train,df_test),axis=0)
    print(df.columns) 
    df = df[['tBodyAcc-mean-X','tBodyAcc-mean-Y','tBodyAcc-mean-Z','tBodyAcc-std-X','tBodyAcc-std-Y','tBodyAcc-std-Z']]
    cols_orig = df.columns

    randint = random.randint(0,len(df)-poolSize+1)
    orig = df[-randint-poolSize:-randint]
    cols_orig = df.columns
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig

def eeg_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0,na_filter=True)  
    print(df_orig.columns) 
    df = df_orig[['P4','Cz','F8','T7']]
    cols_orig = df.columns

    '''
    data = []
    for i in range(len(df)):
        for j in range(len(df.columns)):
            col = df.columns[j]
            if 'f8' in col.lower():
                value = df.loc[i,col]
                data.append(value)
    '''
    orig = df[-poolSize:]
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig