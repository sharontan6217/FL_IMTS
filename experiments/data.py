import pandas as pd
import numpy as np
import model
from model.Config import fl_config
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random
import time

import gc

scaler = StandardScaler()
#scaler = MinMaxScaler()
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
    return orig,cols_orig
def mimicicu_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)    
    df_orig = df_orig.drop(['hadm_id','flag'],axis=1)
    df_filtered = df_orig[['itemid','valuenum']]
    df_filtered = df_filtered.groupby(['itemid']).filter(lambda x: len(x)>2000).reset_index()
    df_item = df_filtered[['itemid']].drop_duplicates()
    item_list = df_item['itemid'].values
    for i in range(len(item_list)):
        item = item_list[i]
        if i==0:
            df_merged = df_orig[df_orig['itemid']==item]
        else:
            df_temp = df_orig[df_orig['itemid']==item]
            df_merged = df_merged.merge(df_temp,how='outer',on=['subject_id','charttime'],suffixes=('', '_'+str(item)))

    #df_merged=df_merged.replace(np.nan,-9999)
    df_merged.to_csv('merged.csv')  
    df=pd.DataFrame()
    for item in item_list:
        print(item)
        item = str(item)
        for col in df_merged.columns:
            if item in col :
                col_name = 'valuenum_'+item
                print(col_name)
                #print(df_merged[['valuenum_50882']])
                df[item]=df_merged[[col_name]]

    randint = random.randint(50,len(df)-poolSize+1)
    
    orig = df[-randint-poolSize:-randint]
    orig.to_csv('orig_mimic.csv')
    print(orig)
    cols_orig = df.columns
    print(cols_orig)
    return orig,cols_orig 
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
    return orig,cols_orig

def test_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0,na_filter=True)  
    print(df_orig.columns) 
    df = df_orig[['ACLIgG','ACLIgM','25-VITD3','25-VITD','LA']]
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
    return orig,cols_orig
def finance_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0,na_filter=True)  
    print(df_orig.columns) 
    df = df_orig[['823 | Share Price (Daily)(HK$)','Gold Price','Treasury 5 years Yield']]
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
    return orig,cols_orig