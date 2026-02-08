import pandas as pd
import numpy as np
import model
from model.Config import fl_config
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch import load
from torch.utils.data import Dataset,DataLoader
import os

import random
import time

import gc

config = fl_config()
poolSize = config.poolSize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def airquality_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)    
    #df_merged = df_merged.replace('-9999',np.nan)
    #df_merged = df_merged.dropna()
    df = df_orig.drop(['stationName','longitude','latitude','utc_time'],axis=1)

    print(len(df))
    '''
    unimputed = df[-randint-poolSize:-randint]
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
    orig = df[-poolSize:]
    print(orig)
    cols_orig = df.columns
    print(cols_orig)
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig,cols_orig
def mimicicu_dataLoad(data_dir):
    gc.collect()
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)
    print(len(df_orig))
    df = df_orig
    '''
    item_list=[]
    for col in df_merged.columns:
        if 'ITEMID_' in col:
            item = col.split('_')[1]
            print(item)
            item_list.append(item)

    


    df_orig = df_orig.drop(['ROW_ID','HADM_ID','VALUE','FLAG'],axis=1)
    df_filtered = df_orig[['ITEMID','VALUENUM']]
    df_filtered = df_filtered.groupby(['ITEMID']).filter(lambda x: len(x)>50000).reset_index()
    df_item = df_filtered[['ITEMID']].drop_duplicates()
    item_list = df_item['ITEMID'].values
    print(len(item_list))
    for i in range(len(item_list)):
        item = item_list[i]
        if i==0:
            df_merged = df_orig[df_orig['ITEMID']==item]
        else:
            df_temp = df_orig[df_orig['ITEMID']==item]
            df_merged = df_merged.merge(df_temp,how='outer',on=['SUBJECT_ID','CHARTTIME'],suffixes=('', '_'+str(item)))

    #df_merged=df_merged.replace(np.nan,-9999)
    df_merged.to_csv('merged.csv',chunksize=10000)  
    
    print(df_merged[:10000])
    df_merged[2000:10000].to_csv('samples.csv')
    df=pd.DataFrame()
    for item in item_list:
        print(item)
        item = str(item)
        for col in df_merged.columns:
            if item in col :
                col_name = 'VALUENUM_'+item
                print(col_name)
                df[item]=df_merged[[col_name]]
    df.to_csv('mimic_preprocessed.csv',chunksize=10000)
    '''
    orig = df[-poolSize:]
    

    orig = orig.reset_index()
    orig = orig.drop(['Unnamed: 0','index','50800','50802','50804','50818','50821'],axis=1)
    orig.to_csv('orig_mimic.csv')
    
    print(orig)
    cols_orig = orig.columns
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
    orig = df[-poolSize:]
    cols_orig = df.columns
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig,cols_orig
def uci_dataLoad(train_data_dir,test_data_dir):
    global cols_orig
    df_train = pd.read_csv(train_data_dir,header=0,na_filter=True)  
    print(df_train.columns) 
    
    df_test = pd.read_csv(test_data_dir,header=0,na_filter=True)  
    df = pd.concat((df_train,df_test),axis=0)
    print(df.columns) 
    df = df[['tBodyAcc-mean-X','tBodyAcc-mean-Y','tBodyAcc-mean-Z','tBodyAcc-std-X','tBodyAcc-std-Y','tBodyAcc-std-Z']]

    cols_orig = df.columns

    orig = df[-poolSize:]
    cols_orig = df.columns
    #y_orig = data[-poolSize:]
    #x = np.array(x).reshape(-1,1)
    #y = np.array(y).reshape(-1,1)
    #print(len(data),len(x_orig),len(y_orig))
    return orig,cols_orig
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
def physionet_dataLoad(data_dir):
    global cols_orig

    orig=[]
    if  device == torch.device("cpu"):
        for r,d,f in os.walk(data_dir):
            for f_ in f:
                file_dir = r+'/'+f_
                orig_ = torch.load(file_dir,map_location=torch.device('cpu'))
                print(len(orig_))
                for i in range(len(orig_)):
                    orig.append(orig_[i])
    else:        
        for r,d,f in os.walk(data_dir):
            for f_ in f:
                file_dir = r+'/'+f_
                orig_ = torch.load(file_dir)
                for i in range(len(orig_)):
                    
                    orig.append(orig_[i])
    df_orig =  pd.DataFrame(orig,columns=['file_id','a1','b2','c3'])
    print(df_orig['b2'])


    #df_orig = pd.read_csv(data_dir,header=0,na_filter=True)  
    #print(df_orig)
    #df_orig.to_csv('physionet_orig.csv')

    b2=[]
    df_value=pd.DataFrame()
    for i in range(len(df_orig)):
        #print(b2)
        #file_id = df_orig['file_id'][i]      
        b2_array= df_orig.loc[i,'b2'].numpy()       
        #c3_array = df_orig.loc[i,'c3'].numpy()
        
        for j in range(len(b2_array)):
            #print(b2_item)
            b2_item=b2_array[j]
            #print(b2_item)
            count = 0
            for item in b2_item:
                if item>0:
                    count+=1
            #print(count)
            if count>10:
                b2.append(b2_item)
        i+=1
        '''
        for c3_item in c3_array:
            #print(c3_item)
            c3.append(c3_item)
        '''
    
      
    df_value = pd.DataFrame(b2)
    df_value.to_csv('physionet_value.csv',chunksize=1000)
    for col in df_value.columns:
        if sum(df_value[col])==0:
            df_value=df_value.drop(col,axis=1)


    #df['c3']=c3
    print(df_value)
    df_value = df_value.replace(0,np.nan)
    #print(df_value.columns)
    for col in df.columns:
        col_name = 'physionet_'+str(col)
        df[col_name]=df[col]
        df=df.drop(col,axis=1)
    
    cols_orig = df.columns

    orig = df[-poolSize:]

    return orig,cols_orig
def activity_dataLoad(data_dir):
    df_orig = pd.read_csv(data_dir,names=['sensor','code','number','time_stamp','data_1','data_2','data_3','activity_category'])
    df= df_orig.sort_values(['time_stamp','activity_category'])[['data_1','data_2','data_3']]
    print(df)
    cols_orig = df.columns

    orig = df[-poolSize:]
    print(orig)
    return orig,cols_orig
    


def climate_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)
    df_orig.to_csv('orig.csv')
    elements = set(df_orig['ELEMENT'])
    print(elements)
    #print(df_orig)
    df_orig = df_orig[['COOP_ID','YEAR','MONTH','DAY','ELEMENT','VALUE','DATE']]
    print(df_orig)
    df_merged = pd.DataFrame()  
    for element in elements:
        print(element)
        df_merged_ = df_orig[df_orig['ELEMENT']==element]
        if len(df_merged)==0:
            df_merged = df_merged_
            print(df_merged)
            df_merged['VALUE_'+element]=df_merged['VALUE']
        else:
            print(df_merged_)
            df_merged = df_merged.merge(df_merged_,how='outer',on=['COOP_ID','YEAR','MONTH','DAY'],suffixes=('', '_'+element))
            print(df_merged)
        df_merged[element]=df_merged['VALUE_'+element]
        df_merged = df_merged.replace('-9999',np.nan)
        #df_merged = df_merged.dropna()

    print(df_merged)
    sample_coop_id = list(set(df_merged['COOP_ID']))[0]
    df_merged.to_csv('merged.csv')
    df_sample = df_merged[df_merged['COOP_ID']==sample_coop_id]
    df = pd.DataFrame()
    for element in elements:
        df['COOP_ID'] = df_sample['COOP_ID']
        df['YEAR'] = df_sample['YEAR']
        df['MONTH'] = df_sample['MONTH']
        df['DAY'] = df_sample['DAY']
        df[element] = df_sample['VALUE_'+element]
    df = df.sort_values(['COOP_ID','YEAR','MONTH','DAY']).reset_index()
    df.to_csv('clean.csv')



    print(df)
    df = df.drop(['index','COOP_ID','YEAR','MONTH','DAY'],axis=1)

    orig = df[-poolSize:]
    print(orig)
    cols_orig = df.columns
    print(cols_orig)
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