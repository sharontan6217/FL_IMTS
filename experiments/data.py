import pandas as pd
import numpy as np
import model
from model.Config import fl_config
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import random
import time

import gc

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
    gc.collect()
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)
    print(len(df_orig))
    df = df_orig[110000:130000]
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
                #print(df_merged[['valuenum_50882']])
                df[item]=df_merged[[col_name]]
    df.to_csv('mimic_preprocessed.csv',chunksize=10000)
    '''
    randint = random.randint(0,len(df)-poolSize+1)
    orig = df[-randint-poolSize:-randint]   
    

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
    randint = random.randint(0,len(df)-poolSize+1)
    orig = df[-randint-poolSize:-randint]
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

    randint = random.randint(0,len(df)-poolSize+1)
    orig = df[-randint-poolSize:-randint]
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
def climate_dataLoad(data_dir):
    global cols_orig
    df_orig = pd.read_csv(data_dir,header=0)    
    #df_merged = df_merged.replace('-9999',np.nan)
    #df_merged = df_merged.dropna()
    df = df_orig.drop(['DATE','COOP_ID','TIME_STAMP'],axis=1)

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
def climate_dataLoad_samples(data_dir):
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