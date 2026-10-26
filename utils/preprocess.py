import numpy as np
import pandas as pd
import model
from model import Config_ablation_gru, brnn
from model.Config_ablation_gru import fl_config
import utils
from utils.optimization import critical_sample_size,critical_intensity_estimate,critical_sample_size,sample_complexity_lower_bound
import torch
import train
from train import impute
import random
import gc
config = fl_config()
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def dataSplit(orig,timeSequence,opt,cols_orig,na_replacement):
    gc.collect()
    '''
    x_train = np.array(x[-start:-start+trainSize])
    y_train = np.array(y[-start:-start+trainSize])
    x_test = np.array(x[-start+trainSize:-start+trainSize+testSize])
    y_test = np.array(y[-start+trainSize:-start+trainSize+testSize])
    y_actual = np.array(y[-start+trainSize+predictSize:-start+trainSize+testSize+predictSize])
    '''
    print(len(orig))

    orig_mask = orig.drop('TIMEGAP',axis=1)
    #compute sample size


    if config.check_critical_threshold==True:
        critical_threshold=critical_intensity_estimate.critial_threshold_estimate(orig,opt,timeSequence)
        print(f"Approximate critical intensity where variance begins to diverge: {critical_threshold:.2f}")
        with open ('optimized_parameters.log', 'a') as f:
            f.write('----------------------------parameters collection UID {}, data_dir {} ------------------------------\n'.format(str(timeSequence),opt.data_dir))
            f.write('Approximate critical intensity where variance begins to diverge: {} \n'.format(str(critical_threshold)))
            f.close()
    mask, estimate,event  = critical_intensity_estimate.data_mask(orig_mask,0.25)
    print('masked data is: ',mask)

    #mask = datamask(orig)
    #mask=orig
    #print(mask)
    orig = orig.reset_index().drop('index',axis=1)
    #print(orig)

    mask['TIMEGAP']=orig['TIMEGAP']
    #print(mask)
    #df_imts = pd.DataFrame()
    df_imts=mask.replace(np.nan,na_replacement)
    #df_imts=orig.replace(np.nan,-1)
    print(len(mask),len(df_imts))
    data_dir = opt.data_dir

    '''
    for col in df_imts.columns:
        df_imts =df_imts[df_imts[col]!=-1 ]
    '''
    orig_sample = orig.replace(np.nan,0)
    orig_sample_ = orig_sample.drop('TIMEGAP',axis=1)
    x_total = orig_sample_[:-1]
    y_total = orig_sample_[1:]
    print(len(x_total),len(y_total))
    risk_thread = config.risk_thread

    critical_sampleSize = critical_sample_size(orig_sample_,risk_thread)
    dataset = np.concatenate([x_total,y_total],axis=1)
    minimum_samples = sample_complexity_lower_bound.compute_sample_lower_bound( x_total,y_total, dataset,config.poolSize)
    print(minimum_samples)
    print(critical_sampleSize)
    if minimum_samples>critical_sampleSize:
        sampleSize=minimum_samples
    print(sampleSize)
    with open ('optimized_parameters.log', 'a') as f:
        f.write('Minimum sample size of {} and {}: {} \n'.format(str(sampleSize),str(minimum_samples),str(data_dir)))
        f.close()

    trainSize = int(sampleSize)
    testSize = int(sampleSize*0.25)
    predictSize = int(max(testSize*0.1,10))

    print(sampleSize,trainSize,testSize,predictSize)



    totalSize = trainSize+testSize+predictSize
    #start = random.randint(totalSize+1,len(orig)-1)
    start = random.randint(totalSize+1,len(orig)-1)
    #start = len(orig)-1
    #start = 1378
    
    print('---------------------------------imts is ------------------------------')
    print(df_imts)
    #df_imts = df_imts.drop(('index'),axis=1)
    x = df_imts[-start-1:-1]
    y = df_imts[-start:]
    y_orig = orig.drop('TIMEGAP',axis=1)[-start:]
    #x = x.reset_index()
    #y = y.reset_index()
    y_orig.to_csv('orig.csv')

    '''
    if  device != torch.device("cpu"):
        x = torch.tensor(x.values).to(device).float()
        y = torch.tensor(y.values).to(device).float()
    
    #print(len(total))

    
    #x_impute = impute.impute(x,na_replacement)
    #y_impute = impute.impute(y,na_replacement)
    x_impute = x
    y_impute = y
    print(x_impute)
    print(y_impute)
    x_train = np.array(x_impute[:trainSize]).astype(np.float32)
    y_train = np.array(y_impute[:trainSize]).astype(np.float32)
    x_test = np.array(x_impute[trainSize:trainSize+testSize]).astype(np.float32)
    y_test = np.array(y_impute[trainSize:trainSize+testSize]).astype(np.float32)
    x_sample = np.array(x[trainSize:trainSize+testSize+predictSize]).astype(np.float32)
    y_actual = np.array(y[trainSize+testSize:trainSize+testSize+predictSize]).astype(np.float32)
    x= np.array(x)
    y= np.array(y)
    '''
    x_train,y_train,x_test,y_test,x_impute=impute.brnn_impute(x,y,df_imts,start,trainSize,testSize,timeSequence,opt,cols_orig,na_replacement)
    #x_train,y_train,x_test,y_test,x_impute=impute.brnn_impute(x,y,start,timeSequence,opt,cols_orig)
    y_actual = np.array(y_orig[trainSize+testSize:trainSize+testSize+predictSize]).astype(np.float32)
    
    #y_actual = scaler_y.transform(y_actual)

    #print(len(x_total))
    #print(len(y_total))


    
    print(len(x),len(y),len(x_train),len(y_train),len(x_test),len(y_test),len(y_actual))
    #print(x_train,y_train,x_test,y_test,y_actual)
    
    #print(y_train)
    #x_actual = np.array(x[-start+trainSize+testSize-predictSize:-start+trainSize+testSize+predictSize]
    #x_train,x_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,shuffle=True)

    return  x,y,x_impute,x_train,y_train,x_test,y_test,y_actual,start, trainSize,testSize,predictSize,sampleSize
def datamask(data):
    count = 0
    data = data.reset_index()
    '''
    y_mask = np.random.rand(len(data))  < 0.05
    for i in range(len(data)):
        if y_mask[i] == True:
            data.loc[i:i+5,:] =  np.nan
        i+=1
    '''
    for col in data.columns:
        y_mask = np.random.rand(len(data))  < 0.25
        print(y_mask)
        for i in range(len(data)):
            if y_mask[i] == True:

                data.loc[i,col] = np.nan
                '''
                if col == 'SNOW':
                    data.loc[i,'SNWD'] =  np.nan
                elif col == 'SNWD':
                    data.loc[i,'SNOW'] =  np.nan
                '''
                count +=1

            i+=1
    
    data = data.drop(('index'),axis=1)
    print(data)
    return data

