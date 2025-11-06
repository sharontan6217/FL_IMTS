import numpy as np
import pandas as pd
import model
from model import Config, brnn
from model.Config import fl_config
import train
from train import imputate, predict
import random
config = fl_config()
def dataSplit(orig,timeSequence,opt,cols_orig):
    '''
    x_train = np.array(x[-start:-start+trainSize])
    y_train = np.array(y[-start:-start+trainSize])
    x_test = np.array(x[-start+trainSize:-start+trainSize+testSize])
    y_test = np.array(y[-start+trainSize:-start+trainSize+testSize])
    y_actual = np.array(y[-start+trainSize+predictSize:-start+trainSize+testSize+predictSize])
    '''
    print(len(orig))
    mask = datamask(orig)
    #df_imts = pd.DataFrame()
    df_imts=mask.replace(np.nan,-1)
    #df_imts=orig.replace(np.nan,-1)
    '''
    for col in df_imts.columns:
        df_imts =df_imts[df_imts[col]!=-1 ]
    '''
    trainSize = config.trainSize
    testSize = config.testSize
    predictSize = config.predictSize

    totalSize = trainSize+testSize+predictSize

    totalSize = trainSize+testSize+predictSize
    start = random.randint(totalSize+1,len(orig)-1)
    print('---------------------------------imts is ------------------------------')
    print(df_imts)
    #df_imts = df_imts.drop(('index'),axis=1)
    x = df_imts[-start-1:-1]
    y = df_imts[-start:]
    y_orig = mask[-start:]
    #x = x.reset_index()
    #y = y.reset_index()

    #print(len(total))
    print(x.columns)
    '''
    x_imputate = imputate(x,imputated_value=-1)
    y_imputate = imputate(y,imputated_value=-1)
    x_train = np.array(x_imputate[:trainSize]).astype(np.float32)
    y_train = np.array(y_imputate[:trainSize]).astype(np.float32)
    x_test = np.array(x_imputate[trainSize:trainSize+testSize]).astype(np.float32)
    y_test = np.array(y_imputate[trainSize:trainSize+testSize]).astype(np.float32)
    y_actual = np.array(y[trainSize+testSize:trainSize+testSize+predictSize]).astype(np.float32)
    x= np.array(x)
    y= np.array(y)
    '''
    
    x_train,y_train,x_test,y_test,y_actual,x_imputate=imputate.brnn_imputate(x,y,start,timeSequence,opt,cols_orig)
    #y_actual = np.array(y_orig[trainSize+testSize:trainSize+testSize+predictSize]).astype(np.float32)
    #y_actual = scaler_y.transform(y_actual)

    #print(len(x_total))
    #print(len(y_total))

   

    
    print(len(x_train),len(y_train),len(x_test),len(y_test),len(y_actual))
    #print(x_train,y_train,x_test,y_test,y_actual)
    
    #print(y_train)
    #x_actual = np.array(x[-start+trainSize+testSize-predictSize:-start+trainSize+testSize+predictSize]
    #x_train,x_test,y_train,y_test = train_test_split(x_,y_,test_size=0.2,shuffle=True)

    return  x,y,x_imputate,x_train,y_train,x_test,y_test,y_actual,start
def datamask(data):
    count = 0
    data = data.reset_index()
    for col in data.columns:
        y_mask = np.random.rand(len(data))  < 0.25
        print(y_mask)
        for i in range(len(data)):
            if y_mask[i] == True:
                data.loc[i,col] = -1
                count +=1
            i+=1
    data = data.drop(('index'),axis=1)
    print(data)
    return data