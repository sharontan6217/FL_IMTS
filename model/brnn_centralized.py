#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  11 11:05:30 2019

@author: sharontan
"""

#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
import sklearn.metrics
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import utils
from utils.optimization import birnn_stability, compute_gradient,compute_critical_rounds
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,  GRU, Bidirectional
from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import L1L2
import utils
from utils.utils import fl_convertion
import numpy as np
import math
import os
import pandas as pd
import datetime
from pandas import read_csv,DataFrame
from numpy import log10
from datetime import timedelta
import random
import operator
import gc

from model import Config_ablation_gru

model_config = Config_ablation_gru.brnn_config()
config = Config_ablation_gru.fl_config()
# init data

#scaler = StandardScaler()
scaler = MinMaxScaler()

class neuralNetwork():

    def myBiRNN(x_train,
                y_train,
                x_test,
                y_test,
                gru_units=model_config.gru_units,
                dense_units=model_config.dense_units,
                input_shape=model_config.input_shape,
                drop_out=model_config.drop_out,
                patience=model_config.patience,
                batch_size=model_config.batch_size,
                epochs=None,
                NUM_CLIENTS =None):
        '''
        model = tf.keras.models.Sequential()
        reg = keras.regularizers.L1L2(l1=model_config.l1, l2=model_config.l2)
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units,dropout=drop_out,activation=model_config.activation,recurrent_activation=model_config.recurrent_activation,recurrent_regularizer=reg,
                                   return_sequences=True),
                                   input_shape=input_shape,
                                   merge_mode="concat"))
        
        model.add(tf.keras.layers.Dense(units=1))
        #model.add(Activation('softmax'))
        model.add(tf.keras.layers.GlobalAveragePooling1D())

        


        
        '''
        
        reg = L1L2(l1=model_config.l1, l2=model_config.l2)
        inputs = keras.Input(shape=input_shape, name = 'digits')
        
        x = Bidirectional(GRU(units=gru_units,dropout=drop_out,activation=model_config.activation,recurrent_activation=model_config.recurrent_activation,recurrent_regularizer=reg,
                                   return_sequences=True),
                                   input_shape=input_shape,
                                   merge_mode="concat")(inputs)
        x = Dense(units=1)(x)
        outputs = GlobalAveragePooling1D()(x)
        '''
        if x.shape[0]==None:
            print(x.shape[0])
            outputs = x
        else:
            print(x.shape[0])
            dimensions = int(x.shape[0]/NUM_CLIENTS)
            print(dimensions,NUM_CLIENTS)
            outputs = Reshape((dimensions,NUM_CLIENTS))(x)
        '''
        model = Model(inputs, outputs)
        
        print (model.summary())
        early_stopping=tf.keras.callbacks.EarlyStopping(monitor="var_loss", patience=patience)
        
        lr_schema= tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config.learning_rate,decay_steps=config.decay_steps, decay_rate=config.decay)
        optimizer = keras.optimizers.Adamax(learning_rate=lr_schema)
        model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mae'])

        
        #history_callback=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2, validation_data=[x_test,y_test],shuffle = True)
        history_callback=model.fit(x_train,y_train,batch_size,epochs,verbose=2, validation_data=[x_test,y_test], shuffle = True)

        return model,history_callback
    def create_tf_dataset_for_client(x, y):
        global NUM_CLIENTS
        NUM_CLIENTS = x.shape[1]
        client_data = []
        data_per_client = len(x) 
        #print(x)
        #print(y)
        df_x = pd.DataFrame(data=x)
        df_y = pd.DataFrame(data=y)
        x_=[]
        y_=[]
        for col in df_x.columns:
            x_.append(df_x[col])
        for col in df_y.columns:
            y_.append(df_y[col])
        x_ = np.asarray(x_).reshape(-1,1,1)
        y_ = np.asarray(y_).reshape(-1,1)

        return x_,y_
    def train(x_train,y_train,x_test,y_test,x_sample,y_actual):

        gc.collect()
        #data load
        print("#data load:")
        print(np.count_nonzero(x_train),np.count_nonzero(y_train),np.count_nonzero(x_test),np.count_nonzero(y_test))
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        y_train=scaler.fit_transform(y_train)
        y_test=scaler.transform(y_test)
        x_sample=scaler.transform(x_sample)
        
        x_train_,y_train_ = neuralNetwork.create_tf_dataset_for_client(x_train,y_train)
        x_test_,y_test_ = neuralNetwork.create_tf_dataset_for_client(x_test,y_test)
        dataset = tf.convert_to_tensor(x_train)

        print(x_train_)
        print(x_train_.shape,y_train_.shape)
        print(x_test_)
        print(x_test_.shape,y_test_.shape)
        #train
        if config.check_communication_rounds == True:
            critical_communication_rounds = compute_critical_rounds.compute_critical_rounds(dataset,NUM_CLIENTS)
            print('optimized communication rounds of {} are required.'.format(critical_communication_rounds))
            with open ('optimized_parameters.log','a') as f:
                f.write('optimized communication rounds of {} are required.\n'.format(str(critical_communication_rounds)))
                f.close()   
        NUM_ROUNDS = critical_communication_rounds

        dimensions_train = x_train.shape[0]

        model,history_callback=neuralNetwork.myBiRNN(x_train_,y_train_,x_test_,y_test_,epochs=NUM_ROUNDS,NUM_CLIENTS=NUM_CLIENTS)

        results = model.evaluate(x_test_,y_test_)
        print(results)
        trainPredict=model.predict(x_train_)
        print(trainPredict.shape)
        trainScore=math.sqrt(mean_squared_error(y_train_,trainPredict)) 
        print('Train Score: %.5f RMSE' % (trainScore))
        
        testPredict=model.predict(x_test_)
        testScore=math.sqrt(mean_squared_error(y_test_,testPredict)) 
        print('Test Score: %.5f RMSE' % (testScore)) 
        
        trainSize = len(x_train)*NUM_CLIENTS
        SampleSize = len(x_sample)*NUM_CLIENTS
        TestSize = len(x_test)*NUM_CLIENTS
        PredictSize = SampleSize-TestSize
        x_sample_ = fl_convertion(x_sample).astype(np.float32)    
        sample=pd.DataFrame(data=x_sample)
        sample.to_csv('sample.csv')
        x_actual = x_sample_ [:TestSize]

        df_actual=pd.DataFrame(data=x_actual)
        df_actual.to_csv('actual.csv')     
        
        
        print(trainSize,SampleSize,TestSize,PredictSize,len(x_actual))

        
        for j in range(PredictSize):
            if len(x_actual) == TestSize:
                y_predict = testPredict
                print(y_predict)
                #model,history_callback=neuralNetwork.myBiRNN(x_train=x_train_,y_train=y_train_,x_test=x_test_,y_test=y_test_,NUM_CLIENTS=NUM_CLIENTS)
            else:
                #model,history_callback=neuralNetwork.myBiRNN(x_train=x_train_,y_train=y_train_,x_test=y_actual1,y_test=y_predict,NUM_CLIENTS=NUM_CLIENTS)
                x_actual = x_sample_[:TestSize+j]
                y_predict = model.predict(x_actual1)
            x_new=[y_predict[-1]]
            
            x_actual=np.append(x_actual,x_new,axis=0)
            x_actual1=x_actual[j:TestSize+j]
            x_actual1=np.reshape(x_actual1,(len(x_actual1),1,1))
            #x_actual1=scaler.transform(x_actual)
            #print(len(x_actual1))
            

            y_predict=model.predict(x_actual1)
            j=j+1
        #y_actual = np.where(np.isnan(y_actual),-1,y_actual)
        y_predict = y_predict[-PredictSize:]
        print(y_actual,y_predict)
        print(len(y_actual))
        print(len(y_predict))
        print(y_actual.shape,y_predict.shape)
        print(NUM_CLIENTS)
        dimensions = len(y_actual)
        print(dimensions)
        y_actual=scaler.transform(y_actual)
        #y_actual = np.reshape(y_actual,(dimensions,NUM_CLIENTS))
        y_predict = np.reshape(y_predict,(dimensions,NUM_CLIENTS))
        
        return y_predict,y_actual



    
