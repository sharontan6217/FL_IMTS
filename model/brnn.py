#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  11 11:05:30 2019

@author: sharontan
"""

#!/usr/bin/env python
# coding: utf-8

import sklearn.metrics
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
from keras.models import Sequential
from keras.layers import Dense,  GRU, Bidirectional, Activation
from keras.layers import GlobalAveragePooling1D, BatchNormalization, TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import L1L2

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
import model
from model import Config

model_config = Config.brnn_config()

# init data

scaler = StandardScaler()


class neuralNetwork():
    def myBiRNN(gru_units=model_config.gru_units,
                dense_units=model_config.dense_units,
                input_shape=model_config.input_shape,
                drop_out=model_config.drop_out,
                patience=model_config.patience):

        model = Sequential()
        reg = L1L2(l1=model_config.l1, l2=model_config.l2)
        model.add(Bidirectional(GRU(units=gru_units,dropout=drop_out,activation=model_config.activation,recurrent_activation=model_config.recurrent_activation,recurrent_regularizer=reg,
                                   return_sequences=True),
                                   input_shape=input_shape,
                                   merge_mode="concat"))
        '''
        model.add(BatchNormalization())
        model.add(TimeDistributed(Dense(dense_units,activation=model_config.recurrent_activation)))
        model.add(BatchNormalization())

        model.add(Bidirectional(GRU(units=gru_units,dropout=drop_out,activation=model_config.activation,recurrent_activation=model_config.recurrent_activation,recurrent_regularizer=reg,
                                   return_sequences=True),
                                   input_shape=input_shape,
                                   merge_mode="concat")) 

        model.add(BatchNormalization())
        '''
        model.add(Dense(units=1))
        #model.add(Activation('softmax'))
        model.add(GlobalAveragePooling1D())
        print (model.summary())
        


        
        return model