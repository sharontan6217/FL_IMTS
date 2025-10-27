import os
import torch
import time
import numpy as np
import keras
from keras.regularizers import L1L2
import ml_collections

def brnn_config():
    config = ml_collections.ConfigDict()
    config.gru_units=200
    config.dense_units = 10
    config.tol = 1e-5
    config.drop_out=0.1
    config.reg= L1L2(l1=0.2, l2=0.2)
    config.activation='tanh'
    config.recurrent_activation='relu'
    config.patience=5
    config.batch_size=128
    config.input_shape=(None,1)
    config.poolSize = 7000
    config.trainSize = 5000
    config.testSize = 1000
    config.predictSize = 100
    return config
