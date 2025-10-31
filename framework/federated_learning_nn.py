import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
#import dataset
#from dataset import dataLoad,dataSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import model
from model import Config, brnn
from model.Config import brnn_config,fl_config
from model.brnn import neuralNetwork
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import random
import gc
import sys, asyncio

if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

config = fl_config()

model_config = brnn_config()



gc.collect()

# Simulate client data (e.g., each client gets a subset)
'''
def create_tf_dataset_for_client(x, y, num_clients):
    
    client_data = []
    data_per_client = len(x) // num_clients
    print(x)
    print(len(x),len(y),data_per_client)
    for i in range(num_clients):
        start_index = i * data_per_client
        end_index = (i + 1) * data_per_client
        print(start_index,end_index)
        client_data.append(tf.data.Dataset.from_tensor_slices((x[start_index:end_index], y[start_index:end_index])).batch(config.batch_size))
    return client_data
'''
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
    y_ = np.asarray(y_).reshape(-1,1,1)
    #print(x_)
    #print(y_)
    #print(len(x),len(y),data_per_client)
    for i in range(x.shape[1]):        
        start_index = i * data_per_client
        end_index = (i + 1) * data_per_client
        client_data.append(tf.data.Dataset.from_tensor_slices((x_[start_index:end_index], y_[start_index:end_index])).batch(config.batch_size))
    return client_data





def dataProcess(x_train,y_train,x_test,y_test):
    global client_datasets,test_datasets
    client_datasets = create_tf_dataset_for_client(x_train, y_train)
    test_datasets =  create_tf_dataset_for_client(x_test, y_test)
    return client_datasets,test_datasets

# 2. TFF Components
def model_fn():
    keras_model = neuralNetwork.myBiRNN(gru_units=model_config.gru_units,drop_out=model_config.drop_out,input_shape=model_config.input_shape)
    return  tff.learning.from_keras_model(
        keras_model,
        input_spec=client_datasets[0].element_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    

def train(client_datasets):
    
    #model_weights,iterative_process = model_fn(model_predict,client_datasets)
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adamax(learning_rate=config.learning_rate))
    # 3. Iterative Training
    state = iterative_process.initialize()

    loss=[]
    mae = []
    for round_num in range(config.NUM_ROUNDS):
        print(f'Round {round_num + 1}/{config.NUM_ROUNDS}')
        
        # Select clients for this round (e.g., all clients in a simulation)
        selected_client_data = [client_datasets[i] for i in range(NUM_CLIENTS)]
        state, metrics = iterative_process.next(state, selected_client_data)
        loss.append(list(metrics.items())[2][1]['loss'])
        mae.append(list(metrics.items())[2][1]['mean_absolute_error'])
        
        print(f'Round {round_num + 1} metrics: {metrics}')
    return state,metrics,loss,mae
def eval(test_datasets,state,metrics):
    # 5. Evaluation (optional)
    # You can build a federated evaluation computation using tff.learning.build_federated_evaluation
    # and evaluate the final model.

    selected_test_data = [test_datasets[i] for i in range(NUM_CLIENTS)]
    eval_process = tff.learning.build_federated_evaluation(model_fn)
    test_metrics = eval_process(state.model, selected_test_data)
    print(f'metrics: {test_metrics}')
    model_predict = neuralNetwork.myBiRNN(gru_units=model_config.gru_units,drop_out=model_config.drop_out,input_shape=model_config.input_shape)
    model_predict.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    state.model.assign_weights_to(model_predict)
    return model_predict,test_metrics
def fl_visualize(loss,mae,timeSequence,start,brnn_graph_dir):
    # First, some preprocessing to smooth the training and testing arrays for display.
    window_length = 100
    train_mse = np.r_[
        loss[window_length - 1 : 0 : -1],
        loss,
        loss[-1:-window_length:-1],
    ][100:]
    train_mae = np.r_[
        mae[window_length - 1 : 0 : -1],
        mae,
        mae[-1:-window_length:-1],
    ][100:]
    #w = np.hamming(window_length)
    #train_y = np.convolve(w / w.sum(), train_s, mode="valid")
    #test_y = np.convolve(w / w.sum(), test_s, mode="valid")
    # Display the training accuracies.
    fig,ax1=plt.subplots()
    x = np.arange(0, len(train_mse), 1)
    color = 'tab:red'
    ax1.set_ylabel('MSE',color=color)
    ax1.plot( x, train_mse,color=color)
    ax1.tick_params(axis='y',color=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('MAE',color=color)
    ax2.plot( x, train_mae,color=color)
    ax2.tick_params(axis='y',color=color)
    fig.tight_layout()
    plt.grid()
    #plt.title('Plot Graph of Loss/Accuracy')

    fig_name='fl_birnn_loss_'+timeSequence+'_'+str(start)+'.png'
    plt.savefig(brnn_graph_dir+fig_name)
    plt.close()


    return fig
