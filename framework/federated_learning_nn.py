import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import torch
#import dataset
#from dataset import dataLoad,dataSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
import model
from model import Config, brnn
from model.Config_ablation_gru import brnn_config,fl_config
from model.brnn import neuralNetwork
import utils
from utils import utils
from utils.optimization import birnn_stability, compute_gradient,compute_critical_rounds,lipschitz_gap_map_decay
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import random
import gc
import sys, asyncio
device = torch.device("cpu")

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
        data = tf.data.Dataset.from_tensor_slices((x_[start_index:end_index], y_[start_index:end_index])).batch(config.batch_size)
        client_data.append(data)
    
    return client_data




def dataProcess_impute(x_train,y_train,x_test,y_test,timegaps_x_train,timegaps_y_train):
    global client_datasets,test_datasets, dataset, timegap_datasets
    dataset = tf.convert_to_tensor(x_train)
    client_datasets = create_tf_dataset_for_client(x_train, y_train)
    test_datasets =  create_tf_dataset_for_client(x_test, y_test)
    timegap_datasets = tf.data.Dataset.from_tensor_slices((timegaps_x_train, timegaps_y_train)).batch(config.batch_size)

    return client_datasets,test_datasets

def dataProcess_predict(x_train,y_train,x_test,y_test):
    global client_datasets,test_datasets, dataset
    dataset = tf.convert_to_tensor(x_train)
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

    
    if config.check_communication_rounds == True:
        critical_communication_rounds = compute_critical_rounds.compute_critical_rounds(dataset,NUM_CLIENTS)
        print('optimized communication rounds of {} are required.'.format(critical_communication_rounds))
        with open ('optimized_parameters.log','a') as f:
            f.write('optimized communication rounds of {} are required.\n'.format(str(critical_communication_rounds)))
            f.close()   
    NUM_ROUNDS = critical_communication_rounds
    lr_schema= tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config.learning_rate,decay_steps=config.decay_steps, decay_rate=config.decay)
    
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        #client_optimizer_fn=lambda: tf.keras.optimizers.Adamax(learning_rate=config.learning_rate,decay=config.decay))
        client_optimizer_fn=lambda: tf.keras.optimizers.Adamax(learning_rate=lr_schema))
        #client_weight_fn = model_fn().report_local_outputs)
    # 3. Iterative Training
    state = iterative_process.initialize()


    loss=[]
    mae=[]
    total_jacobians=[]
    L0=[]
    L1=[]
    early_stop = 0
    for round_num in range(NUM_ROUNDS):
        print(f'Round {round_num + 1}/{NUM_ROUNDS}')


        # Select clients for this round (e.g., all clients in a simulation)
        selected_client_data = [client_datasets[i] for i in range(NUM_CLIENTS)]
        state, metrics = iterative_process.next(state, selected_client_data)
        print(f'Round {round_num + 1} metrics: {metrics}')
        updated_weights = state.model
        if len(loss)>1:
            if (list(metrics.items())[2][1]['loss']>loss[-1] and list(metrics.items())[2][1]['mean_absolute_error']>mae[-1]) ==True:
                print(early_stop)
                early_stop+=1

        
        if early_stop<model_config.patience:
            loss.append(list(metrics.items())[2][1]['loss'])
            mae.append(list(metrics.items())[2][1]['mean_absolute_error'])
        else:
            print('No improvement for {} epchs, stop the training'.format(early_stop))
            #break
        if config.check_keras_model==True:
            radius_=[]
            for i in range(len(selected_client_data)):
                data_slice = selected_client_data[i]
                for x,y in data_slice:
                    radius = birnn_stability.stability(x,updated_weights)
                    radius_.append(radius)
            max_radius = max(radius_)
            with open ('optimized_parameters.log','a') as f:
                f.write('maximum radius is: {}\n'.format(str(max_radius)))
                f.close()          
        if config.check_gradients==True:
            gradients=[]
            max_gradients=[]
            min_gradients=[]
            for i in range(len(selected_client_data)):
                data_slice = selected_client_data[i]
                for x,y in data_slice:
                    gradient =  compute_gradient(x,y,updated_weights)
                    gradients.append(gradient)
                    max_gradient = max(gradient)
                    min_gradient = min(gradient)
                    max_gradients.append(max_gradient)
                    min_gradients.append(min_gradient)
            max_gradient_total = max(max_gradients)
            min_gradient_total = min(min_gradients)
            with open ('optimized_parameters.log','a') as f:
                f.write('maximum gradient is: {}, minimum gradient is: {}\n'.format(str(max_gradient_total),str(min_gradient_total)))
                f.close()              







        
    return state,metrics,loss,mae
def eval(test_datasets,state):
    # 5. Evaluation (optional)
    # You can build a federated evaluation computation using tff.learning.build_federated_evaluation
    # and evaluate the final model.

    selected_test_data = [test_datasets[i] for i in range(NUM_CLIENTS)]
    eval_process =  tff.learning.build_federated_evaluation(model_fn)
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
    fig1,ax1=plt.subplots()
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
    fig1.tight_layout()
    plt.grid()
    fig_name='fl_birnn_loss_'+timeSequence+'_'+str(start)+'.png'
    plt.savefig(brnn_graph_dir+fig_name)
    plt.close()

    if config.check_bound_of_maximum_timegap==True:
        max_timegaps,min_timegaps, time_length, bounds = check_valid_preset_decay(timegap_datasets)
        fig = utils.lower_bound_minimum_decay.visualize_t(max_timegaps,config.decay,time_length,timeSequence,start,brnn_graph_dir)




    return fig1

def check_valid_preset_decay(client_timegaps):
    bounds=[]
    max_timegaps=[]
    min_timegaps=[]
    for batch in client_timegaps:
        print(batch)
        batch_value = np.array(batch).reshape(-1,)
        max_timegap = np.max(batch_value)
        min_timegap = np.min(batch_value)

        L0 = lipschitz_gap_map_decay(config.decay,min_timegap)
   
        if L0<1  :
            print('The gated recurrence is contractive with the optimal decay rate of {}.'.format (L0))
        else:
            print('The gated recurrence is not contractive with the optimal decay rate of {}.'.format (L0))
            break
        time_length = len(batch)
        theoretical_lower_bound = utils.lower_bound_minimum_decay.lower_bound_minimum_decay(time_length, max_timegap, config.decay, gamma_change_per_step=0.01 )
        bounds.append(theoretical_lower_bound)
        max_timegaps.append(max_timegap)
        min_timegaps.append(min_timegap)
    return max_timegaps,min_timegaps, time_length, bounds