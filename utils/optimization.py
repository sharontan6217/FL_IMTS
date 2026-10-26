
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import torch
#import dataset
#from dataset import dataLoad,dataSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler,normalize,KBinsDiscretizer
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error,mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, KernelDensity
from scipy.spatial.distance import sqeuclidean,jaccard,canberra,cdist,euclidean,jensenshannon
import model
from model import Config, brnn
from model.Config_ablation_gru import brnn_config,fl_config
from model.brnn import neuralNetwork
import math
import utils
from utils import utils
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
def AEC(dataset):
	scaler = MinMaxScaler()
	x = scaler.fit_transform(dataset)
	kde=KernelDensity(kernel='gaussian').fit(x)
	dens=kde.score_samples(x)          

	average_dens=np.mean(dens)
      
		
	if abs(average_dens)<2:
		min_samples_aec=2
	else:
		min_samples_aec=int(abs(average_dens))

	x_ = np.array(x)
	dist=[]
	for i in range (len(x)):

		dist_pair=cdist(np.array(np.reshape(x[i],(1,len(x[i])))),x_,'euclidean')
		average_dist_= np.mean(dist_pair)

		dist.append(average_dist_)

	average_dist=max(dist)
	eps_aec=average_dist/(2*min_samples_aec)

	print("eps is: ", eps_aec)
		
	

	return eps_aec
def critical_sample_size(data, risk_thread):
	kde=KernelDensity(kernel='gaussian').fit(data)
	#kde=KernelDensity(kernel='linear').fit(data)
	dens=kde.score_samples(data)    
	distance=1/max(abs(dens))
	print('average density is: ', distance)    
	features = data.columns
	print(len(features))
	w = len(features)
	sampleSize = w**2/(risk_thread-(distance/2)**2)
	print('sample size is: ',sampleSize)

	return sampleSize

class sample_complexity_lower_bound():

	def estimate_mutual_information(x, y,n_bins):
		"""
		Estimate mutual information between X and Y using discretization.
		Args:
			x: array-like, shape (n_samples, feature_dim_x)
			y: array-like, shape (n_samples, feature_dim_y)
			n_bins: number of bins for discretization
		Returns:
			MI estimate
		"""
		# Discretize x
		discretizer_x = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
		x_binned = discretizer_x.fit_transform(x)
		
		# Discretize y
		discretizer_y = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
		y_binned = discretizer_y.fit_transform(y)
		
		# Flatten for mutual_info_score
		x_flat = x_binned.astype(int).astype(str).astype(str).flatten()
		y_flat = y_binned.astype(int).astype(str).astype(str).flatten()
		
		# Compute mutual information
		MI = mutual_info_score(x_flat, y_flat)
		return MI

	def compute_sample_lower_bound( x,y, dataset,n_bins):
		"""
		Compute the minimal number of samples needed based on MI.
		Args:
			MI: mutual information estimate
			epsilon: desired confidence/error margin
			c: constant (default 1.0)
		Returns:
			n_samples: lower bound on sample size
		"""

		scaler = MinMaxScaler()
		dataset_norm = scaler.fit_transform(dataset)

		mse_target = 0.008
		eps = mse_target/np.std(dataset_norm) # confidence/error margin
		#eps = AEC(dataset)
		print("eps is: ", eps)
		
		# Estimate mutual information
		MI_estimate = sample_complexity_lower_bound.estimate_mutual_information(x, y,n_bins)
		print(f"Estimated mutual information I(X;Y): {MI_estimate:.4f}")
		c=1.0
		# Compute sample complexity lower bound
		min_samples =  c * MI_estimate / (eps ** 2)
		print(f"Minimum samples needed for reliable superiority: {min_samples}")
		return int(np.ceil(min_samples))




class birnn_stability():
	def compute_jacobian(x,weights):
		'''
			
		batches = int(len(dataset)/config.batch_size)
		last_batch_size = len(dataset)-batches*config.batch_size
		if last_batch_size>0:
			batches=batches+1
			jacobians = []
		for i in range(batches-2):
			print(i)
			start_index = i*config.batch_size
			end_index = min((i+1)*config.batch_size,len(dataset)-i*config.batch_size)
			inputs = dataset[start_index:end_index]
		'''
		test_model = neuralNetwork.myBiRNN(gru_units=model_config.gru_units,drop_out=model_config.drop_out,input_shape=model_config.input_shape)
		#tff.learning.models.assign_weights_to_keras_model(test_model,weights)
		weights.assign_weights_to(test_model)

		with tf.GradientTape() as tape:
			tape.watch(x)
			outputs= test_model(x)
			#loss = tf.keras.losses.MSE(y,outputs)
		#print(loss)
		jacobians = tape.jacobian(outputs,x)
		#trainable = tf.convert_to_tensor(trainable)

		print("Jacobian shape:", jacobians.shape)
		return jacobians
	def spectral_radius(matrix):
		matrix_ = matrix.numpy()
		elgenvalues = np.linalg.eigvals(matrix_)

		return max(abs(elgenvalues))
	def stability(x,y,updated_weights,spectral_threshold=1.00):
		gc.collect()
		jacobians,gradients = birnn_stability.compute_jacobian(x,updated_weights)
		jacobians_matrix = tf.reshape(jacobians,[jacobians.shape[0],jacobians.shape[2]])
		radius = birnn_stability.spectral_radius(jacobians_matrix)
		print(f"Spectral radius of the Jacobian: {radius}")

		if radius<spectral_threshold:
			print("Model is stable under the spectral radius threshold")
		else:
			("Model exceeds the spectral radius threshold; consider regularization")
		return radius 
def lipschitz_gap_map_decay(decay_rate,delta_time):
	return decay_rate*np.exp(-decay_rate*delta_time)
def lipschitz_gated_update(decay_rate,data,delta):
	return data*np.exp(-decay_rate*delta)

def compute_gradient(x,y,weights):
	gc.collect()

	gradient_model = neuralNetwork.myBiRNN(gru_units=model_config.gru_units,drop_out=model_config.drop_out,input_shape=model_config.input_shape)
	#tff.learning.models.assign_weights_to_keras_model(test_model,weights)
	weights.assign_weights_to(gradient_model)


	y = tf.reshape(y,[y.shape[0],1])
	#trainable = tf.convert_to_tensor(trainable)
	with tf.GradientTape() as tape:
		tape.watch(x)
		outputs= gradient_model(x)
		#loss = tf.keras.losses.MSE(y,outputs)
	gradients = tape.gradient(outputs,x)
	print("Gradients shape:", gradients.shape)
	return gradients
class critical_intensity_estimate():

	def data_mask(data,alpha ):
		data = data.reset_index()

		'''
		y_mask = np.random.rand(len(data))  < 0.05
		for i in range(len(data)):
			if y_mask[i] == True:
				data.loc[i:i+5,:] =  np.nan
			i+=1
		'''
		
		count=0
		mask=data
		estimate = data
		print('------------------------------the length of data is: {}----------------------------'.format(len(data)))
		
		for col in data.columns:
			N = np.random.poisson(alpha * len(data))
			alpha_N = N/len(data)
			print(alpha_N)
			y_mask = np.random.rand(len(data))  < alpha_N
			#print(y_mask)
			for i in range(len(mask)):
				if y_mask[i] == True:

					mask.loc[i,col] = np.nan
					'''
					if col == 'SNOW':
						data.loc[i,'SNWD'] =  np.nan
					elif col == 'SNWD':
						data.loc[i,'SNOW'] =  np.nan
					'''
					mask.loc[i,col] = 0
					count +=1

				i+=1
		
		mask = mask.drop(('index'),axis=1)
		#print('event is', count)
		#print(mask)
		return mask, estimate,count
	
	def critial_threshold_estimate(data,opt,timeSequence):
		gc.collect()
		print(data.shape)
		

		variances = []
		event_times=[]
		num_simulations = 20  # number of simulations for averaging
		alphas = np.linspace(0.01, 1, 50) 
		print('alpha is: ',alphas)
		for alpha in alphas:
			var_estimate = []
			event_time =[]
			for _ in range(num_simulations):
				marks, estimate,event = critical_intensity_estimate.data_mask(data, alpha)
				var_estimate_ = np.var(estimate)
				event_time.append(event)
				var_estimate.append(var_estimate_)
				#print('variance estimated is: {}'.format(var_estimate))
			event_times.append(np.mean(event_time))
			
			variances.append(np.mean(var_estimate))
		print(event_times)
		print(alpha)
		print(variances)
		graph_dir = opt.graph_dir
		# Plotting the results
		fig,ax1=plt.subplots()
		color = 'tab:red'
		ax1.set_ylabel('Estimated Variance',color=color)
		ax1.plot(alphas, variances,color=color)
		ax1.tick_params(axis='y',color=color)
		ax2 = ax1.twinx()
		color = 'tab:blue'
		ax2.set_ylabel('Event Occurences',color=color)
		ax2.plot( alphas, event_times,color=color)
		ax2.tick_params(axis='y',color=color)
		fig.tight_layout()
		plt.grid(True)
		plt.title('Impact of Mask Process Intensity on Variance')
		plt.legend()
		#plt.show()
		plt.savefig(graph_dir+'Impact of Mask Process Intensity on Variance_'+timeSequence+'.png')

		# Critical regime: as lambda -> 0, variance diverges
		critical_threshold = alphas[np.argmax(variances)]
		print(f"Approximate critical intensity where variance begins to diverge: {critical_threshold:.2f}")
		return critical_threshold

class compute_critical_rounds():

	def compute_gradient_model(dataset):
		gc.collect()
		
			
		batches = int(len(dataset)/config.batch_size)
		last_batch_size = len(dataset)-batches*config.batch_size
		if last_batch_size>0:
			batches=batches+1
			jacobians = []
		for i in range(batches-1):
			start_index = i*config.batch_size
			end_index = min((i+1)*config.batch_size,len(dataset)-i*config.batch_size)
			inputs = dataset[start_index:end_index]
		
		gradient_model = neuralNetwork.myBiRNN(gru_units=model_config.gru_units,drop_out=model_config.drop_out,input_shape=model_config.input_shape)
		#tff.learning.models.assign_weights_to_keras_model(test_model,weights)
		#weights.assign_weights_to(gradient_model)


		#y = tf.reshape(y,[y.shape[0],1])
		#trainable = tf.convert_to_tensor(trainable)
		with tf.GradientTape() as tape:
			tape.watch(dataset)
			outputs= gradient_model(dataset)
			#loss = tf.keras.losses.MSE(y,outputs)
		gradients = tape.gradient(outputs,dataset)
		print("Gradients shape:", gradients.shape)
		return gradients,batches
	def calculate_heterogeneity(dataset):
		total_divergence = 0
		count = 0
		for i in range(len(dataset)):
			for j in range(i+1,len(dataset)):
				divergence = jensenshannon(dataset[i],dataset[j])
				
				if math.isnan(divergence)==False:
					total_divergence +=divergence
					count +=1
		gamma = total_divergence/count 
		return gamma
	def loss_parameters(basic_model,gradients):
		parameters =basic_model.trainable_variables
		print(len(parameters))

		L_estimates=[]
		mu_estimates=[]
		for i in range(1,len(parameters)):
			#print(type(parameters[i][0]))
			#print(type(parameters[i-1][0]))
			delta_parameters = parameters[i][0]-parameters[i-1][0]
			delta_parameters=tf.reshape(delta_parameters,[1,-1])
			delta_grads=gradients[i]-gradients[i-1]

			delta_grads= tf.reshape(delta_grads,[-1,1])
			norm_delta_params = np.linalg.norm(delta_parameters)
			if norm_delta_params==0:
				continue
			ratio = np.linalg.norm(delta_grads)/norm_delta_params
			L_estimates.append(ratio)
			parameters_norm = np.linalg.norm(delta_parameters)**2

			#parameters_norm= tf.reshape(parameters_norm,[-1,1])
			mu_value = np.dot(delta_grads,delta_parameters)/parameters_norm
			mu_estimates.append(mu_value)

		return L_estimates, mu_estimates
	def compute_critical_rounds(dataset, K):
		gc.collect()
		'''
		Parameters:
		eps: Desired deviation/accuracy target
		E: Number of local updates performed by each client per round
		gamma: Degree of non-IID data heterogeneity (Gamma>0)
		sigma_sq: Maximum variance of client stochastc gradients
		gamma_sq: Bound on the squared norm of the gradients
		K: Number of participating clients per round
		L: Smothness parameter of the loss function
		mu: Strong convexity parameter of the loss function
		
		Returns:
		R: Minium communication rounds required
		'''
		#eps = compute_critical_rounds.AEC(dataset)
		basic_model = neuralNetwork.myBiRNN(gru_units=model_config.gru_units,drop_out=model_config.drop_out,input_shape=model_config.input_shape)
		gradients,E = compute_critical_rounds.compute_gradient_model(dataset)

		L_estimates, mu_estimates = compute_critical_rounds.loss_parameters(basic_model,gradients)
		
		
		eps = 0.008/np.std(dataset) # confidence/error margin
		#eps = AEC(dataset)
		print('eps is: ',eps)
		print(np.std(dataset))
		#eps = 1e-4
		
		
		gamma = compute_critical_rounds.calculate_heterogeneity(dataset)

		sigma_sq = np.max(np.var(gradients))

		gamma_sqs=[]

		for gradient_ in gradients:
			#print(gradient_)
			gradients_norm = np.sqrt(np.linalg.norm(gradient_))
			gamma_sqs.append(gradients_norm)

		gamma_sq=np.max(np.asarray(gamma_sqs))
		print('gamma_sq is: ',gamma_sq)
	
		L=0
		mu=0
		if len(L_estimates)>0:
			L = np.max(L_estimates)
		print(L)
		print(max(L_estimates))
		if len(mu_estimates)>0:
			min_mus=[]
			for mu_estimate in mu_estimates:
				if mu_estimate.shape[1]>0:
					for mu_ in mu_estimate:
						min_mu = min(mu_)
						min_mus.append(min_mu)
			mu = min(min_mus)
		#calculate the condition number aspect of the problem
		kappa = L/mu
		#constant_multiplier = kappa/mu
		constant_multiplier = 1
		print('constant multiplier is: ',constant_multiplier)
		variance_term = sigma_sq/K
		heterogeneity_term = gamma
		drift_penalty = (gamma_sq/E)+(E*gamma_sq)
		T=(constant_multiplier/eps)*(variance_term+heterogeneity_term+drift_penalty)
		R=math.ceil(T)
		return max(1,R)

