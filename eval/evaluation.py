import pandas as pd
import numpy as np
from numpy import log10, log2, exp2
import os
import sklearn
from sklearn import tree
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import auc,f1_score,accuracy_score,mean_absolute_error,mean_squared_error
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,roc_curve, precision_recall_curve,auc, f1_score,silhouette_score,normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import train_test_split
import model
from model import brnn
from model.brnn import neuralNetwork

import keras
import math
import argparse
import datetime
import random
import time
import matplotlib.pyplot as plt
from framework import federated_learning_nn
import gc









def evaluation(actual,predict,opt):
    valid_indice = ~np.isnan(actual)
    actual = actual[valid_indice]
    predict = predict[valid_indice]
    f1score = f1_score((actual*100).astype('int32'),(predict*100).astype('int32'),average='micro')
    accuracy = accuracy_score((actual*100).astype('int32'),(predict*100).astype('int32'))
    mse = mean_squared_error(actual,predict)
    mae = mean_absolute_error(actual,predict)
    #fpr,tpr,thresholds=roc_curve(actual,predict)
    #roc_auc=auc(fpr,tpr)
    #precision,recall,thresholds=precision_recall_curve(actual,predict)
    #prc_auc=auc(recall,precision)
    #cm_predict=confusion_matrix(actual,predict)
    #auc_predict=roc_auc_score(actual,predict,average='micro')
    #pauc_predict=roc_auc_score(actual,predict,average='micro',max_fpr=0.1)
    #ari=adjusted_rand_score(actual,predict)
    #nmi=normalized_mutual_info_score(actual,predict)
    with open (opt.log_dir+'result.log','a') as f:

        f.write('----------------------------------------------------\n')
        #f.write('confusion matrix={}\n'.format(cm_predict))
        #f.write('auc={}\n'.format(auc_predict))
        #f.write('pauc={}\n'.format(pauc_predict))
        #f.write('roc_auc={}\n'.format(roc_auc))
        #f.write('prc_auc={}\n'.format(prc_auc))
        #f.write('ARI={}\n'.format(ari))
        #f.write('NMI={}\n'.format(nmi))
        f.write('F Measure={}\n'.format(f1score))
        f.write('Accuracy Score={}\n'.format(accuracy))
        f.write('mse={}\n'.format(mse))
        f.write('mae={}\n'.format(mae))
        f.close()
    return f1score,accuracy,mse,mae