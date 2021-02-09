import tensorflow as tf
from sklearn.utils import shuffle
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras.backend as K
from keras.models import Sequential

import keras.layers as L
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Masking
from keras.layers import Lambda
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization

from keras.optimizers import RMSprop,Adam
from keras import callbacks

import wtte.weibull as weibull
import wtte.wtte as wtte
from wtte.wtte import WeightWatcher

reduce_lr = callbacks.ReduceLROnPlateau(monitor='loss', 
                                        factor  =0.5, 
                                        patience=50, 
                                        verbose=0, 
                                        mode='auto', 
                                        epsilon=0.0001, 
                                        cooldown=0, 
                                        min_lr=1e-8)

nanterminator = callbacks.TerminateOnNaN()
history = callbacks.History()
weightwatcher = WeightWatcher(per_batch =False,per_epoch= True)
n_features = 2

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


path_x_tr = './x_train_m/'
path_y_tr = './y_train_m/'

x_tr_f = os.listdir(path_x_tr)
y_tr_f = os.listdir(path_y_tr)

init_alpha = 47.05906039370126
mask_value = -1.3371337


def nanmask_to_keras_mask(x,y,mask_value,tte_mask):
    """nanmask to keras mask.
        :param float mask_value: Use some improbable telltale value 
                                (but not nan-causing)
        :param float tte_mask: something that wont NaN the loss-function
    """
    # Use some improbable telltale value (but not nan-causing)
    x[:,:,:][np.isnan(x)] = mask_value
    y[:,:,0][np.isnan(y[:,:,0])] = tte_mask
    y[:,:,1][np.isnan(y[:,:,1])] = 0.95
    sample_weights = (x[:,:,0]!=mask_value)*1.
    return x,y,sample_weights
        
def tf_data_generator3(filelist, directory = [], batch_size = 5):
    i = 0
    x_t = os.listdir(directory[0])
    y_t = os.listdir(directory[1])
    while True:
        print(i)
        file_chunk = filelist[i*batch_size:(i+1)*batch_size] 
        X_a = []
        Y_a = []
        for fname in file_chunk:
            x_info = np.load(str(path_x_tr)+str(fname))
            y_info = np.load(str(path_y_tr)+str(fname))
            X_a.append(x_info)
            Y_a.append(y_info)
        X_a = np.concatenate(X_a)
        Y_a = np.concatenate(Y_a)
        tte_mean_train = np.nanmean(Y_a[:,:,0])
        mask_value = -1.3371337
        X_a,Y_a,W_a = nanmask_to_keras_mask(X_a,Y_a,mask_value,tte_mean_train)
        listed_no_mask = []
        listed_no_mask_Y = []
        listed_no_mask_w = []
        for j in range(X_a.shape[0]):
            if (X_a[j]==mask_value).all() == True:
                pass
            else:
                listed_no_mask.append(X_a[j])
                listed_no_mask_Y.append(Y_a[j])
                listed_no_mask_w.append(W_a[j])
        listed_no_mask = np.concatenate(listed_no_mask)
        listed_no_mask_Y = np.concatenate(listed_no_mask_Y)     
        listed_no_mask_w = np.concatenate(listed_no_mask_w) 
        shape1 = np.int(listed_no_mask.shape[0]/99)
        listed_no_mask = listed_no_mask.reshape(shape1,99,2)
        listed_no_mask_Y = listed_no_mask_Y.reshape(shape1,99,2)
        listed_no_mask_w = listed_no_mask_w.reshape(shape1,99)
        listed_no_mask = listed_no_mask
        listed_no_mask_Y = listed_no_mask_Y
        yield listed_no_mask, listed_no_mask_Y
        i = i + 1
        
def base_model():
    model = Sequential()
    model.add(Masking(mask_value=mask_value,input_shape=(None, 2)))
    model.add(GRU(3,activation='tanh',return_sequences=True))
    return model
    
def wtte_rnn():
    model = base_model()
    model.add(TimeDistributed(Dense(2)))
    model.add(Lambda(wtte.output_lambda, 
                     arguments={"init_alpha":init_alpha, 
                                "max_beta_value":4.0,
                                "alpha_kernel_scalefactor":0.5}))

    loss = wtte.loss(kind='discrete',reduce_loss=False).loss_function
    model.compile(loss=loss, optimizer=Adam(lr=.01,clipvalue=0.5),sample_weight_mode='temporal')
    return model        
        
generated_train_data = tf_data_generator3(x_tr_f, ['./x_train_m/', './y_train_m/'], batch_size = 2)

model = wtte_rnn()
model.summary()

K.set_value(model.optimizer.lr, 0.01)
model.fit(generated_train_data,
          epochs=10,
          verbose=1)
          
          
# after restarting the kernel and instead of L136-L144 having the following
# model = wtte_rnn()
# model.summary()

# a,b=next(generated_train_data)
# K.set_value(model.optimizer.lr, 0.01)
# model.fit(a,b,
#           epochs=10,
#           verbose=1)
#
# the model would run properly
