#!/usr/bin/env python
# coding: utf-8
#
# t1_gm_parc_Depot_WR_3Blks_AvgPool_0300PM.py

import sys, os
os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
import nibabel as nib
import numpy as np
import pandas as pd
import keras
from keras import regularizers
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Activation, Conv3D, AveragePooling3D, Flatten, Dense,BatchNormalization, SpatialDropout3D
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

# updated Mar 24, 2019 04:00 AM
from keras.layers import concatenate # to add covariates
from keras.engine.topology import Input # to add covariates

## Hyper-parameters
n_epoch = np.int8(sys.argv[3])


## Load Class DataGenerator for the specific ROI
getClass='DataGenerator_ROI_WR'
getClass_string='from ' + getClass + ' import DataGenerator'
exec(getClass_string)

## Initialize dimension parameters for the specific ROI
ROI_info = pd.read_csv(os.getcwd() + '/ROICropThreshSummary_processed.csv')
ROI_dim = ROI_info[ROI_info['ROI']==sys.argv[2]]['Max.distance'].values.repeat(3)

## Load Data
# data_dir = "//depot/yuzhu/data/ABCD/TrainValid/4_TrainValid"

train_labels = pd.read_csv('training_fluid_intelligenceV1.csv', index_col=0)
train_labels = train_labels.rename(index=str, columns={'residual_fluid_intelligence_score': 'score'})
train_labels.index = train_labels.index.str.replace('_','')

valid_labels = pd.read_csv('validation_fluid_intelligenceV1.csv', index_col=0)
valid_labels = valid_labels.rename(index=str, columns={'residual_fluid_intelligence_score': 'score'})
valid_labels.index = valid_labels.index.str.replace('_','')

labels_concat = pd.concat([train_labels, valid_labels])

partition = {'train': list(train_labels.index),'validation': list(valid_labels.index)}
labels = dict(zip(list(labels_concat.index), list(labels_concat.score)))

## Parameters
params = {'dim': ROI_dim,
          'batch_size': 32,
          'n_channels': 1,
          'shuffle': True,
          'ROI_ID': sys.argv[2]}

## Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

################# DESIGN MODEL #################
tf.keras.backend.clear_session()
model = Sequential()

# Repeated Block 1
model.add(Conv3D(8, (3, 3, 3), strides=1, kernel_regularizer=keras.regularizers.l2(0.0001), input_shape=np.concatenate((ROI_dim, [1])), data_format="channels_last"))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))
model.add(SpatialDropout3D(rate=0.4))

# Repeated Block 2
model.add(Conv3D(16, (3, 3, 3), strides=1, kernel_regularizer=keras.regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))
model.add(SpatialDropout3D(rate=0.4))

# Repeated Block 3
model.add(Conv3D(32, (3, 3, 3), strides=1, kernel_regularizer=keras.regularizers.l2(0.0001)))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))
model.add(SpatialDropout3D(rate=0.3)) # 0.4

# Repeated Block 4
# model.add(Conv3D(64, (3, 3, 3), strides=1, kernel_regularizer=keras.regularizers.l2(0.0001)))
# model.add(BatchNormalization())
# model.add(Activation('elu'))
# model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))
# model.add(SpatialDropout3D(rate=0.4))

# Repeated Block 5
# model.add(Conv3D(128, (3, 3, 3), strides=1, kernel_regularizer=keras.regularizers.l2(0.0001)))
# model.add(BatchNormalization())
# model.add(Activation('elu'))
# model.add(AveragePooling3D(pool_size=(2, 2, 2), strides=2))
# model.add(SpatialDropout3D(rate=0.4))
####

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(1))

# 2019.2.15 Change learning rate = 0.001
# sgd = SGD(lr=0.01,decay=5e-5,momentum=0.9,nesterov=True)
# sgd = SGD(lr=0.0001,decay=5e-5,momentum=0.9,nesterov=True)
# sgd = SGD(lr=0.1,decay=5e-5,momentum=0.9,nesterov=True)
adam = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=5e-5, amsgrad=False)

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

model.compile(loss='mse', # 'mean_absolute_error',
              optimizer='adam',)
              #metrics=['mse', 'acc'])
              #               metrics=[soft_acc])

model.summary()
###################################################

## Train model on dataset
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

history = model.fit_generator(generator=training_generator,
                              validation_data=validation_generator,
                              epochs=n_epoch, #25
                              max_queue_size=32,
                              use_multiprocessing=True,
                              workers=100,
                              callbacks=[early_stop])

## Save the model
saveModel='models/20190324_t1_gm_parc_ROI{}'.format(sys.argv[2]) + '_epoch' + sys.argv[3] + '.h5'
saveModel_string='model.save("' + saveModel + '")'
exec(saveModel_string)
