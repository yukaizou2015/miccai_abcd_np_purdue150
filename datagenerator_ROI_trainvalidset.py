#!/usr/bin/env python
# coding: utf-8
#
# DataGenerator_ROI_WR.py
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

import os
import keras
import numpy as np
import nibabel as nib
import pandas as pd

# added on Mar 24, 2019 5:20 PM
from keras.layers import concatenate # to add covariates
from keras.engine.topology import Input # to add covariates

ROI_info = pd.read_csv(os.getcwd() + '/ROICropThreshSummary_processed.csv')

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=3, dim=(240,240,240), n_channels=1,
                 shuffle=True, ROI_ID=9999): # n_classes=None
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.roi_ID = str(ROI_ID)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

	# Include BTSV data (covariates)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#             X[i,] = nib.load(t1_gm_parc_dir + str(ID) + '_t1_brain_gm_parc_3.nii.gz').get_data()[1:131, 87:217, 69:199]
            index = i
            #example_img = '//depot/yuzhu/data/ABCD/TrainValid/4_TrainValid/' + str(ID).replace('_','')
            example_img = os.getcwd() + '/trainvaliddata/' + str(ID).replace('_','')
            files = os.listdir(example_img)
            
            index_mask = self.roi_ID
            xmin = int(ROI_info.loc[ROI_info['ROI']==index_mask]['X.min.adjusted'].values)
            xmax = int(ROI_info.loc[ROI_info['ROI']==index_mask]['X.max.adjusted'].values)
            ymin = int(ROI_info.loc[ROI_info['ROI']==index_mask]['Y.min.adjusted'].values)
            ymax = int(ROI_info.loc[ROI_info['ROI']==index_mask]['Y.max.adjusted'].values)
            zmin = int(ROI_info.loc[ROI_info['ROI']==index_mask]['Z.min.adjusted'].values)
            zmax = int(ROI_info.loc[ROI_info['ROI']==index_mask]['Z.max.adjusted'].values)
            
#             ni_mask = nib.load(example_img + '/' + files[0]).get_fdata()[xmin:xmax, ymin:ymax, zmin:zmax]
#             img = nib.load(example_img + '/' + files[1]).get_fdata()[xmin:xmax, ymin:ymax, zmin:zmax]
# 2019.3.20 flipped for Depot
            ni_mask = nib.load(example_img + '/' + files[1]).get_fdata()[xmin:xmax, ymin:ymax, zmin:zmax]
            img = nib.load(example_img + '/' + files[0]).get_fdata()[xmin:xmax, ymin:ymax, zmin:zmax]
            
            myMask = np.zeros_like(img,dtype = np.bool)
            myMask[ni_mask == np.int16(index_mask)] = True
            X[i,] = np.multiply(img,myMask)
            
            m = np.max(X[i,])
            mi = np.min(X[i,])
            X[i,] = (X[i,] - mi)/(m-mi)
            # Store class
            y[i] = self.labels[ID]
        # images = images.reshape(images.shape[0],190,190,190,1)
        # X = X.reshape(X.shape[0],240,240,240,self.n_channels)
        X = X.reshape(X.shape[0],*self.dim,self.n_channels)

        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)


