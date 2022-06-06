
#!/usr/bin/env python
# coding: utf-8

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
from keras.layers import Activation, Conv3D, MaxPooling3D, AveragePooling3D, Flatten, Dense,BatchNormalization, SpatialDropout3D
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split

## Load model
getModel='//depot/yuzhu/data/ABCD/Purdue150/Ikbeom/models/20190324_t1_gm_parc_ROI{}'.format(sys.argv[2]) + '_epoch' + sys.argv[3] + '.h5'
getModel_string='model1 = load_model("' + getModel + '")'
exec(getModel_string)

## Load Class DataGenerator for the specific ROI
getClass='DataGenerator_ROI_WRtest'
getClass_string='from ' + getClass + ' import DataGenerator'
exec(getClass_string)

## Initialize dimension parameters for the specific ROI
ROI_info = pd.read_csv(os.getcwd() + '/ROICropThreshSummary_processed.csv')
ROI_dim = ROI_info[ROI_info['ROI']==str(sys.argv[2])]['Max.distance'].values.repeat(3)

## Load Data
data_dir = "//depot/yuzhu/data/ABCD/TrainValid/4_TrainValid"

train_labels = pd.read_csv('training_fluid_intelligenceV1.csv', index_col=0)
train_labels = train_labels.rename(index=str, columns={'residual_fluid_intelligence_score': 'score'})
train_labels.index = train_labels.index.str.replace('_','')

valid_labels = pd.read_csv('validation_fluid_intelligenceV1.csv', index_col=0)
valid_labels = valid_labels.rename(index=str, columns={'residual_fluid_intelligence_score': 'score'})
valid_labels.index = valid_labels.index.str.replace('_','')

test_labels = pd.read_csv('abcdnp_testing_template_part1.csv', index_col=0)
test_labels = test_labels.rename(index=str, columns={'predicted_score': 'score'})
test_labels.index = test_labels.index.str.replace('_','')

labels_concat = pd.concat([train_labels, valid_labels, test_labels])

partition = {'train': list(train_labels.index),'validation': list(valid_labels.index), 'test': list(test_labels.index)}
labels = dict(zip(list(labels_concat.index), list(labels_concat.score)))

## Parameters
params = {'dim': ROI_dim,
          'batch_size': 1,
          'n_channels': 1,
          'shuffle': False,
          'ROI_ID': sys.argv[2]}
## Generators
validation_generator = DataGenerator(partition['test'], labels, **params)

predicted_score = model1.predict_generator(generator=validation_generator,
					   max_queue_size=5,
					   use_multiprocessing=True,
					   workers=1,
					   verbose=1)

subject = test_labels.index.str.replace('NDAR', 'NDAR_')
out = {'subject': list(subject),'predicted_score': list(predicted_score)}
out2 = pd.DataFrame(out)
out2['predicted_score'] = out2['predicted_score'].astype(np.float32)

## Save the output
saveOutput_string='out2.to_csv("pred_validation/20190324_pred_testset_ROI{}'.format(sys.argv[2]) + '_epoch' + sys.argv[3] + '_part1.csv")'
exec(saveOutput_string)
