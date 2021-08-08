#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:10:20 2021

@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This script loads geometrical facial features (which have been pre-extracted
previously by the extract_features.py script) and trains a Partial Leasts 
Squares facial expression analysis model. The combination of these 2 scripts, 
is also included in the notebook file Extract_features_and_train_model.ipynb, 
in a more explanatory way. In order for this script to run successfuly, you
need first to extract and save the features, either by running 
extract_features.py or Extract_features_and_train_model.ipynb.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
from joblib import dump

plt.close('all')
plt.style.use('seaborn')



def get_mse(y_gt,y_predict):
    # Compute mean square error
    MSE = np.mean((y_gt - y_predict)**2, axis=0)
#    return np.sqrt(MSE).tolist()
    return MSE.tolist()

def get_ccc(y_gt,y_predict):
    # Compute canonical correlation coefficient
    CCC=[]
    for i in range(y_gt.shape[1]):
        A=y_gt[:,i]
        B=y_predict[:,i]
        pearson = pearsonr(A, B)
        c = ((2 * pearson[0] * A.std() * B.std()) / 
             ( A.std()**2 + B.std()**2 + (A.mean() - B.mean())**2 ))
        CCC.append(c)
    return CCC




# if FULL_FEATURES=False (exclude jawline) resulting dimensionality -> 1276
# if FULL_FEATURES=True (all 68 landmarks) resulting dimensionality -> 2278
FULL_FEATURES = False
COMPONENTS = 31
PATH_MODELS = '../models/'
PATH_DATA = '../data/'


# load datasets and frontalization weights
print('Data loading...')
# ATTENTION. You need to run extract_features.py in order to generate features
# before running this script!
features = np.load(f'{PATH_DATA}features_fullfeatures={FULL_FEATURES}.npy')
df_data = pd.read_csv(f'{PATH_DATA}Morphset.csv')



# split data
np.random.seed(1)
max_subjects = int(df_data['Subject'].values.max())
subjets = np.array(range(1,max_subjects))
np.random.shuffle(subjets)

# 70% train, 20% validation, 10% testing subjects
subjects_train, subjects_val, subjects_test = np.split(
    subjets, 
    [int(.7*len(subjets)), int(.9*len(subjets))]
    )

# subjects to indices
indx_train = list(df_data['Subject'].isin(subjects_train))
indx_val = list(df_data['Subject'].isin(subjects_val))
indx_test = list(df_data['Subject'].isin(subjects_test))

# split features
features_train = features[indx_train, :]
features_val = features[indx_val, :]
features_test = features[indx_test, :]

# split annotations
avi_train = df_data.iloc[indx_train,5:8].values.astype(np.float16)
avi_val = df_data.iloc[indx_val,5:8].values.astype(np.float16)
avi_test = df_data.iloc[indx_test,5:8].values.astype(np.float16)



#--------------------------------------------------------------------------PLS
print('PLS regression...')


pls = PLSRegression(n_components=COMPONENTS)
pls.fit(features_train, avi_train)

y_predict = pls.predict(features_val)
MSE_val = get_mse(avi_val, y_predict)
print ('Validation MSE=',MSE_val)
ccc_val = get_ccc(avi_val, y_predict)
print('Validation CCC=',ccc_val)

y_predict = pls.predict(features_test)
MSE_test = get_mse(avi_test, y_predict)
print ('Test MSE=',MSE_test)
ccc_test = get_ccc(avi_test, y_predict)
print('Test CCC=',ccc_test)

# save model
output = {}
output['model'] = pls
output['full_features'] = FULL_FEATURES
output['components'] = COMPONENTS
dump(output, f'{PATH_MODELS}model_emotion_pls={COMPONENTS}_fullfeatures={FULL_FEATURES}.joblib')
