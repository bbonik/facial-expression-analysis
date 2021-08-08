#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 21:10:20 2021

@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This script extracts and stores geometrical facial features which will later 
be used for training a facial expression analysis model in (train_emotions.py).
The combination of these 2 scripts, is also included in the notebook file
Extract_features_and_train_model.ipynb, in a more explanatory way.
IMPORTANT: you need to unzip the Morphset.csv.zip dataset before you run this.
"""


import pandas as pd
import numpy as np
from emotions_dlib import GeometricFeaturesDlib, LandmarkFrontalizationDlib


# if FULL_FEATURES=False (exclude jawline) resulting dimensionality -> 1276
# if FULL_FEATURES=True (all 68 landmarks) resulting dimensionality -> 2278

FULL_FEATURES = False
PATH_DATA = '../data/'
PATH_MODELS = '../models/'



# load datasets

try:
    df_data = pd.read_csv(f'{PATH_DATA}Morphset.csv')
except:
    print("Please unzip data by running: unzip ../data/Morphset.csv.zip -d ../data/")

geom_feat = GeometricFeaturesDlib(full_size=FULL_FEATURES)    
frontalizer = LandmarkFrontalizationDlib(
    file_frontalization_model=f'{PATH_MODELS}model_frontalization.npy'
    )


# calculate features
ls_features = []
for i in range(len(df_data)):
    print('Processing face ' + str(i) + 
          ' out of ' + str(len(df_data)) + 
          ' [' + str(round((i*100)/len(df_data),3)) + '%]'
          )
    landmarks_raw = df_data.iloc[i,8:].values  # get landmarls (136,1)
    landmarks_raw = np.reshape(landmarks_raw, (2,68)).T  # transform to (68,2)
    dict_landmarks = frontalizer.frontalize_landmarks(
        landmarks_object=landmarks_raw
        )
    landmarks_frontal = dict_landmarks['landmarks_frontal']
    features = geom_feat.get_features(landmarks_frontal).astype(np.float16)
    ls_features.append(features)
         
# save features (for future reuse)    
features = np.array(ls_features, dtype=np.float16)
np.save(
        f'{PATH_DATA}features_fullfeatures={FULL_FEATURES}.npy', 
        features, 
        allow_pickle=True, 
        fix_imports=True
        )




 