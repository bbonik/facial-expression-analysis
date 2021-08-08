#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This script demontrates facial expression analysis on static images. 
Images are open source and downloaded from: https://pixabay.com/
(Free for commercial use, No attribution required)

"""

import dlib
import matplotlib.pyplot as plt
from emotions_dlib import EmotionsDlib, plot_landmarks
import imageio

plt.close('all')



# uncomment and select various example images
image = imageio.imread('../data/images/excited.jpg')
# image = imageio.imread('../data/images/happy.jpg')
# image = imageio.imread('../data/images/pleased.jpg')
# image = imageio.imread('../data/images/sad.jpg')
# image = imageio.imread('../data/images/angry.jpg')


# setup detectors and models
detector = dlib.get_frontal_face_detector()  # face detector
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
emotion_estimator = EmotionsDlib(
    file_emotion_model='../models/model_emotion_pls=30_fullfeatures=False.joblib', 
    file_frontalization_model='../models/model_frontalization.npy'
    )


faces = detector(image)  # detect faces


for i, face in enumerate(faces):  # go through detected faces
    
    landmarks_object = predictor(image, face)  # detect landmarks
    dict_emotions = emotion_estimator.get_emotions(  # get emotions
        landmarks_object
        )
    
    # parse output
    landmarks = dict_emotions['landmarks']['raw']
    landmarks_frontal = dict_emotions['landmarks']['frontal']
    arousal = dict_emotions['emotions']['arousal']
    valence = dict_emotions['emotions']['valence']
    intensity = dict_emotions['emotions']['intensity']
    emotion_name = dict_emotions['emotions']['name']
    
    if landmarks is not None:
        
        # initialize new image
        fig = plt.figure(figsize=(10,3))
        
        # plot detected face
        plt.subplot(1,4,1)
        plt.title('Detected face')
        x1 = landmarks_object.rect.left()
        y1 = landmarks_object.rect.top()
        x2 = x1 + landmarks_object.rect.width()
        y2 = y1 + landmarks_object.rect.height()
        plt.imshow(image[y1:y2, x1:x2, :])
        plt.axis(False)
        
        # plot original and frontalized landmarks
        plt.subplot(1,4,2)
        plt.title('Original landmarks')
        plt.subplot(1,4,3)
        plt.title('Frontalized landmarks')
        plt.suptitle('Face ' + str(i+1))
        plt.tight_layout()
        axes = fig.get_axes()
        plot_landmarks(landmarks, axis=axes[1])
        plot_landmarks(landmarks_frontal, axis=axes[2])
        
        # display emotion info
        plt.subplot(1,4,4)
        ax = plt.gca()
        plt.title('Emotion details')
        text = 'Arousal: ' + str(arousal) + '\n'
        text += 'Valence: ' + str(valence) + '\n'
        text += 'Intensity: ' + str(intensity) + '\n'
        text += 'Name: ' + str(emotion_name) + '\n'
        text += '\n\n\n\n'
        plt.rcParams.update({'font.size': 12})
        ax.set_axis_off()  # set background white
        ax.set_frame_on(True)
        ax.grid(False)
        ax.text(
            x=0.8,
            y=0, 
            s=text,
            horizontalalignment="right",
            color="black"
            )
        
        plt.show()
