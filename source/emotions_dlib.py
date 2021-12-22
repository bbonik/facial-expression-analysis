#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 10:24:27 2021

@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This is a library of classes designed to process facial landmarks extracted by
DLIB and output the dimensional emotion of the face, as well as the 
frontalized version of the coordinate landmarks. 
"""


import numpy as np
import math
from joblib import load
import matplotlib.pyplot as plt



'''
------------------------------------------------------------------------------
                                  EmotionsDlib
------------------------------------------------------------------------------
'''

class EmotionsDlib():
    '''
    Class for estimating dimensional emotions (Arousal, Valence, Intensity) 
    from a set of facial landmarks extracted by DLIB. 
    '''

    def __init__(
            self, 
            file_emotion_model, 
            file_frontalization_model
            ):
        '''
        file_emotion_model: string
            Path to the .joblib model file for emotions. 
        file_frontalization_model: string
            Path to the .npy frontalization weights file. The file should be a 
            saved numpy float array of size (2N+1, 2N), where N is the number 
            landmarks. For DLIB, N=68, which means that the array should be of
            size (137,136). The +1 stands for the interception of the 
            frontalization regression model.
        full_size: Bool
            Include all the landmarks or not. If True, all 68 DLIB landmarks 
            are included, resulting in a feature dimensionality of 2278. If
            False, the first 17 landmarks (0 to 16), corresponding to the 
            jawline, are excluded, resulting in a feature dimensionality of
            1275.
        '''
            
        model = None
        self.full_features = True  # default feature size
        
        # loading pickled emotion model
        try:
            model = load(file_emotion_model)
            print('Emotion model loaded successfully.')
        except: 
            print('Problem loading emotion model!')
         
        if model is not None:
            self.emotion_model = model['model']
            self.full_features = model['full_features']
            self.components = model['components']
            print(
                'Model components:', self.components, 
                  ' | full feature size:', self.full_features
                  )
        
        # determine the size of features
        self.geom_feat = GeometricFeaturesDlib(full_size=self.full_features)        
        self.frontalizer = LandmarkFrontalizationDlib(
            file_frontalization_model=file_frontalization_model
            )
        
        
        
    def get_emotions(self, landmarks_object):
        # Estimates dimensional emotions given a set of DLIB facial landmarks
        
        dict_landmarks = self.frontalizer.frontalize_landmarks(
            landmarks_object
            )
        landmarks_frontal = dict_landmarks['landmarks_frontal']
        landmarks = dict_landmarks['landmarks_raw']
        
        features = self.geom_feat.get_features(landmarks_frontal)
        features = features.reshape(1, -1)
        avi_predict = self.emotion_model.predict(features)
        avi_predict = np.round(avi_predict, 3)
        
        # TODO: estimate angle from original regression outputs, use intensity
        # as radius and re-estimate arousal and valence
        
        # truncate estimations within required limits
        arousal = avi_predict[0][0]
        if arousal > 1: arousal=1
        elif arousal < -1: arousal=-1
        
        valence = avi_predict[0][1]
        if valence > 1: valence=1
        elif valence < -1: valence=-1
        
        # intensity = avi_predict[0][2]  # this overestimates the intensity
        intensity = round(math.sqrt(valence ** 2 + arousal ** 2), 3)  # altern
        if intensity > 1: intensity=1
        elif intensity < 0: intensity=0
        
        emotion_name = self.avi_to_text(
            arousal=arousal, 
            valence=valence
            )
        
        emotions = {}
        emotions['emotions'] = {}
        emotions['emotions']['arousal'] = arousal
        emotions['emotions']['valence'] = valence
        emotions['emotions']['intensity'] = intensity
        emotions['emotions']['name'] = emotion_name
        emotions['landmarks'] = {}
        emotions['landmarks']['raw'] = landmarks
        emotions['landmarks']['frontal'] = landmarks_frontal
        
        return emotions
        
        
    
    def avi_to_text(self, arousal, valence, intensity=None):
        '''
        Generates a text description for a pair of arousal-valence values
        based on Russell's Circulmplex Model of Affect.
        Russell, J. A. (1980). A circumplex model of affect. Journal of 
        Personality and Social Psychology, 39(6), 1161–1178. 
        '''
    
        expression_intensity = "?"
        expression_name = "?"
        
        if intensity is None: intensity = math.sqrt(arousal**2 + valence**2)
            
        ls_expr_intensity = [
            "Slightly", "Moderately", "Very", "Extremely"
            ]
        ls_expr_name = [
            "pleased", "happy", "delighted", "excited", "astonished", 
            "aroused", # first quarter
            
            "tensed", "alarmed", "afraid", "annoyed", "distressed", 
            "frustrated", "miserable", # second quarter
            
            "sad", "gloomy", "depressed", "bored", "droopy", "tired", 
            "sleepy", # third quarter
            
            "calm", "serene", "content", "satisfied"  # fourth quarter
        ]
    
        # analyzing intensity
        if intensity < 0.1:
            expression_name = "Neutral"
            expression_intensity = ""
        else: 
            
            if intensity < 0.325:
                expression_intensity = ls_expr_intensity[0]
            elif intensity < 0.55:
                expression_intensity = ls_expr_intensity[1]
            elif intensity < 0.775:
                expression_intensity = ls_expr_intensity[2]
            else:
                expression_intensity = ls_expr_intensity[3]
    
    
            # analyzing epxression name
    
            # compute angle [0,360]
            if valence == 0:
                if arousal >= 0:
                    theta = 90
                else:
                    theta = 270
            else:
                theta = math.atan(arousal / valence)
                theta = theta * (180 / math.pi)
    
                if valence < 0:
                    theta = 180 + theta
                elif arousal < 0:
                    theta = 360 + theta
    
            # estimate expression name
    
            if theta < 16 or theta > 354:
                expression_name = ls_expr_name[0]
            elif theta < 34:
                expression_name = ls_expr_name[1]
            elif theta < 62.5:
                expression_name = ls_expr_name[2]
            elif theta < 78.5:
                expression_name = ls_expr_name[3]
            elif theta < 93:
                expression_name = ls_expr_name[4]
            elif theta < 104:
                expression_name = ls_expr_name[5]
            elif theta < 115:
                expression_name = ls_expr_name[6]
            elif theta < 126:
                expression_name = ls_expr_name[7]
            elif theta < 137:
                expression_name = ls_expr_name[8]
            elif theta < 148:
                expression_name = ls_expr_name[9]
            elif theta < 159:
                expression_name = ls_expr_name[10]
            elif theta < 170:
                expression_name = ls_expr_name[11]
            elif theta < 181:
                expression_name = ls_expr_name[12]
            elif theta < 192:
                expression_name = ls_expr_name[13]
            elif theta < 203:
                expression_name = ls_expr_name[14]
            elif theta < 215:
                expression_name = ls_expr_name[15]
            elif theta < 230:
                expression_name = ls_expr_name[16]
            elif theta < 245:
                expression_name = ls_expr_name[17]
            elif theta < 260:
                expression_name = ls_expr_name[18]
            elif theta < 280:
                expression_name = ls_expr_name[19]
            elif theta < 300:
                expression_name = ls_expr_name[20]
            elif theta < 320:
                expression_name = ls_expr_name[21]
            elif theta < 340:
                expression_name = ls_expr_name[22]
            elif theta < 354:
                expression_name = ls_expr_name[23]
            else:
                expression_name = "Unknown"
                expression_intensity = ""
        
        # TODO: return also variable output and not only string
    
        return expression_intensity + " " + expression_name
        



'''
------------------------------------------------------------------------------
                            GeometricFeaturesDlib
------------------------------------------------------------------------------
'''


class GeometricFeaturesDlib():
    '''
    Class for extracting geometric features from a set of facial landmarks.
    The class assumes 68 landmarks, following the DLIB annotation style. 
    '''

    def __init__(self, full_size=True):
        '''
        full_size: Bool
            Include all the landmakrs or not. If True, all 68 DLIB landmarks 
            are included, resulting in a feature dimensionality of 2278. If
            False, the first 17 landmarks (0 to 16), corresponding to the 
            jawline, are excluded, resulting in a feature dimensionality of
            1275.
        '''
        
        TOTAL_LANDMARKS = 68  # based on DLIB annotation
        
        if full_size is True:
            landmark_indx = tuple(i for i in range(TOTAL_LANDMARKS))   #[0,67]
        else:
            landmark_indx = tuple(i for i in range(17,TOTAL_LANDMARKS))#[17,67]
            
        
        # computing feature template by estimating all the unique pairs  
        # (N choose 2, where N is the number of landmarks) between all possible
        # pairs of landmaks
        
        feature_template = []  
        
        for i in range(len(landmark_indx)):
            for j in range(i+1,len(landmark_indx)):
                feature_template.append([i,j])
        
        self.feature_template = np.array(feature_template, dtype=np.int16)
        self.landmark_indx = landmark_indx
        print('Feature template size:', self.feature_template.shape)
                
        
        
    
    def get_features(self, landmarks_dlib):
        '''
        landmarks_dlib: numpy float array of size (68,2) of DLIB landmarks
            Column 0 includes all X ladmark coordinates
            Column 1 includes all Y ladmark coordinates
        '''
        
        # compute normalised Eucledian distances between landmark pairs
        
        distance = (landmarks_dlib[self.feature_template[:,0]] - 
                    landmarks_dlib[self.feature_template[:,1]])
        distance = distance.astype(np.float32)  # otherwise np.sqrt compains
        distance = distance * distance  # ^2
        geometric_features = np.sqrt(distance[:,0] + distance[:,1])
        geometric_features /= self.get_scale(landmarks_dlib)
        
        return geometric_features
    
    
    

    def get_scale(self, landmarks_dlib):
        '''
        Computes an estimation of scale for a set of DLIB facial landmarks.
        Scale is defined as the mean eucledian distance of all the landmarks 
        to the mean x,y landmark of the face.
        sqrt( mean ( (Lx-Lxmean)^2 + (Ly-Lymean)^2 ) ) )  
        '''
        
        # keep only the available landmarks
        landmarks = landmarks_dlib[self.landmark_indx, :]
        landmarks_standard = landmarks - np.mean(landmarks, axis=0)
        landmark_scale = math.sqrt(
            np.mean(
                np.sum(landmarks_standard**2, axis=1)
                )
            )
            
        return landmark_scale
        
 

'''
------------------------------------------------------------------------------
                        LandmarkFrontalizationDlib
------------------------------------------------------------------------------
'''
  

class LandmarkFrontalizationDlib():
    '''
    Class for frontalizing facial landmarks extracted by DLIB.
    
    file_frontalization_weights: string
            Path to the .npy frontalization weights file. The file should be a 
            saved numpy float array of size (2N+1, 2N), where N is the number 
            landmarks. For DLIB N=68, which means that the array should be of
            size (137,136). The +1 stands for the interception of the 
            frontalization regression model.
    '''

    def __init__(self, file_frontalization_model):
        '''
        Important: the frontalization model (weights) will work only with DLIB.
        For other landmark engines, the frontalization model needs to be 
        retrained. For this check: 
        https://github.com/bbonik/facial-landmark-frontalization
    
        file_frontalization_model: string
            Path to the .npy frontalization weights file. The file should be a 
            saved numpy float array of size (2N+1, 2N), where N is the number 
            landmarks. For DLIB N=68, which means that the array should be of
            size (137,136). The +1 stands for the interception of the 
            frontalization regression model.
        '''
        
        TOTAL_LANDMARKS = 68  # based on DLIB annotation
        frontalization_weights = None
        
        # loading frontalization weights
        try:
            frontalization_weights = np.load(file_frontalization_model)
            print('Frontalization weights loaded successfully.')
        except: 
            print('Problem loading frontalization weights!')
        
        if ((frontalization_weights.shape[0] != 2 * TOTAL_LANDMARKS + 1) |
            (frontalization_weights.shape[1] != 2 * TOTAL_LANDMARKS)):
            print('Frontalization weights not adequate for DLIB landmarks!')
            
        self.TOTAL_LANDMARKS = TOTAL_LANDMARKS
        self.frontalization_weights = frontalization_weights
        
        
        
        
    def frontalize_landmarks(self, landmarks_object):
    
        '''
        ----------------------------------------------------------------------
                          Frontalize a non-frontal face shape
        ----------------------------------------------------------------------
        Takes an array or a list of facial landmark coordinates and returns a 
        frontalized version of them (how the face shape would look like from 
        the frontal view). Assumes 68 points with a DLIB annotation scheme. As
        described in the paper: 
        V. Vonikakis, S. Winkler. (2020). Identity Invariant Facial Landmark 
        Frontalization for Facial Expression Analysis. ICIP2020, October 2020.
        
        INPUTS
        ------
        landmarks_object: numpy array [68,2] or a dlib landmark object.
            The landmark array of the input face shape. Should follow the DLIB 
            annotation scheme. You can directly pass a 
            dlib.full_object_detection of the facial landmarks and it will be 
            converted to a numpy array. 
    
        OUTPUT
        ------
        landmarks: dictionary of landmarks (numpy array [68,2])
            The landmark array of the raw and frontalized input face shape. 
        '''
        
        if type(landmarks_object) is not np.ndarray:
            landmarks_array = self.get_landmark_array(landmarks_object)
        else:
            landmarks_array = landmarks_object
        landmarks_standard = self.get_procrustes(
            landmarks_array, 
            template_landmarks=None
            )
        landmark_vector = np.hstack(
            (
                landmarks_standard[:,0].T, 
                landmarks_standard[:,1].T, 
                1  # add interception
                )  
            )
        landmarks_frontal = np.matmul(
            landmark_vector, 
            self.frontalization_weights
            )
        landmarks_frontal = self.get_landmark_matrix(landmarks_frontal)
        
        landmarks = {}
        landmarks['landmarks_frontal'] = landmarks_frontal
        landmarks['landmarks_raw'] = landmarks_array
        
        return landmarks
        

    
    def get_landmark_array(self, landmarks_dlib_obj):
        # Gets a DLIB landmarks object and returns a [68,2] numpy array with 
        # the landmark coordinates.
        
        landmark_array = np.zeros([self.TOTAL_LANDMARKS,2])
        
        for i in range(self.TOTAL_LANDMARKS):
            landmark_array[i,0] = landmarks_dlib_obj.part(i).x
            landmark_array[i,1] = landmarks_dlib_obj.part(i).y
            
        return landmark_array  # numpy array
    
    
    
    def get_landmark_matrix(self, ls_coord):
        # Gets a list of landmark coordinates and returns a [N,2] numpy array
        # of the coordinates. Assumes that the list follows the scheme:
        # [x1, x2, ..., xN, y1, y2, ..., yN]
        
        mid = len(ls_coord) // 2
        landmarks = np.array( [ ls_coord[:mid], ls_coord[mid:] ] )
        return landmarks.T
    
    
    
    def get_procrustes(
        self,
        landmarks, 
        translate=True, 
        scale=True, 
        rotate=True, 
        template_landmarks=None):
        '''
        ----------------------------------------------------------------------
                            Procrustes shape standardization
        ----------------------------------------------------------------------
        Standardizes a given face shape, compensating for translation, scaling 
        and rotation. If a template face is also given, then the standardized 
        face is adjusted so as its facial parts will be displaced according to  
        the template face. More information can be found in this paper:
            
        V. Vonikakis, S. Winkler. (2020). Identity Invariant Facial Landmark 
        Frontalization for Facial Expression Analysis. ICIP2020, October 2020.
        
        INPUTS
        ------
        landmarks: numpy array [68,2]
            The landmark array of the input face shape. Should follow the DLIB 
            annotation scheme.
        translate: Boolean
            Whether or not to compensate for translation.
        scale: Boolean
            Whether or not to compensate for scaling.
        rotation: Boolean
            Whether or not to compensate for rotation.
        template_landmarks: numpy array [68,2] or None
            The landmark array of a template face shape, which will serve as 
            guidence to displace facial parts. Should follow the DLIB 
            annotation scheme. If None, no displacement is applied. 
        
        OUTPUT
        ------
        landmarks_standard: numpy array [68,2]
            The standardised landmark array of the input face shape.
            
        '''
        
        landmarks_standard = landmarks.copy()
        
        # translation
        if translate is True:
            landmark_mean = np.mean(landmarks, axis=0)
            landmarks_standard = landmarks_standard - landmark_mean
        
        # scale
        if scale is True:
            landmark_scale = math.sqrt(
                np.mean(np.sum(landmarks_standard**2, axis=1))
                )
            landmarks_standard = landmarks_standard / landmark_scale
        
        if rotate is True:
            # rotation
            center_eye_left, center_eye_right = self.get_eye_centers(
                landmarks_standard
                )
            # distance between the eyes
            dx = center_eye_right[0] - center_eye_left[0]
            dy = center_eye_right[1] - center_eye_left[1]
        
            if dx != 0:
                f = dy / dx
                a = math.atan(f)  # rotation angle in radians
                # ad = math.degrees(a)
                # print('Eye2eye angle=', ad)
        
            R = np.array([
                [math.cos(a), -math.sin(a)], 
                [math.sin(a), math.cos(a)]
                ])  # rotation matrix
            landmarks_standard = np.matmul(landmarks_standard, R)
        
        '''
        adjusting facial parts to a tamplate face displacing face parts to 
        predetermined positions (as defined by the template_landmarks), except 
        from the eyebrows, which convey important expression information 
        attention! this only makes sense for frontal faces!
        '''
        if template_landmarks is not None:
            
            # mouth
            anchorpoint_template = np.mean(template_landmarks[50:53,:], axis=0)
            anchorpoint_input = np.mean(landmarks_standard[50:53,:], axis=0)
            displacement = anchorpoint_template - anchorpoint_input
            landmarks_standard[48:,:] += displacement
            
            # right eye
            anchorpoint_template = np.mean(template_landmarks[42:48,:], axis=0)
            anchorpoint_input = np.mean(landmarks_standard[42:48,:], axis=0)
            displacement = anchorpoint_template - anchorpoint_input
            landmarks_standard[42:48,:] += displacement
            # right eyebrow (same displaycement as the right eye)
            landmarks_standard[22:27,:] += displacement  # TODO: only X?
            
            # left eye
            anchorpoint_template = np.mean(template_landmarks[36:42,:], axis=0)
            anchorpoint_input = np.mean(landmarks_standard[36:42,:], axis=0)
            displacement = anchorpoint_template - anchorpoint_input
            landmarks_standard[36:42,:] += displacement
            # left eyebrow (same displaycement as the left eye)
            landmarks_standard[17:22,:] += displacement  # TODO: only X?
            
            # nose
            anchorpoint_template = np.mean(template_landmarks[27:36,:], axis=0)
            anchorpoint_input = np.mean(landmarks_standard[27:36,:], axis=0)
            displacement = anchorpoint_template - anchorpoint_input
            landmarks_standard[27:36,:] += displacement
            
            # jaw
            anchorpoint_template = np.mean(template_landmarks[:17,:], axis=0)
            anchorpoint_input = np.mean(landmarks_standard[:17,:], axis=0)
            displacement = anchorpoint_template - anchorpoint_input
            landmarks_standard[:17,:] += displacement
            
        return landmarks_standard
    
    
    
    def get_eye_centers(self, landmarks):
        # Given a numpy array of [68,2] facial landmarks, returns the eye  
        # centers of a face. Assumes the DLIB landmark scheme.
    
        landmarks_eye_left = landmarks[36:42,:]
        landmarks_eye_right = landmarks[42:48,:]
        
        center_eye_left = np.mean(landmarks_eye_left, axis=0)
        center_eye_right = np.mean(landmarks_eye_right, axis=0)
        
        return center_eye_left, center_eye_right



'''
------------------------------------------------------------------------------
                             Helper functions
------------------------------------------------------------------------------
'''



def plot_landmarks(landmarks, axis=None, color='k', title=None):
    '''
    ---------------------------------------------------------------------------
                      Creates a line drawing of a face shape
    ---------------------------------------------------------------------------
    Plots line segments between facial landmarks to form a face line drawing.
    Assumes 68 points with a DLIB annotation scheme.
    
    INPUTS
    ------
    landmarks: numpy array [68,2]
        The landmark array of the input face shape. Should follow the DLIB 
        annotation scheme.
    axis: matplotlib axis object or None
        If None, a new image will be created. If an axis object is passed, the
        new image will be drawn in the given axis.
    color: string
        The color with which the line segments will be drawn. Follows the
        matplotlib color scheme.
    title: string
        Title of the face line drawing. If None, no title is included.
    
    OUTPUT
    ------
    Line drawing of the input face shape.
        
    '''
    
    if axis is None:  # for standalone plot
        plt.figure()
        ax = plt.gca()
    else:  # for plots inside a subplot
        ax = axis
    
    # format shape
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_aspect(aspect=1)
    if title is not None: ax.set_title(title)
    
    # plotting points
    ax.plot(landmarks[:17,0], landmarks[:17,1], color)  # jaw
    
    ax.plot(landmarks[17:22,0], landmarks[17:22,1], color)  #left brow
    ax.plot(landmarks[22:27,0], landmarks[22:27,1], color)  # right brow
    
    ax.plot(landmarks[27:31,0], landmarks[27:31,1], color)  # nose top
    ax.plot(landmarks[31:36,0], landmarks[31:36,1], color)  # nose base
    
    ax.plot(landmarks[36:42,0], landmarks[36:42,1], color)  # left eye
    ax.plot([landmarks[41,0], landmarks[36,0]], 
            [landmarks[41,1], landmarks[36,1]], color)  # close left eye
    ax.plot(landmarks[42:48,0], landmarks[42:48,1], color)  # right eye
    ax.plot([landmarks[47,0], landmarks[42,0]], 
            [landmarks[47,1], landmarks[42,1]], color)  # close right eye
    
    ax.plot(landmarks[48:60,0], landmarks[48:60,1], color)  # outer mouth
    ax.plot([landmarks[59,0], landmarks[48,0]], 
            [landmarks[59,1], landmarks[48,1]], color)  # close outer mouth
    
    ax.plot(landmarks[60:68,0], landmarks[60:68,1], color)  # inner mouth
    ax.plot([landmarks[67,0], landmarks[60,0]], 
            [landmarks[67,1], landmarks[60,1]], color)  # close inner mouth
    
    if axis is None:  # for standalone plots
        plt.tight_layout()
        plt.show()
    