#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vasileios Vonikakis
@email: bbonik@gmail.com

This script gives a real-time demonstration of the facial expression analysis
model, and updates a simple set of graphs. It makes use of your camera and
analyses your face in real-time. DLIB landmarks are not very illumination
invariant, so this works better when there are no shadows on the face.
"""


import numpy as np
import cv2
import dlib
import matplotlib.pyplot as plt
from emotions_dlib import EmotionsDlib, plot_landmarks


plt.close('all')
plt.style.use('seaborn')
GRAPH_LENGTH = 30  # frames
EXP_AVG = 0.7  # 70% the new value 30% the old value


cap = cv2.VideoCapture(0)  # camera object
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

emotion_estimator = EmotionsDlib(
    file_emotion_model='../models/model_emotion_pls=30_fullfeatures=False.joblib', 
    file_frontalization_model='../models/model_frontalization.npy'
    )



# initialize visualizations
fig = plt.figure(figsize=(15,10))
grid = plt.GridSpec(5, 6)
plt.suptitle('(press q to quit) \n        ')

plt.subplot(grid[:2,:2])
plt.title('Original landmarks')

plt.subplot(grid[:2,2:4])
plt.title('Frontalized landmarks')

plt.subplot(grid[:2,4:])
plt.title('Arousal Valence Space')
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.axhline(linewidth=3, color='k')
plt.axvline(linewidth=3, color='k')
plt.grid(True)

plt.subplot(grid[2,:])
plt.ylabel('Valence')
# plt.xlabel('Frames')
plt.ylim((-1.01, 1.01))
plt.xticks([])
plt.yticks([-1, -0.5, 0, 0.5, 1])  
plt.xlim((0 ,GRAPH_LENGTH-1))
plt.grid(True)

plt.subplot(grid[3,:])
plt.ylabel('Arousal')
# plt.xlabel('Frames')
plt.ylim((-1.01, 1.01))
plt.xlim((0, GRAPH_LENGTH-1))
plt.xticks([])  
plt.yticks([-1, -0.5, 0, 0.5, 1])  
plt.grid(True)

plt.subplot(grid[4,:])
plt.ylabel('Intensity')
# plt.xlabel('Frames')
plt.ylim((0, 1.01))
plt.xticks([])
plt.yticks([-1, -0.5, 0, 0.5, 1])  
plt.xlim((0, GRAPH_LENGTH-1))
plt.grid(True)

plt.tight_layout()
plt.show()
axes = fig.get_axes()






f = 0
points_av = None
points_arousal = None
polyg_arousal = None
points_valence = None
polyg_valence = None
points_intensity = None
polyg_intensity = None

ls_arousal = []
ls_valence = []
ls_intensity = []

ls_captures = []

disp_arousal = 0
disp_valence = 0
disp_intensity = 0

if __name__=="__main__":

    while(f < 300):  # for 300 frames
        print('frame', f)
        
        #TODO: find a better way of closing this window. At the moment it
        #keeps going!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Q pressed')
            cap.release()
            cv2.destroyAllWindows()
            f=300
            break
    
        
        
        ret, frame = cap.read()  # capture a frame 
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(image)  # detect faces
        
        if len(faces) > 0:  # if there are faces detected
        
            # find larger detected face and select it 
            face_size = 0
            idx_largest_face = 0
            if len(faces) > 1:
                for i,face in enumerate(faces):
                    current_size = ((face.bottom() - face.top()) * 
                                    (face.right() - face.left()))
                    if face_size < current_size: 
                        face_size = current_size
                        idx_largest_face = i
             
            face = faces[idx_largest_face]  # the face to be processed
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
            
            # explonential averaging
            disp_valence = EXP_AVG * valence + (1-EXP_AVG) * disp_valence
            disp_arousal = EXP_AVG * arousal + (1-EXP_AVG) * disp_arousal
            disp_intensity = EXP_AVG * intensity + (1-EXP_AVG) * disp_intensity
            
            # append buffer lists
            ls_arousal.append(disp_arousal)
            ls_valence.append(disp_valence)
            ls_intensity.append(disp_intensity)
            if len(ls_arousal) > GRAPH_LENGTH: 
                del ls_arousal[0]
                del ls_valence[0]
                del ls_intensity[0]
            ls_xs = [i for i in range(len(ls_arousal))]
            ls_0z = [0 for i in range(len(ls_arousal))]
            
            # visualizations         
            if landmarks is not None:
                
                # original landmarks
                axes[0].clear()
                plot_landmarks(
                    landmarks, 
                    axis=axes[0], 
                    title='Original'
                    )
                
                # frontalized landmarks
                axes[1].clear()
                plot_landmarks(
                    landmarks_frontal, 
                    axis=axes[1], 
                    title='Frontalized landmarks'
                    )
                
                # arousal-valence plane
                if points_av is not None: points_av.remove()
                points_av, = axes[2].plot(
                    disp_valence, 
                    disp_arousal, 
                    color='r', 
                    marker='.', 
                    markersize=20
                    )
                axes[2].set_title(
                    'AR=' + str(arousal) + 
                    ' | VA=' + str(valence) + 
                    ' | IN=' + str(intensity) + 
                    '| \n' + emotion_name
                    )
                
                if points_valence is not None: 
                    points_valence.remove()
                    polyg_valence.remove()
                points_valence, = axes[3].plot(
                    ls_valence, 
                    linewidth=1, 
                    color=(0.8,0.2,0.2)
                    )
                polyg_valence = axes[3].fill_between(
                    ls_xs, 
                    ls_valence, 
                    ls_0z, 
                    interpolate = True, 
                    alpha=0.3, 
                    color=(0.8,0.2,0.2)
                    )

                if points_arousal is not None: 
                    points_arousal.remove()
                    polyg_arousal.remove()
                points_arousal, = axes[4].plot(
                    ls_arousal, 
                    linewidth=1, 
                    color=(0.2,0.2,0.8)
                    )
                polyg_arousal = axes[4].fill_between(
                    ls_xs, 
                    ls_arousal, 
                    ls_0z, 
                    interpolate = True, 
                    alpha=0.3, 
                    color=(0.2,0.2,0.8)
                    )
                
                if points_intensity is not None: 
                    points_intensity.remove()
                    polyg_intensity.remove()
                points_intensity, = axes[5].plot(
                    ls_intensity, 
                    linewidth=1, 
                    color=(0.2,0.8,0.2)
                    )
                polyg_intensity = axes[5].fill_between(
                    ls_xs, 
                    ls_intensity, 
                    ls_0z, 
                    interpolate = True, 
                    alpha=0.3, 
                    color=(0.2,0.8,0.2)
                    )
                
                plt.pause(0.001)  # needed for updating the graph
                plt.show()               
                
            f += 1
    
    
    # When everything done, release resources
    cap.release()
    cv2.destroyAllWindows()