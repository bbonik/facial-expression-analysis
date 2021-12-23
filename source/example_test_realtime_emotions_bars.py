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
EXPORT_TO_GIF = False  # set to True if you want to record into a gif movie
EXP_AVG = 0.7  # 70% the new value 30% the old value (temporal smoothing)


cap = cv2.VideoCapture(0)  # camera object
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

emotion_estimator = EmotionsDlib(
    file_emotion_model='../models/model_emotion_pls=30_fullfeatures=False.joblib', 
    file_frontalization_model='../models/model_frontalization.npy'
    )



# initialize visualizations
fig = plt.figure(figsize=(15,10))
grid = plt.GridSpec(4, 3)
plt.suptitle('(press q to quit) \n        ')

plt.subplot(grid[0,0])
plt.title('Captured face')
plt.grid(False)
plt.axis((False))
plt.subplot(grid[0,1])
plt.title('Original landmarks')

plt.subplot(grid[0,2])
plt.title('Frontalized landmarks')

plt.subplot(grid[1,:])
plt.title('Valence')
plt.xlim((-1,1))
plt.xticks(ticks=[-1,1], labels=['Negative', 'Positive'])
plt.yticks([])
plt.grid(False)

plt.subplot(grid[2,:])
plt.title('Arousal')
plt.xlim((-1,1))
plt.xticks(ticks=[-1,1], labels=['Passive', 'Energetic'])
plt.yticks([])
plt.grid(False)

plt.subplot(grid[3,:])
plt.title('Intensity')
plt.xlim((0,1))
plt.xticks(ticks=[0,1], labels=['Neutral', 'Intense'])
plt.yticks([])
plt.grid(False)

plt.tight_layout()
plt.show()
axes = fig.get_axes()


f = 0
points_av = None
points_arousal = None
points_valence = None
points_intensity = None
disp_arousal = 0
disp_valence = 0
disp_intensity = 0
ls_captures = []

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
            disp_valence = round(
                EXP_AVG * valence + (1-EXP_AVG) * disp_valence, 2
                )
            disp_arousal = round(
                EXP_AVG * arousal + (1-EXP_AVG) * disp_arousal, 2
                )
            disp_intensity = round(
                EXP_AVG * intensity + (1-EXP_AVG) * disp_intensity, 2
                )
            

            
            # visualizations         
            if landmarks is not None:
                
                # face
                left = face.left()
                right = face.right()
                top = face.top()
                bottom = face.bottom()
                face_crop = frame[top:bottom, left:right,:]
                face_crop = face_crop[:,:,::-1]  # bgr -> rgb
                
                # original face
                axes[0].clear()
                axes[0].grid(False)
                axes[0].axis('off')
                axes[0].imshow(face_crop)
                axes[0].set_title('Estimated emotion name: \n' + emotion_name)
                
                # original landmarks
                axes[1].clear()
                plot_landmarks(
                    landmarks, 
                    axis=axes[1], 
                    title='Original'
                    )
                
                # frontalized landmarks
                axes[2].clear()
                plot_landmarks(
                    landmarks_frontal, 
                    axis=axes[2], 
                    title='Frontalized landmarks'
                    )
                
                # valence
                if points_valence is not None: 
                    points_valence.remove()
                if (disp_valence <= 0.33) & (disp_valence >= -0.33):
                    color = (1,0.7,0.7)
                elif (disp_valence <= 0.66) & (disp_valence >= -0.66):
                    color = (1,0.35,0.35)
                else:
                    color = (1,0,0)
                points_valence, = axes[3].barh(
                    y=0, 
                    width=disp_valence, 
                    color=color
                    )
                axes[3].set_title(f'Valence: {disp_valence}')
                
                # arousal
                if points_arousal is not None: 
                    points_arousal.remove()
                if (disp_arousal <= 0.33) & (disp_arousal >= -0.33):
                    color = (0.7,0.7,1)
                elif (disp_arousal <= 0.66) & (disp_arousal >= -0.66):
                    color = (0.35,0.35,1)
                else:
                    color = (0,0,1)
                points_arousal, = axes[4].barh(
                    y=0, 
                    width=disp_arousal, 
                    color=color
                    )
                axes[4].set_title(f'Arousal: {disp_arousal}')
                
                # intensity
                if points_intensity is not None: 
                    points_intensity.remove()
                if (disp_intensity <= 0.33) & (disp_intensity >= -0.33):
                    color = (0.7,1,0.7)
                elif (disp_intensity <= 0.66) & (disp_intensity >= -0.66):
                    color = (0.35,1,0.35)
                else:
                    color = (0,1,0)
                points_intensity, = axes[5].barh(
                    y=0, 
                    width=disp_intensity, 
                    color=color
                    )
                axes[5].set_title(f'Intensity: {disp_intensity}')
                

                plt.pause(0.001)  # needed for updating the graph
                plt.show()   
                
                
                # get image from plot (to be saved later as gif)
                if EXPORT_TO_GIF is True:
                    ax = fig.gca()
                    fig.tight_layout(pad=0)
                    ax.margins(0)  # To remove the huge white borders
                    fig.canvas.draw()
                    image_from_plot = np.frombuffer(
                        fig.canvas.tostring_rgb(), 
                        dtype=np.uint8
                        )
                    image_from_plot = image_from_plot.reshape(
                        fig.canvas.get_width_height()[::-1] + (3,)
                        )
                    ls_captures.append(image_from_plot)
                
            f += 1
    
    
    # When everything done, release resources
    cap.release()
    cv2.destroyAllWindows()
    
    
# uncomment to save to a gif image
if EXPORT_TO_GIF is True:
    import imageio
    imageio.mimsave(
        '../data/images/example2.gif', 
        ls_captures, 
        format='GIF', 
        fps=3
        )