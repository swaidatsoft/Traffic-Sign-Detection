# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:51:05 2019
Test Model to test our model result reading by images file
This test model to test JUST FOUR Traffic Sign from dataset (4000 imgs) because rasspberry pi has no enough memory to handel all TS
@author: Sweidat

"""
from matplotlib import pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import keras
import glob




labels = ["No entry for vehicular traffic" , "noovertaking" , "Priority at next intersection" , "Stop and give way"] 
#for test_img in glob.glob("./Frames/*.jpg"):    
 #image = cv2.imread(test_img)
 #print(image)
 
image = cv2.imread("C:/Users/User/.spyder-py3/Our TSD-Final/4New Model/no-overtaking.jpg") # Reading by Image File "Enter your Image Path"
#plt.imshow(image) 
image = cv2.resize(image,(64,64))
img=image.reshape(-1,64,64,3)
img = np.array(img).astype(np.float32)
model = tf.keras.models.load_model("traffic_Sign_Sweidat4TS.model")# Training Modelfour TS
pred= model.predict_classes(img)
plt.imshow(image)
print(labels[int(pred)])
  


#print(np.max(pred))
#y_classes = pred.argmax(axis=-1)
#print(y_classes)
#y_classes = keras.np_utils.probas_to_classes(pred)
#print(y_classes)