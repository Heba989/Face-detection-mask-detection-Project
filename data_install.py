
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

Wfiles= os.listdir("Final Project\Face Mask Dataset\Test\WithMask")
WOfiles= os.listdir("Final Project\Face Mask Dataset\Test\WithoutMask")
trwfiles =os.listdir("Final Project\Face Mask Dataset\Train\WithMask")
trwofiles=os.listdir("Final Project\Face Mask Dataset\Train\WithoutMask")
valwfiles=os.listdir("Final Project\Face Mask Dataset\Validation\WithMask")
valwofiles=os.listdir("Final Project\Face Mask Dataset\Validation\WithoutMask")
testimg =[]
testlabels=[]


# x=cv2.imread(f"Final Project\Face Mask Dataset\Test\WithMask\{Wfiles[0]}", 0)
# print(x, x.shape)
for file in Wfiles:
    x= cv2.imread(f"Final Project\Face Mask Dataset\Test\WithMask\{file}",0)
    testlabels.append('with')
    x=cv2.resize(x, (64,64))
    testimg.append(x)
for file in WOfiles:
     x= cv2.imread(f"Final Project\Face Mask Dataset\Test\WithoutMask\{file}",0)
     testlabels.append('without')
     x=cv2.resize(x, (64,64))
     testimg.append(x)
trainimg =[]
trainlabels=[]
for file in trwfiles:
    tr= cv2.imread(f"Final Project\Face Mask Dataset\Train\WithMask\{file}",0)
    trainlabels.append('with')
    tr=cv2.resize(tr, (64,64))
    trainimg.append(tr)
for file in trwofiles:
     tr= cv2.imread(f"Final Project\Face Mask Dataset\Train\WithoutMask\{file}",0)
     trainlabels.append('without')
     tr=cv2.resize(tr, (64,64))
     trainimg.append(tr)
valimg=[]
vallabels=[]
for file in valwfiles:
    val= cv2.imread(f"Final Project\Face Mask Dataset\Validation\WithMask\{file}",0)
    vallabels.append('with')
    val=cv2.resize(val, (64,64))
    valimg.append(val)
for file in valwofiles:
     val= cv2.imread(f"Final Project\Face Mask Dataset\Validation\WithoutMask\{file}",0)
     vallabels.append('without')
     val=cv2.resize(val, (64,64))
     valimg.append(val)

#normalizing :
trainim = (np.array(trainimg)/ 255)
testimg=(np.array(testimg)/255)
valimg=(np.array(valimg)/255)

#label encoder:
le = LabelEncoder()
tr_labels = le.fit_transform(trainlabels)
val_labels=le.fit_transform(vallabels)
test_labels= le.fit_transform (testlabels)

tr_labels= np_utils.to_categorical(tr_labels,2)
val_labels=np_utils.to_categorical(val_labels,2)
test_labels= np_utils.to_categorical(test_labels,2)
#saving data
np.save('training.npy', trainim)
np.save('training_label.npy', tr_labels) 
np.save('val.npy',valimg)
np.save('val_label.npy',val_labels)
np.save('tesy.npy',testimg)
np.save('test_label.npy', test_labels)



