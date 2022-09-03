
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from numpy import load

# x= load('training.npy')
# y=load('training_label.npy')
# x_val =load('val.npy')
# y_val=load('val_label.npy')
# x_test=load('tesy.npy')
# y_test = load('test_label.npy')

#with =1 0 
#without=0 1
Face = load_model('mask.h5')
imgtest1= cv2.imread('badboy.png',0)
# imgtest2=cv2.imread('tst2with.jpg',0)
imgtest1=cv2.resize(imgtest1,(64,64))
# imgtest2=cv2.resize(imgtest2,(64,64))
img1=(np.array(imgtest1)/ 255)
img1=np.reshape(img1,(1,64,64))
# img2=(np.array(imgtest2)/ 255)

prediction1 = Face.predict(img1)
# prediction2 = Face.predict(img2)
predict_class1=np.argmax(prediction1,axis = 1)
# predict_class2=np.argmax(prediction2,axis = 1) 
print(prediction1, predict_class1) #'\n', predict_class2