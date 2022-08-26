from cProfile import label
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from numpy import load
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD, Adam, Nadam 
from tensorflow.keras.layers import Activation, Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.metrics import classification_report, confusion_matrix,precision_score, recall_score, f1_score
#Flickr-Faces-HQ Dataset (FFHQ)-data set 
#https://github.com/NVlabs/ffhq-dataset

#face detection model 

#mask detection model
x= load('training.npy')
y=load('training_label.npy')
x_val =load('val.npy')
y_val=load('val_label.npy')
x_test=load('tesy.npy')
y_test = load('test_label.npy')
print(y_test[2], y_test[-2])
# dropouts = [0.1, 0.2, 0.3, 0.4]
# for i in dropouts:
        
model = Sequential()
model.add(Conv2D(64, (5,5),input_shape = (64,64,1),
                activation = "relu", padding = "same"))

model.add(MaxPooling2D(2))
model.add(Dropout(0.1))
model.add(Conv2D(128,(3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(2))
model.add(Dropout(0.1))
# model.add(Conv2D(512, (3,3), activation="relu", padding="same"))
# model.add(MaxPooling2D(2))
# model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation = "relu"))
model.add(Dense(128, activation='relu',kernel_initializer ="he_uniform"))
model.add(Dense(2, activation='sigmoid', kernel_initializer ="he_uniform"))

model.summary()

model.compile(loss = "binary_crossentropy", 
            optimizer=("Adam"),
            metrics = ["accuracy"])

clf = model.fit(x,y,validation_data = (x_val, y_val), batch_size = 64, epochs = 15)

#test image 
imgtest1= cv2.imread('testwith.jpg',0)
# imgtest2=cv2.imread('tst2with.jpg',0)
imgtest1=cv2.resize(imgtest1,(64,64))
# imgtest2=cv2.resize(imgtest2,(64,64))
img1=(np.array(imgtest1)/ 255)
# img2=(np.array(imgtest2)/ 255)

prediction= model.predict(img1)
predict_class=np.argmax(prediction,axis = 1)
print(predict_class)


# plt.subplot(1,2,1)
# plt.plot(clf.history['loss'],'-g', label="Training")
# plt.plot(clf.history['val_loss'],'-y', label="Val")
# plt.ylabel(f"Loss {dropouts}")
# plt.subplot(1,2,2)
# plt.plot(clf.history['accuracy'],'-g', label="Training")
# plt.plot(clf.history['val_accuracy'],'-y', label="Val")
# plt.ylabel(f"Accuracy {dropouts}")
# plt.legend()
# plt.show()

# loss, accuracy = model.evaluate(x,y,verbose=2)
# print(loss, accuracy)
# lossval, accuracyval= model.evaluate(x_val, y_val,verbose=2)
# print(lossval, accuracyval)
# losstest, accuracytest=model.evaluate(x_test,y_test, verbose=2)
# print(losstest, accuracytest)
#sub plot:

# model.save('mask.h5')

# prediction= model.predict(x_test)
# predict_class=np.argmax(prediction,axis = 1) 
# y_label= np.argmax(y_test, axis=1)
# print(confusion_matrix(y_label,predict_class))
# print(classification_report(y_label,predict_class))
# print("f1_score=",f1_score(y_label,predict_class),'\n', "precision=", precision_score(y_label,predict_class),
#     '\n',"recal", recall_score(y_label,predict_class))
