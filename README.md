# Face-detection-mask-detection-Project
This code is a prerequisite for passing Upskilling program established by HTU university


The data set used in this project is:
https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

The repository include file for downloading and preprocessing the images, file for model and prediction, file for API that 
used the saved model to create a usefl userinterface that recieve image and make prediction and saveed it into local database (SQLlite3)
that is sychonorized with online database (Non-SQl database firebase) anf allow user to create csv report for retrieve historical data ,
and file for api requests.

install data is file for reading training,testing, and validating images from dataset and pre-processing it and saveit as Numpy array

Mask and face detection: is file of model structure, compiling, fitting, and testing.

api images: is file that has two api to read from user an image, load model, make prediction and save results in databases, and api to retrieve data
from databases 

send.py: file that has the api requests command 

