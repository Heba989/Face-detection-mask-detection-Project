import os
import time
import json
from tensorflow.keras.models import load_model
from flask import Flask,  request
import numpy as np
from datetime import datetime
import sqlite3
import pandas as pd
from PIL import Image, ImageOps
import cv2
import firebase_admin
from firebase_admin import credentials, db


# #----------------------------------------------------------------------------


#class to handle database 
class Db_Handler():
    def __init__(self, Cato,tref): #recieve class of image predicted and time of predictions in local time format
        self.pred = str(Cato)
        self.year= tref[0]
        self.mon = tref[1]
        self.day=tref[2]
        self.hrs = tref[3:6]
        self.flag = 0
        self.conn = sqlite3.connect("Mask.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL")
        self.conn.commit()
    #-------------------------------------------------------------
    #method to save a results in a local datbase (sql database)
    def uploadlocal (self):
        self.cursor.execute('''create table if not exists images (year, mon, day, time, class , flag)''')
        self.cursor.execute(f'''INSERT INTO images (year, mon, day, time, class, flag) VALUES ('{self.year}','{self.mon}', '{self.day}', '{self.hrs}', '{self.pred}', '{self.flag}');''')
        self.conn.commit()
    #----------------------------------------------------
    #method to make a copy on firebase
    def uploadonline (self):
        data = self.cursor.execute('''select * from images where flag = '0';''').fetchall()
        self.conn.commit()
        #firebases accessing
        try:
            cred = credentials.Certificate('C:\\Users\\user\\Desktop\\ICT\\Python\\face-mask-detection-project-firebase-adminsdk-fc688-221b654ce7.json')
            firebase_admin.initialize_app(cred,{'databaseURL':'https://face-mask-detection-project-default-rtdb.firebaseio.com/','timeout':30})
            ref ="/"
            root =db.reference(ref)
            for all in data:
                #we use push to avoid overwrite on database
                root.push({'date':[all[0], all[1], all[2]],'hrs':all[3], "pred":all[4]})
            #update local data base flag that it was upload and change flag to 1
            self.cursor.execute('''UPDATE images SET flag = '1' WHERE flag = '0';''')
            self.conn.commit()
            return "sucess"
        except:
            return "try again"
    def closeconn(self):
        self.conn.close()

    
        


#---------------------------------------------------------------

#preprocessing and prediction part
def preprocessing (input):
      image1=ImageOps.grayscale(input)
      image1.show()
      image1=image1.resize((64,64))
      img1=(np.array(image1)/ 255)
      img1=np.reshape(img1,(1,64,64))
      return img1


#----------------------------------
#APi to recieve image (post image)
app = Flask(__name__)
APP_ROUTE = os.path.dirname(os.path.abspath(__file__))
@app.route("/predict", methods = ['POST'])
def get_image():
    file = request.files["image"]
    #save time of recieve image
    tref = time.localtime()
    #read image using image library
    img = Image.open(file.stream)
    #preprocessing image    
    input=preprocessing(img)
    #NumPy array of image returned ready for prediction
    #------------------------------------------
    #load Model 
    Face = load_model('mask.h5')
    #make prediction
    prediction = Face.predict(input) #it just probabilities, we need class
    predict_class=np.argmax(prediction,axis = 1) #return zero or one 

    #labelling class
    if predict_class == [0]:
        catog = "with"
    elif predict_class == [1]:
        catog = "without"
    #---------------------------
    #saving result on the local database

    #1)define report as database class
    report = Db_Handler(catog,tref)
    #save data in the local database
    report.uploadlocal()
    #make a copy of data in online database (firebase)
    c=report.uploadonline()
    #close connection 
    report.closeconn()
    #return prediction
    return str (catog)
###------------------------------



#Second API for make report and reteive history 
#app = Flask(__name__)
@app.route("/retrieve", methods= ['get','post'])
def mk_rp():
    #connect to SQlite3
    conn = sqlite3.connect("Mask.db")
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cond= request.args.get("time")
    rep = cursor.execute(f'''select * from images where {cond};''').fetchall()
    #convert data to dataframe type from a list of tuples
    df= pd.DataFrame(rep, columns=['year', 'mon', 'day', 'hrs', 'pred' , 'flag'])
    #save data retrieve to a csv file named report
    df.to_csv('report.csv')
    #save change and commit
    conn.commit()
    conn.close()
    return'data saved in rport.csv file'
  
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

