from matplotlib import test
from rest_framework.views import APIView
# from .models import *
from rest_framework.response import Response
from PIL import Image
import pickle
from tensorflow import keras
model = keras.models.load_model('my_model.h5',compile=False)

import numpy as np

arr = np.random.rand(256, 256) * 255
arr = arr.astype(np.uint8)
img = Image.fromarray(arr)

pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model_potato = pickle.load(open('potatomodel.pkl', 'rb'))
pickled_model_bell = pickle.load(open('bellmodel.pkl', 'rb'))
# cnn = pickle.load(open('CNNALLPLNT94.pkl','rb'))
cnn_diseases=['Bacterial spot',
 'healthy',
 'Early blight',
 'Late blight',
 'healthy',
 'Bacterial spot',
 'Early blight',
 'Late blight',
 'healthy']

class GetTomato (APIView):
    def post (self,request) :
        output = {}
       

        img_data  = request.data['imagefile']
        plant_name = request.data['plantname']
        cnnimg = request.data['CNNImg']
        print(len(img_data),plant_name,"TANMAY")
        # op = cnn.predict([img_data])
   
        # arr = np.random.rand(256, 256) * 255
        # arr = cnnimg.astype(np.unt8)
        # f = open("output2.txt",'a')
        # f.write(str(cn))
        # f.close()
     
        prediction = cnn_diseases[np.argmax(model.predict([cnnimg])[0])]

        if prediction != "healthy" :
            prediction = "Suffering  with" +prediction
        
        output["result"] = prediction
        output["status"] = prediction

        # if plant_name == "tomato" :
        #     problems = [" Healthy", " Suffering with  Bacterial Spot"," Suffering with Early Blight"," Suffering with  Late Blight"]
        #     output["result"] = problems[(pickled_model.predict([img_data])[0])-1]
        #     output['status']="Success"
        # elif plant_name == "potato" :
        #     problems = [" Healthy", " Suffering with  Bacterial Spot"," Suffering with Early Blight"," Suffering with  Late Blight"]
        #     output["result"] = problems[(pickled_model_potato.predict([img_data])[0])-1]
        #     output['status']="Success"
        # elif plant_name == "bellpepper" :
        #     problems = [" Healthy", " Suffering with  Bacterial Spot"," none","none"]
        #     output["result"] = problems[(pickled_model_bell.predict([img_data])[0])-1]
        #     output['status']="Success"
            
        # f = open("output1.txt",'a')
        # f.write(str(img_data))
        # f.close()
        return Response(output)

