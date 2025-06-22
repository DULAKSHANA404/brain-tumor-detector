from flask import Flask,render_template,request
import cv2
import numpy as np
from keras.models import load_model

model_path = r"C:\Users\user\Desktop\brain_tumer\CNN\model.keras"

model = load_model(model_path)
result_dict ={0:"normal",1:"Tumer"}

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/result",methods=["POST"])
def result():
    result = request.files["image"]
    image = np.frombuffer(result.read(),np.uint8)
    image = cv2.imdecode(image,cv2.IMREAD_COLOR)
    
    image = cv2.resize(image,(224,224))
    image = image.reshape(1,224,224,3)
    image = np.array(image)
    image = image /255
    
    predict = model.predict(image)
    result = np.argmax(predict,axis=1)[0]
    lable = result_dict[result]
    accurncy = np.max(predict,axis=1)[0]
    accurncy = accurncy*100
    
        
    return render_template("result.html",lable = lable,accurncy=accurncy)