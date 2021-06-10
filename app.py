import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from flask import Flask , request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = load_model("mymodel.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        #print("current path")
        basepath = os.path.abspath('')
        #print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (64,64))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        preds = model.predict_classes(x)
        print("prediction",preds)
            
        index=['Left Bundle Branch Block','Normal','Premature Atrial Contraction','Premature Ventricular Contractions','Right Bundle Branch Block','Ventricular Fibrillation']
        
        text = "the predicted arrhythmia  is : " + str(index[preds[0]])
        
    return text
if __name__ == '__main__':
    app.run(debug = False,threaded=False)