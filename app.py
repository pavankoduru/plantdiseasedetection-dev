from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from tensorflow import keras

from tensorflow.keras.preprocessing.image import img_to_array

import pickle 
import cv2
import os

DEFAULT_IMAGE_SIZE = tuple((256, 256))
model=tf.keras.models.load_model('models\plant_disease_model.h5')
filename = 'models\plant_disease_label_transform.pkl'
image_labels = pickle.load(open(filename, 'rb'))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


app=Flask(__name__)
#UPLOAD_FOLDER='/UPLOAD'
#app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route("/")
def func():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    # print(request.files)
    if 'image_file' not in request.files:
        return "no image"
    else:
        image_file=request.files['image_file']
        path=os.path.join(image_file.filename)
        image_file.save(path)
        image_array = convert_image_to_array(path)
        np_image = np.array(image_array, dtype=np.float16) / 225.0
        np_image = np.expand_dims(np_image,0)
        #print(np_image)
        model.summary()
        result = model.predict_classes(np_image)
        #print(result)
        x=image_labels.classes_[result][0]
        #print(x)


    return render_template('result.html',name=x)

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5500,debug =True)
