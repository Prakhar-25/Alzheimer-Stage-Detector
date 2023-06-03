import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
#from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
# import import_ipynb
# from model_3 import test_set
categories = {0:"Mild Demented",1: "Moderate Demented",2: "Non Demented",3:"Very Mild Demented"}

def predict(img):
    model = load_model('alzheimer2.h5')
    x = img_to_array(img)
    x = np.swapaxes(x,0,1)
    x = x/255
    x = np.expand_dims(x,axis = 0)
    im = preprocess_input(x)
    p = model.predict(im,verbose = 2)
    y_pred = np.argmax(p,axis = 1)
    return y_pred

st.markdown("""# Alzheimer's Stage Detector""")
st.subheader("Welcome to AI based online Alzheimer's stage detector")
st.write("Please upload the MRI scan image below:")
im = st.file_uploader(label = "Image size should be less than 200 MB")
#image = im.getvalue()
if im is not None:
    st.write()
    st.write("Uploaded Image:")
    st.image(im,width = 300)
    if st.button(label="Get Prediction"):
        im = load_img(im)
        index = predict(im).astype(int)
        result = categories.get(index[0])
        st.write("Prediction Result: ")
        st.success(result)