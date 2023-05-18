import streamlit as st
import tensorflow as tf
from tensorflow import expand_dims, newaxis
import cv2
from PIL import Image, ImageOps
import numpy as np
import os
path = os.path.dirname(__file__)

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(os.path.join(path,"Modeleye1.h5"))
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
 
st.write("""
         # Glaucoma/Normal Eye Image Classification
         """
         )
file = st.file_uploader("Upload the image to be classified", type=["jpg", "png"])
st.set_option('deprecation.showfileUploaderEncoding', False)
 
def upload_predict(upload_image, model):
        labels = ['glaucoma','normal']
        size = (224,224)
        img = upload_image.resize(size, Image.Resampling.LANCZOS) #resizes to 224,224
        img_array = np.array(img)
        img_array = expand_dims(img_array, 0)

        predictions = model.predict(img_array)

        predicted_class = labels[np.argmax(predictions[0])]
        confidence = round( (np.max(predictions[0])), 2)
    
        return predicted_class, confidence
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    image_class,score = upload_predict(image, model)
    st.write("The image is classified as",image_class)
    st.write("The similarity score is approximately",score)
    print("The image is classified as ",image_class, "with a similarity score of",score)

