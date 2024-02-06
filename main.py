import streamlit as st
import cv2
import numpy as np 
from PIL import Image



# Making Title and header.

st.title('Face Detection Model.')
st.header('Detect face in uploaded image file.')

model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

user_img = st.file_uploader('Upload an Image!',type=['jpg','png','jpeg'])



if user_img is not None:
    with open("Image.jpg","wb") as f:
        f.write(user_img.read())

if user_img:   
    img_arr=cv2.imread("Image.jpg")
    gray_img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    faces = model.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=8)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_arr, (x, y), (x+w, y+h), (0, 0, 256), 5)  
    
    st.image(img_arr, caption='Image with rectangle')
