
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import pyttsx3
# Tensorflow
import tensorflow as tf
from keras.utils import load_img, img_to_array, to_categorical
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import load_model
from keras import Sequential

from tkinter import *
from tkinter import filedialog
import tkinter as tk
from PIL import Image,ImageTk
import os


labels = {0:'Speed limit(20km/h)',
              1:'Speed limit(30km/h)',
              2:'Speed limit(50km/h)',
              3:'Speed limit(60km/h)',
              4:'Speed limit(70km/h)',
              5:'Speed limit(80km/h)',
              6:'End of speed limit(80km/h)',
              7:'Speed limit(100km/h)',
              8:'Speed limit(120km/h)',
              9:'No passing',
              10:'No passing for vehicles over 3.5 metric tons',
              11:'Right-of-way at the next intersection',
              12:'Priority road',
              13:'Yield',
              14:'Stop',
              15:'No vehicles',
              16:'Vehicles over 3.5 metric tons prohibted',
              17:'No entry',
              18:'General caution',
              19:'Dangerous curve to the left',
              20:'Dangerous curve to the right',
              21:'Double curve',
              22:'Bumpy road',
              23:'Slippery road',
              24:'Road narrows on the right',
              25:'Road work',
              26:'Traffic signals',
              27:'Pedestrians',
              28:'Children crossing',
              29:'Bicycles crossing',
              30:'Beware of ice/snow',
              31:'Wild animals crossing',
              32:'End of all speed and passing limits',
              33:'Turn right ahead',
              34:'Turn left ahead',
              35:'Ahead only',
              36:'Go straight or right',
              37:'Go straight or left',
              38:'Keep right',
              39:'Keep left',
              40:'Roundabout mandatory',
              41:'End of no passing',
              42:'End of no passing by vehicles over 3.5 metric tons'}


def speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def loadModel():
    # import Dataset
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('Test.csv')

    #print(train_df.head())
    #print(test_df.head())
    #print(train_df.info(), '\n', test_df.info())
    # Creating labels 
    

    IMG_SIZE = 32
    NUM_CLASSES = len(labels)
    def preprocess_data(df):
        X = []
        y = []
        for index, row in df.iterrows():
            img = load_img(row.Path, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(to_categorical(row.ClassId, num_classes=NUM_CLASSES))
        return np.array(X), np.array(y)


    model=load_model("model.h5")
    return model

def output(model,path):
    
    img=load_img(path,target_size=(32,32))

    i= img_to_array(img)
    i=preprocess_input(i)
    input_arr=np.array([i])

    # Get the predicted labels for the test data
    y_pred = model.predict(input_arr)

    #print(y_pred)

    print("\n")
    print(np.argmax(y_pred))
    pred_label = np.argmax(y_pred)
    print(labels[pred_label])

    speech(labels[pred_label])

def predict(path):
    model=loadModel()

    output(model,path)


def image():
    filename=filedialog.askopenfilename(initialdir="C:/Users/SIVAKUMAR/Downloads",title="Select a image",filetypes=(("png files","*.png"),("jpg files","*.jpg"),("all files","*.*")))
    img=Image.open(filename)
    img=ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image=img
    predict(filename)

root=Tk()

frame=Frame(root)
frame.pack(side=BOTTOM,padx=15,pady=15)

lbl=Label(root)
lbl.pack()

btn=Button(frame,text="Select Image",command=image)
btn.pack(side=tk.LEFT)

btn2=Button(frame,text="Exit",command=lambda:exit())
btn2.pack(side=tk.LEFT,padx=10)

root.title("test")
root.geometry("500x300")
root.mainloop()