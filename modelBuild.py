# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Tensorflow
import tensorflow as tf
from keras.utils import load_img, img_to_array, to_categorical
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import load_model
from keras import Sequential


# import Dataset
train_df = pd.read_csv('Train.csv')
test_df = pd.read_csv('Test.csv')

print(train_df.head())
print(test_df.head())
print(train_df.info(), '\n', test_df.info())
# Creating labels 
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

print(labels.values())


# getting only the Path and ClassId 
train_df = train_df[['ClassId', 'Path']]
test_df = test_df[['ClassId', 'Path']]

# add the complete path
train_df['Path'] = train_df['Path'].apply(lambda x:''+x)
test_df['Path'] = test_df['Path'].apply(lambda x:''+x)

print(train_df)

# Split the data into training and validation sets
train_data = train_df.sample(frac=0.8, random_state=0)
valid_data = train_df.drop(train_df.index)

# Preprocess the data
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

X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_df)

print(f'The shape of X_train : {X_train.shape}')
print(f'The shape of X_test : {y_train.shape}')


# Define the model architecture

model = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2),
    Conv2D(128, 3, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    MaxPool2D(2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


print(model.summary())



history = model.fit(X_train, 
                    y_train, 
                    epochs=5)



print("history",history)

model.save("model.h5",overwrite=True)

print("model saved")
