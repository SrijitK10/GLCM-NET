import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
import os
import argparse

from data import load_samples, train_data_generator, validation_data_generator
from model import Inception_Net
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


batch_size = 32
learning_rate = 0.01
epochs = 30
num_classes = 2
img_shape = (256, 256, 1)

def main():

    # Load and preprocess the data
    train_data_path = './csv/train.csv'     # Path to the training data
    validation_data_path='./csv/validation.csv' # Path to the validation data


    train_samples = load_samples(train_data_path)
    validation_samples = load_samples(validation_data_path)



    print(len(train_samples))
    print(len(validation_samples))



    train_generator=train_data_generator(train_samples,batch_size,img_size=256) 
    validation_generator = validation_data_generator(validation_samples,batch_size,img_size=256)

    # Create and compile the model
    model = Inception_Net(in_shape=img_shape, num_classes=num_classes)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model_filepath = './models/Inception_Net.keras'  # Path to save the model for grayscale images
    

    checkpoint=ModelCheckpoint(model_filepath,monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')

    model.fit(train_generator, 
                    epochs=epochs, 
                    steps_per_epoch=len(train_samples)//batch_size, 
                    validation_data=validation_generator, 
                    validation_steps=len(validation_samples)//batch_size, 
                    callbacks=[checkpoint])




if __name__ == '__main__':
    main()
