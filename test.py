import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from sklearn.utils import shuffle
import os
import argparse
import keras

from data import load_samples, test_data_generator


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

batch_size = 32


def main():
    # Define image shape based on input type
    img_size = 256
    batch_size = 32
    test_data_path = "./csv/test.csv"  # Path to the training data

    test_samples = load_samples(test_data_path)

    print(len(test_samples))

    test_generator = test_data_generator(
        test_samples, batch_size, img_size=img_size
    )

    # Load the model
    model = keras.models.load_model("./models/Inception_Net.keras")

    # Evaluate the model
    results = model.evaluate(test_generator, batch_size=batch_size, verbose=1)
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])


if __name__ == '__main__':

    # parser.add_argument('--train_dir', type=str, required=True, help='Directory for training images')

    main()
