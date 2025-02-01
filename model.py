import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Dropout
from keras.models import Model

def Inception_Net(num_classes=2):
    # Load InceptionV3 model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(256,256,3))

    input_layer = Input(shape=(256,256,1))
    x = Conv2D(3, (3, 3), padding='same')(input_layer)

    # Add the rest of the model
    x = base_model(x)
    x = GlobalAveragePooling2D()(x) 
    x = Dropout(0.5)(x) 
    output_layer = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model