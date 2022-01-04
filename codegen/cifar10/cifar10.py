import os,sys
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model, save_model
import numpy as np

PATH = os.path.abspath(os.path.join(os.getcwd(),".."))
sys.path.append(PATH)

from comman import *

model_name = 'cifar10_model.h5'
save_dir = model_name #os.path.join(os.getcwd(), model_name)

def dense_block(x, k):
    x1 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    x2 = concatenate([x, x1],axis=-1)
    x2 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    x3 = concatenate([x, x1, x2],axis=-1)
    x3 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)

    x4 = concatenate([x, x1, x2, x3],axis=-1)
    x4 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x4)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)
    
    x5 = concatenate([x, x1, x2, x3, x4], axis=-1)
    return x5

def train(x_train, y_train, x_test, y_test, batch_size= 64, epochs = 100):

    inputs = Input(shape=x_train.shape[1:])
    x = Conv2D(12, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    # dense block 1
    x = dense_block(x, k=12)

    # bottleneck
    x = Conv2D(36, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # dense block 2
    x = dense_block(x, k=12)

    x = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(10)(x)

    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # save best
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True)

    # free the session to avoid nesting naming while we load the best model after.
    save_model(model, save_dir)
    del model
    tf.keras.backend.clear_session()
    return history

def main(weights='weights.h'):
    epochs = 3 # reduced for CI
    num_classes = 10
    dataset = 'cifar'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # quantize the range to 0~1
    x_test = x_test.astype('float32')/255
    x_train = x_train.astype('float32')/255
    print("data range", x_test.min(), x_test.max())

    # train
    #history = train(x_train,y_train, x_test, y_test, batch_size=128, epochs=epochs)

    model = load_model(save_dir)
    generate_model(model, x_test[:100], name=weights)
    return model,x_train,y_train,x_test,y_test

if __name__ == "__main__":
    main()   