import os
from tensorflow.keras.models import load_model, save_model

model_name = 'mnist_simple_trained_model.h5'

print("test keras")
model = load_model(model_name)
print("test load")
for layer in model.layers:
    print(type(layer))
#model.summary()