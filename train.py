import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
import cv2
from sklearn.utils import shuffle

from keras.utils import to_categorical
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense, BatchNormalization

#Định danh
target_size = 299
batch_size = 50

# Set dữ liệu train
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5])

# Set dữ liệu validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Lấy data train
train_generator = train_datagen.flow_from_directory(
    'data_process/train',
    target_size=(target_size, target_size),
    batch_size=batch_size,
    color_mode='grayscale',
    class_mode='categorical')

# Lấy data validation
validation_generator = validation_datagen.flow_from_directory(
    'data_process/validation',
    target_size=(target_size, target_size),
    batch_size= batch_size,
    color_mode='grayscale',
    class_mode='categorical')

# Build model
input_tensor = Input(shape=(target_size, target_size, 1))
model = keras.applications.inception_v3.InceptionV3(include_top=True, weights=None, input_tensor=input_tensor, classes=16)

# compile the model 
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Tao thu muc models
path = "models/train_pc"
if not os.path.exists(path):
    os.makedirs(path)

# Save model
with open(path + '/report.txt', 'w') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))

model_json = model.to_json()
with open(path + "/model.json", "w") as json_file:
    json_file.write(model_json)

# Check point
cp_callback = keras.callbacks.ModelCheckpoint(path+"/weights.{epoch:02d}.h5",
                                              save_weights_only=True,
                                              verbose=1)

#Train    
history = model.fit_generator(
                    train_generator,
                    steps_per_epoch= 3000,
                    validation_data= validation_generator,
                    validation_steps= 1000,
                    epochs=50,
                    shuffle=True,
                    callbacks=[cp_callback])


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(path + '/acc.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(path + '/loss.png')
plt.show()
