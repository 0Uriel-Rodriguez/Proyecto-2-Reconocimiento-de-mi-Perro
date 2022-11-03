#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os.path import isfile,isdir, join
import numpy
import datetime
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array
from keras.preprocessing import image
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
from tensorflow.keras.optimizers import SGD

ih, iw = 150, 150 #tamano de la imagen
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales
train_dir = 'perros/Dogs_new/dogos/Train' #directorio de entrenamiento
test_dir = 'perros/Dogs_new/dogos/Test' #directorio de prueba

num_class = 10 #cuantas clases
epochs = 150 #cuantas veces entrenar. En cada epoch hace una mejora en los parametros
batch_size = 2 #batch para hacer cada entrenamiento. Lee 2 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria
num_train = 1400 #numero de imagenes en train
num_test = 230 #numero de imagenes en test

epoch_steps = num_train // batch_size
test_steps = num_test // batch_size

gentrain = ImageDataGenerator(rescale=1. / 255.,
    zoom_range=0.2,
    shear_range=0.4,
    rotation_range = 5,
    vertical_flip=True,
    horizontal_flip=True) 

train = gentrain.flow_from_directory(train_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='categorical')
gentest = ImageDataGenerator(rescale=1. / 255., )
test = gentest.flow_from_directory(test_dir,
                batch_size=batch_size,
                target_size=(iw, ih),
                class_mode='categorical')
model = Sequential()
model.add(Conv2D(30, (3, 3), input_shape=(ih, iw,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(10, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(80))#70
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(10))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])
model.summary()

log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

print("Logs:")
print(log_dir)
print("____")

model.fit(train,steps_per_epoch=epoch_steps,epochs=epochs,validation_data=test,
          validation_steps=test_steps, callbacks=[tbCallBack])
#----------------------------------------------------------------------------------------------------------------
model.save('razasperros.h5')

