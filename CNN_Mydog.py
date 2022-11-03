#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import listdir 
from os.path import isfile,isdir, join 
import numpy 
import datetime 
import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model 
from keras.preprocessing import image 
from keras import layers, models 
from keras.models import Sequential 
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten 
from tensorflow.keras.optimizers import SGD 
 
#archivo_lectura = open('razas.hdf5','r') 
#new_model = load_model(archivo_lectura) 
#archivo_lectura.close() 
new_model = keras.models.load_model('razasperros.h5') 
 
ih, iw = 150, 150 #tamano de la imagen 
input_shape = (ih, iw, 3) #forma de la imagen: alto ancho y numero de canales 
train_dir = 'dogggos/Dogos/Train_1' #directorio de entrenamiento 
test_dir = 'dogggos/Dogos/Test_1' 
 
num_class = 2 #cuantas clases 
epochs = 30 #cuantas veces entrenar. En cada epoch hace una mejora en los parametros 
batch_size = 5 #batch para hacer cada entrenamiento. Lee 2 'batch_size' imagenes antes de actualizar los parametros. Las carga a memoria 
num_train = 500 #numero de imagenes en train 
num_test = 100 #numero de imagenes en test 
 
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
                class_mode='binary') 
gentest = ImageDataGenerator(rescale=1. / 255., ) 
test = gentest.flow_from_directory(test_dir, 
                batch_size=batch_size, 
                target_size=(iw, ih), 
                class_mode='binary') 
#Usar modelo pre-entrenado 
model= keras.models.Sequential() 
#model.add(Conv2D(64, kernel_size=3, input_shape=(ih,iw,3))) 
model.add(new_model.layers[0]) 
model.add(new_model.layers[1]) 
model.add(new_model.layers[2]) 
model.add(new_model.layers[3]) 
model.add(new_model.layers[4]) 
model.add(new_model.layers[5]) 
model.summary() 
 
#Congelar capas 
for layer in model.layers[:6]: 
    layer.trainable = False  
    
#Agregar capas densas    
model.add(Flatten()) 
model.add(Dense(30)) 
model.add(Activation('relu', name = "dense_1")) 
model.add(Dropout(0.2)) 
model.add(Dense(1, name = "dense_2")) 
model.add(Activation('sigmoid', name = "dense_3")) 
model.compile(loss='binary_crossentropy', 
              optimizer='SGD', 
              metrics=['accuracy']) 
 
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") 
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True) 
 
print("Logs:") 
print(log_dir) 
print("____") 
 
model.fit(train,steps_per_epoch=epoch_steps,epochs=epochs,validation_data=test, 
          validation_steps=test_steps, callbacks=[tbCallBack]) 
 


# In[12]:


import numpy  
import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model 
from keras.preprocessing import image 
from keras import layers, models 

 
image = load_img('dogggos/Dogos/Test_1/My_dog/WhatsApp Image 2022-10-06 at 11.57.38 AM.jpeg', target_size=(150, 150)) 
img = numpy.array(image) 
img = img / 255.0 
img = img.reshape(1,150,150,3) 
label = model.predict(img) 
print("Predicción (< 0.5 - Mi_Perro , > 0.5 - Otros_Perros): ", label[0][0])


# In[9]:


import numpy  
import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model 
from keras.preprocessing import image 
from keras import layers, models 

 
image = load_img('dogggos/Dogos/Test_1/My_dog/WhatsApp Image 2022-10-08 at 2.07.54 PM (1).jpeg', target_size=(150, 150)) 
img = numpy.array(image) 
img = img / 255.0 
img = img.reshape(1,150,150,3) 
label = model.predict(img) 
print("Predicción (< 0.5 - Mi_Perro , > 0.5 - Otros_Perros): ", label[0][0])


# In[11]:


import numpy  
import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model 
from keras.preprocessing import image 
from keras import layers, models 

 
image = load_img('dogggos/Dogos/Test_1/My_dog/WhatsApp Image 2022-10-08 at 3.11.36 PM (1).jpeg', target_size=(150, 150)) 
img = numpy.array(image) 
img = img / 255.0 
img = img.reshape(1,150,150,3) 
label = model.predict(img) 
print("Predicción (< 0.5 - Mi_Perro , > 0.5 - Otros_Perros): ", label[0][0])


# In[6]:


from os import listdir 
from os.path import isfile,isdir, join 
import numpy  
import datetime 
import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model 
from keras.preprocessing import image 
from keras import layers, models 
from keras.models import Sequential 
from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten 
from tensorflow.keras.optimizers import SGD 
 
image = load_img('dogggos/Dogos/Test_1/Other dogs/n02099601_4678.jpg', target_size=(150, 150)) 
img = numpy.array(image) 
img = img / 255.0 
img = img.reshape(1,150,150,3) 
label = model.predict(img) 
print("Predicción (< 0.5 - Mi_Perro , > 0.5 - Otros_Perros): ", label[0][0])


# In[8]:


import numpy  
import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model 
from keras.preprocessing import image 
from keras import layers, models 
 
image = load_img('dogggos/Dogos/Test_1/Other dogs/n02085620_1620.jpg', target_size=(150, 150)) 
img = numpy.array(image) 
img = img / 255.0 
img = img.reshape(1,150,150,3) 
label = model.predict(img) 
print("Predicción (< 0.5 - Mi_Perro , > 0.5 - Otros_Perros): ", label[0][0])


# In[10]:


import numpy  
import keras 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.utils import load_img, img_to_array 
from keras.models import load_model 
from keras.preprocessing import image 
from keras import layers, models 
 
image = load_img('dogggos/Dogos/Test_1/Other dogs/n02099601_3853.jpg', target_size=(150, 150)) 
img = numpy.array(image) 
img = img / 255.0 
img = img.reshape(1,150,150,3) 
label = model.predict(img) 
print("Predicción (< 0.5 - Mi_Perro , > 0.5 - Otros_Perros): ", label[0][0])

