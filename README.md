# Proyecto 2: Reconocimiento de mi Perro
En este proyecto fueron creadas 2 Redes Convolucionales.

-	La primera: Una red para identificar distintas razas de perros 
-	La segunda: Una red para identificar específicamente a mi perro de otros

A continuación, se dará detalles del proceso de elaboración de cada una de las redes, así como la captura de dichas redes siendo ejecutadas, estas se encontrarán en la carpeta llamada Capturas_CNN.  
# Creación de la red “CNN_DOGS_BREEDS”.
Esta es la primera red para el reconocimiento de diferentes razas de perros.

Primeramente, se decidió usar la base de datos “Stanford Dogs Dataset” que contiene ya clasificadas distintas razas de perros, dejo el enlace de la base de datos por si gustan checarla http://vision.stanford.edu/aditya86/ImageNetDogs/. Esta base de datos contiene más de 20,000 imágenes con 120 categorías (razas de perros). Así que decidí dividir las imágenes de cada categoría en 2 carpetas, (Una para train y la otra para test), cada carpeta tendría la misma cantidad de imágenes, pero obviamente serian estas diferentes. Una vez ya listas nuestras carpetas con los datos procedimos a construir la red.

La estructura de la red fue bastante sencilla, fue una CNN con 120 número de clases, 50 épocas, un batch_size de 10, las imágenes de train eran aproximadamente 10,000 y de la misma manera en test. También se hizo uso de “ImageDataGenerator” para generar más imágenes tanto de test como de train. Así mismo se usó 3 capas Convolucionales y ‘categorical_crossentropy’ dando como resultado una espera de 7 horas para que llegara a un aprendizaje de 0.2 estancándose desde la época 20.


[![Capt.png](https://i.postimg.cc/RFQBB9Nb/Capt.png)](https://postimg.cc/dk1X2bR8)


Se decidió reducir el número de clases a 10 para mejorar el tiempo real de aprendizaje, teniendo poco más de 1600 imágenes, (700 en cada carpeta de test y train), y de esta manera supiera más rápido si se estancaría la red o seguiría aprendiendo la red.


[![Captura0.png](https://i.postimg.cc/SKc06q7N/Captura0.png)](https://postimg.cc/DWfYh9Mt)


Dando como resultado una red que llego a aprender hasta un 0.4, (¡¡¡Ya había mejorado!!!), pero ya no aumentaba el aprendizaje por más que añadiera o quitara capas. así que se decidió manipular los datos nuevamente. 

Esta vez se agregarían más imágenes en train, un total de 1400 imágenes, y se reduciría la cantidad de imágenes en test, un total de 230, y de la misma manera se haría uso de “ImageDataGenerator” para aumentar la cantidad de imágenes. 

De esta manera aumento de golpe el aprendizaje llegando a un aprendizaje de 0.6, que era ya más o menos aceptable, así que se procedió a ir configurando la cantidad de capas convolucionales a solo 2, las capas densas, la cantidad de épocas y el batch_size, dando como resultado final la red “CNN_DOGS_BREEDS.py” que es la que se presenta.

# Creación de la red “CNN_Mydog”.
Esta red es la que identifica específicamente a mi perro.

Para comenzar con la elaboración de esta red primero necesitamos la base de datos.

Esta se dividiría en 2 carpetas, la primera sería una unión de fotos de diferentes imágenes de perros, que ya las tenemos en la base de datos anterior solo fue cuestión de juntar barias en una carpeta a la cual nombramos “Other dogs”,  y la segunda seria las fotos de mi perro, así que se tomaron todas las fotos posibles de mi perro de diferentes ángulos y con diferentes fondos. En total se consiguieron 300 imágenes de mi perro así que tome las mismas de diferentes peros para que nuestro dataset fuera de 600 imágenes. Igualmente usamos “ImageDataGenerator” para aumentar el número de imágenes de manera virtual.


[![Captura.png](https://i.postimg.cc/ZK6DsjvV/Captura.png)](https://postimg.cc/nCckC4kD)


Acontinuacion se muestran algunas de las imagenes que contienen las carpetas de My_dog y Other dogs:


[![Captura7.png](https://i.postimg.cc/VsnRzvsN/Captura7.png)](https://postimg.cc/rzyWJ8r6)


Una vez ya teniendo nuestra base de datos empezamos a construir nuestra red.

Como se nos indicó construimos un modelo con las capas convolucionales del modelo entrenado, pero quitándole el clasificador (capas densas del final), aumentamos un clasificador apropiado para distinguir solo si es mi perro o no (una sola neurona de salida) y congelamos los pesos de la parte pre-entrenada (o primeras capas).
Para poder realizar todo esto primero cargamos el modelo que entrenamos anteriormente, el cual como se ve al final del código de la red “CNN_DOGS_BREEDS.py”  lo guardamos como: ‘razasperros.h5’. Entonces como mencionamos anteriormente cargamos el modelo con keras.models.load_model().

Ahora bien, la estructura de nuestra red será básicamente la misma, pero tendrá los cambios que se nos pidieron, entonces, para empezar el numero de clases cambia a 2, (1 mi perro, 2 otros perros), el batch_size lo bajamos a 5 por la cantidad de imágenes que tenemos y las épocas lo dejamos en 30 por costumbre, al igual que en la red anterior se decidió que seria mayor el número de imágenes en train que en test dejándolo de la siguiente manera: 500 train y 100 en test. Se hizo uso de “ImageDataGenerator” para aumentar la cantidad de imágenes en train como las de test, con el único cambio de que en “class_mode=” se colocó 'binary' en lugar de ‘categorical’.

Ahora para usar el modelo pre-entrenado usamos:
 
 
model= keras.models.Sequential() 

model.add(new_model.layers[0]) 

model.add(new_model.layers[1]) 

model.add(new_model.layers[2]) 

model.add(new_model.layers[3]) 

model.add(new_model.layers[4]) 

model.add(new_model.layers[5]) 


Ya que queremos usar hasta la capa 6 de nuestro modelo pre entrenado y para eso es que es muy útil usar: model.summary() ya que este nos enlista el numero de capas que tenemos como se muestra en la siguiente imagen.


[![Captura4.png](https://i.postimg.cc/cCF6dfrj/Captura4.png)](https://postimg.cc/8frTHJdB)


Y procedemos a congelar hasta la capa 6, que son las anteriormente cargadas, con:

for layer in model.layers[:6]: 

layer.trainable = False

Posteriormente agregamos capas densas y las nombramos para que no nos saltara el error de que el nombre debe ser único. quedándonos de la siguiente manera:

model.add(Flatten()) 

model.add(Dense(30)) 

model.add(Activation('relu', name = "dense_1")) 

model.add(Dropout(0.2)) 

model.add(Dense(1, name = "dense_2")) 

model.add(Activation('sigmoid', name = "dense_3")) 

model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])  
              
              
De la misma manera se usó 'binary_crossentropy' en lugar de 'categorical_crossentropy' ya que queremos que nos de como resultado 0 si es nuestro perro o 1 si es otro perro y esa también es la razón por la cual la ultima capa densa de nuestra red es de 1.

Lo que sigue de la red es completamente lo mismo que vemos en el entrenamiento.

Dándonos unos resultados excelentes, llegando a un aprendizaje del 99%, y ya solo falta poner a prueba nuestra red.

Usando el código “model.predict()” lo que procedemos a hacer es cargar una imagen de nuestro perro o de otro perro, la reescalamos y le decimos que nos imprima si el resultado es  <0.5 es mi perro o si es >0.5 es otro perro esto es para no ser tan exigente con la red pero nos dio unos resultados muy buenos, para mi perro dio resultados de 0.14764893, 0.014218322, 0.16072845 y para otros perros unos resultados de  0.999968, 0.9201807 que se le apegan mucho al 1 y al 0 que queríamos en un principio.

# Conclusiones.
Todo este proyecto me hizo pensar mucho en lo útiles que pueden llegar a ser esta clase de redes, pero también en que la cantidad de imágenes debe ser muy grande si hablamos de muchas clases. Así como que también necesito una maquina mas potente para poder hacer CNN mejores o mas bien para que tarde menos si hablamos del tiempo real. Sin duda alguna esta case de redes abren muchas puertas a la investigación y obviamente pueden llegar a ser muy útiles para los médicos en cuanto al uso de imágenes ya sean de rayos x , tomografías ,etc. Por eso Japón las esta usando de manera indiscriminada. Seria algo interesante el que pudiéramos hacer eso en México, pero sabiendo como están las cosas puede que siga solo siendo un sueño lejano.




