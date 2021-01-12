#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando as bibliotecas

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[2]:


# Pré-processamento do training set

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[3]:


# Pré-processamento do test set

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[4]:


# Criando a CNN

cnn = tf.keras.models.Sequential()

# Adicionando a camada de convolução

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Adicionando a camada de pooling

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Efetuando o flattening

cnn.add(tf.keras.layers.Flatten())

# Camada full conectada 

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Camada de saída

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compilando a CNN

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[5]:


# Treiando a CNN no training set

cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[6]:


# Efetuando uma única previsão

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)

