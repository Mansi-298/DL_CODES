#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


# In[7]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[8]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[10]:


x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)


# In[11]:


# base model
conv_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
conv_model.trainable=False


# In[17]:


# add custom classifier
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
model = Sequential([
    conv_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[18]:


# Step (d): Compile and Train model
# ----------------------------
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Display model summary
model.summary()


# In[20]:


his = model.fit(x_train, y_train, epochs=2, batch_size=64, validation_data=(x_test, y_test))


# In[22]:


test_loss, test_accu = model.evaluate(x_test, y_test)


# In[26]:


import matplotlib.pyplot as plt
indices = np.random.choice(len(x_test), 5)
sample_images = x_test[indices]
sample_labels = np.argmax(y_test[indices], axis=1)

pred = model.predict(sample_images)
pred_labels = np.argmax(pred, axis=1)

class_names = ['aeroplane', 'automobile', 'bird', 'cat', 'dog', 'deer', 'frog', 'horses', 'ship', 'truck']

plt.figure(figsize=(10,5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow((sample_images[i]+127.5).astype('uint8'))
    plt.title(f"Pred: {class_names[pred_labels[i]]} \n True: {class_names[sample_labels[i]]}")
    plt.axis('off')
plt.show()


# In[ ]:




