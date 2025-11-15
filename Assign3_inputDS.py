#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf


# In[3]:


x_train = np.loadtxt('C:/Users/waghm/Downloads/DL CODES mansi/input.csv', delimiter=',')
y_train = np.loadtxt('C:/Users/waghm/Downloads/DL CODES mansi/labels.csv', delimiter=',')
x_test = np.loadtxt('C:/Users/waghm/Downloads/DL CODES mansi/input_test.csv', delimiter=',')
y_test = np.loadtxt('C:/Users/waghm/Downloads/DL CODES mansi/labels_test.csv', delimiter=',')


# In[5]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[10]:


x_train = x_train.reshape(x_train.shape[0], 100,100,3)
y_train = y_train.reshape(y_train.shape[0], 1)
x_test = x_test.reshape(x_test.shape[0], 100, 100, 3)
y_test = y_test.reshape(y_test.shape[0], 1)


# In[12]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[13]:


x_train /= 255.0
x_test /= 255.0


# In[14]:


# build model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# In[15]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[16]:


his = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)


# In[17]:


test_loss, test_accu = model.evaluate(x_test, y_test)


# In[59]:


import matplotlib.pyplot as plt
import random
idx = random.randint(0, len(x_test)-1)
plt.imshow(x_test[idx, :])
plt.show()


# In[60]:


pred = model.predict(x_test[idx,:].reshape(1,100,100,3))
pred


# In[61]:


if pred > 0.5:
    print('Cat')
else:
    print('Dog')


# In[62]:


plt.plot(his.history['accuracy'], label='accuracy')
plt.plot(his.history['val_accuracy'], label='validation_accuracy')
plt.show()


# In[ ]:




