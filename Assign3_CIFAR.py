#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import cifar10


# In[2]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# In[3]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[4]:


x_train = x_train.astype('float32')/255.0
x_test =  x_test.astype('float32')/255.0


# In[35]:


# build model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


# In[36]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[37]:


his = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=128)


# In[38]:


test_loss, test_accu = model.evaluate(x_test, y_test)


# In[47]:


import matplotlib.pyplot as plt
import random 

idx = random.randint(0, len(x_test)-1)
plt.imshow(x_test[idx])
plt.show()


# In[48]:


class_names=['aeroplane', 'automobile', 'bird', 'cat','dog', 'deer', 'frog', 'horses', 'ship', 'truck']


# In[49]:


pred = model.predict(x_test)
pred_label = np.argmax(pred[idx])
print(class_names[pred_label])


# In[ ]:




