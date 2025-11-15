#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


# In[2]:


(train_img, train_label), (test_img, test_label) = mnist.load_data()


# In[3]:


train_img.shape, train_label.shape, test_img.shape, test_label.shape


# In[6]:


# flattening
x_train = train_img.reshape(train_img.shape[0], -1).astype('float32')
x_test = test_img.reshape(test_img.shape[0], -1).astype('float32')


# In[7]:


# nomralize
x_train /= 255.0
x_test /= 255.0


# In[9]:


# one hot encoding
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score

lb = LabelBinarizer()
y_train = lb.fit_transform(train_label)
y_test = lb.fit_transform(test_label)


# In[10]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[12]:


# model building
model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()


# In[13]:


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


# In[23]:


his = model.fit(x_train, y_train, epochs=10, validation_split=0.2, batch_size=128)


# In[27]:


test_loss, test_accu = model.evaluate(x_test, y_test)


# In[28]:


y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
y_pred, y_true


# In[29]:


accuracy_score(y_pred, y_true)


# In[31]:


his.history.keys()


# In[35]:


import matplotlib.pyplot as plt
plt.plot(his.history['accuracy'], label="accuracy")
plt.plot(his.history['val_accuracy'], label="validation accurcay")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()


# In[38]:


# prediction
import random
idx = random.randint(0,9999)
plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
plt.show()


# In[39]:


pred = model.predict(x_test)
print(np.argmax(pred[idx]))


# In[ ]:




