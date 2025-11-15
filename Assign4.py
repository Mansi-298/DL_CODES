#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('C:/Users/waghm/Downloads/DL CODES mansi/creditcard.csv')


# In[3]:


df.head()


# In[4]:


df = df.drop(['Time'], axis=1)


# In[6]:


df['Amount'] = StandardScaler().fit_transform(df[['Amount']])


# In[7]:


df.head()


# In[8]:


x = df.drop(['Class'], axis=1)
y = df['Class']


# In[9]:


x


# In[10]:


y


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[12]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[13]:


x_train_normal = x_train[y_train==0]
x_test_normal = x_test[y_test==0]
x_test_fraud = x_test[y_test==1]


# In[14]:


x_train_normal.shape


# In[17]:


# build encoder
from tensorflow.keras.layers import Input, Dense

input_dim = x_train_normal.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(14, activation='relu')(input_layer)
decoder = Dense(input_dim, activation='relu')(encoder)


# In[19]:


from tensorflow.keras.models import Model

model = Model(inputs= input_layer,
             outputs= decoder)

model.summary()


# In[20]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[21]:


his = model.fit(x_train_normal, x_train_normal, validation_data=(x_test_normal, x_test_normal), batch_size=32, epochs=10)


# In[22]:


# determine threshold
pred = model.predict(x_test_normal)
mse = np.mean(np.power(pred - x_test_normal,2), axis=1)
threshold = np.percentile(mse, 95)


# In[23]:


# get y_pred
pred1 = model.predict(x_test)
mse = np.mean(np.power(pred1 - x_test,2), axis=1)
y_pred = (mse > threshold).astype(int)


# In[35]:


import random
idx = random.randint(0, len(x_test)-1)
trans = x_test.iloc[idx:idx+1]
actual_class = y_test.iloc[idx]

pred = model.predict(trans)
error = np.mean(np.power(trans.values - x_test, 2))
pred_class = 1 if error > threshold else 0


# In[36]:


print(actual_class)
print(error)
print(pred_class)


# In[37]:


import random
idx = random.randint(0, len(x_test_fraud)-1)
trans = x_test_fraud.iloc[idx:idx+1]
actual_class = 1

pred = model.predict(trans)
error = np.mean(np.power(trans.values - x_test, 2))
pred_class = 1 if error > threshold else 0


# In[38]:


print(actual_class)
print(error)
print(pred_class)


# In[ ]:




