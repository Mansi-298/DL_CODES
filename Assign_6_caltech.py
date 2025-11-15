#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import tensorflow as tf
import os


# In[16]:


raw_data_dir=r"C:\Users\waghm\Downloads\caltech-101\101_ObjectCategories"
base_dir=r"C:\Users\waghm\Downloads\data_dir"
os.makedirs(base_dir, exist_ok=True)


# In[17]:


for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(base_dir, split), exist_ok=True)


# In[18]:


os.listdir(base_dir)


# In[19]:


train_ratio, valid_ratio, test_ratio = 0.7, 0.15, 0.15


# In[20]:


classes = [c for c in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, c))]


# In[21]:


print(classes)


# In[29]:


import os
import shutil
for cls in classes:
    cls_path = os.path.join(raw_data_dir, cls)
    images = [i for i in os.listdir(cls_path) if i.lower().endswith(('jpg', 'jpeg', 'png'))]

    n_total = len(images) 
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)

    train_files = images[:n_train] 
    valid_files = images[n_valid:n_train + n_valid]
    test_files = images[n_train+n_valid:]

    for split, split_files in zip(['train', 'valid', 'test'], [train_files, valid_files, test_files]):
        split_path = os.path.join(base_dir, split, cls)
        os.makedirs(split_path, exist_ok=True)
        for img in split_files:
            shutil.copy2(os.path.join(cls_path, img), os.path.join(split_path, img))

print("Datasets splits success")


# In[35]:


for split in ['train', 'valid', 'test']:
    split_path = os.path.join(base_dir, split)
    print(f"{split.upper()} -> {len(os.listdir(split_path))}")


# In[43]:


from tensorflow import keras
train_ds = keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, 'train'),
    labels='inferred',
    label_mode='categorical',
    image_size = (224,224),
    batch_size=32,
    shuffle=True
)


# In[44]:


valid_ds = keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, 'valid'),
    labels='inferred',
    label_mode='categorical',
    image_size = (224,224),
    batch_size=32,
    shuffle=True
)


# In[45]:


test_ds = keras.utils.image_dataset_from_directory(
    os.path.join(base_dir, 'test'),
    labels='inferred',
    label_mode='categorical',
    image_size = (224,224),
    batch_size=32,
    shuffle=True
)


# In[46]:


from tensorflow.keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
conv_base.trainable = False


# In[57]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Input, Dense, Rescaling


# In[58]:


n_classes = len(train_ds)


# In[60]:


model = Sequential([
    conv_base,
    Rescaling(1./255, input_shape=(224, 224, 3)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(n_classes, activation='softmax')
])


# In[61]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[62]:


model.summary()


# In[63]:


his = model.fit(train_ds, validation_data=valid_ds, epochs=1)


# In[ ]:





# In[ ]:




