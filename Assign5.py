#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
import gensim

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from gensim.models import Word2Vec


# In[2]:


sample_text = """Machine Learning is a field of computer science which enables a machine to learn without being explictly programmed.
It focuses on development of algorithms that can analyza and interpret patterns in a data.
It has a variety of applications include spam filtering, speech recognition, and computer vision.
They improves their accuracy when they exposed to more data over time.
Learning from data enables them to predict and make decisions based on previous experiences."""


# In[5]:


# regular expression
sentences = re.sub("[^A-Za-z]+",' ', sample_text)
sentences = re.sub(r"(?:^| )\w(?:$| )", ' ', sentences)
sentences = sentences.lower()


# In[6]:


sentences


# In[7]:


# tokenize
sentences = sent_tokenize(sentences)
all_words = [word_tokenize(sent) for sent in sentences]


# In[9]:


print(all_words)


# In[11]:


# stop words removal
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]


# In[13]:


data = all_words
data1 = sum(data, [])


# In[15]:


print(data1)


# In[17]:


# context target pairs
context_target_pairs=[]
window_size=2

for i in range(window_size, len(data1) - window_size):
    context = [data1[i-2], data1[i-1], data1[i+1], data1[i+2]]
    target_word = data1[i]
    context_target_pairs.append((context, target_word))


# In[18]:


context_target_pairs[:10]


# In[22]:


model = Word2Vec(sentences=data, vector_size=50, window=window_size, min_count=1, sg=0)


# In[25]:


target = "without"
similar_words = model.wv.most_similar(target)

for sim_word, score in similar_words:
    print(f"{sim_word} -> {score:.4f}")


# In[28]:


import numpy as np
def predict(context_words):
    valid_words = [w for w in context_words if w in model.wv]

    context_vectors = np.mean(model.wv[valid_words], axis=0)

    pred_word, score = model.wv.similar_by_vector(context_vectors, topn=1)[0]
    print(f"pred_word: {pred_word} -> score: {score:.4f}")


# In[29]:


predict(['field', 'computer', 'enables', 'machine'])


# In[ ]:




