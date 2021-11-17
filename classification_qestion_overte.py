#!/usr/bin/env python
# coding: utf-8

# In[40]:


# import des bibliothèques utiles

from gensim.models import Word2Vec
import nltk
from gensim.models import KeyedVectors

from nltk.cluster import KMeansClusterer
import numpy as np 

from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans 
import matplotlib as plt
import pandas as pd
from pandas import DataFrame


# In[41]:


# import des bibliothèques utiles

from gensim.models import Word2Vec
import nltk
from gensim.models import KeyedVectors

from nltk.cluster import KMeansClusterer
import numpy as np 

from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans 
import matplotlib as plt
import pandas as pd
from pandas import DataFrame


# In[42]:


base=open('E://a.txt','r')


# In[43]:


phrases=[]
new_phrases=[]
 


# In[44]:


for phrase in base:
     
    phrases.append(phrase)


# In[45]:


for ligne in phrases:
    ligne=ligne.replace('\t\n','')
     
    new_phrases.append(ligne)


# In[46]:


print(new_phrases )


# In[47]:


fichier_en='D:\\a.bin'


# In[48]:


w2v = KeyedVectors.load_word2vec_format( fichier_en,binary=True)


# In[49]:


import re

def nlp_pipeline(text):

    text = text.lower() # mettre les mots en minuscule

# Retirons les caractères spéciaux :

    text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
    text = re.sub(r"\&\S*\s", "", text)
    text = re.sub(r"\-", "", text)
    
    return text


# In[50]:


# On commence par utiliser notre pipeline définie plus haut

phrases_propres = []

for phrase in new_phrases:
   phrases_propres.append(nlp_pipeline(phrase))

# Nous devons séparer les phrases en liste de mots

phrases_split = []

for phrase in phrases_propres:
   phrases_split.append(phrase.split(" "))

# C'est là que Word2vec intervient
X = []

for phrase in phrases_split:
   vec_phrase = []
   for mot in phrase:
       vec_phrase.append(w2v[mot])
   X.append(np.mean(vec_phrase,0)) 


# In[54]:


model = KMeans(n_clusters=4,random_state=0) 
model.fit(X) 


# In[55]:


predictions = model.predict(X) 


# In[56]:


model.score(X)


# In[57]:


for index, phrase in zip(new_phrases,predictions):    
    print ( index + ":" +   str(phrase) )
    


# In[58]:


labels={}
  
  


# In[59]:


for index, phrase in zip(new_phrases,predictions):
        labels[phrase]=index


# In[60]:



print(labels   )


# In[61]:


import xlsxwriter
classeur=xlsxwriter.Workbook('E://labels.xlsx')
feuille=classeur.add_worksheet() 
row=0
column=0

for index, phrase in zip(new_phrases,predictions):
    feuille.write(row,column,index)
    feuille.write(row, 1,phrase) 
    row +=1
classeur.close()
     


   
    
    
  
    
 


# In[ ]:





# In[199]:





# In[ ]:





# In[ ]:




