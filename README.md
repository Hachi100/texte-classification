# texte-classifiications
Classification les qestions ouvertes avec python
L'algo Word2vec
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
base=open('E://a.txt','r')
phrases=[]
new_phrases=[]
 
for phrase in base:
     
    phrases.append(phrase)
for ligne in phrases:
    ligne=ligne.replace('\t\n','')
     
    new_phrases.append(ligne)
 print(new_phrases )
['Allègement fiscal', 'Je ne sais pas', 'Non déclaré', 'Subventions', 'Subventions', 'Je ne sais pas', 'Subventions', 'Subventions', 'Allègement fiscal', 'Non déclaré', 'Non déclaré', 'Non déclaré', 'Non déclaré', 'Allègement fiscal', 'Vaccination', 'Subventions', 'Subventions', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Je ne sais pas', 'Digitalisation', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Dons de kit de protection', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Dons de kit de protection', 'Vaccination', 'Subventions', 'Subventions', 'Sensibilisation', 'Allègement fiscal', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Je ne sais pas', 'Réduction du coût de la vie', 'Non déclaré', 'Vaccination', 'Vaccination', 'Accès aux crédit', 'Subventions', 'Vaccination', 'Non déclaré', 'Subventions', 'Subventions', 'Non déclaré', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Accès aux crédit', 'Subventions', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Subventions', 'Je ne sais pas', 'Je ne sais pas', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Vaccination', 'Dons de kit de protection', 'Non déclaré', 'Dons de kit de protection', 'Je ne sais pas', 'Subventions', 'Subventions', 'Non déclaré', 'Sensibilisation', 'Subventions', 'Allègement fiscal', 'Je ne sais pas', 'Sensibilisation', 'Subventions', 'Subventions', 'Sensibilisation', 'Allègement fiscal', 'Subventions', 'Accès aux crédit', 'Non déclaré', 'Non déclaré', 'Subventions', 'Subventions', 'Réduction du coût de la vie', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Non déclaré', 'Allègement fiscal', 'Vaccination', 'Allègement fiscal', 'accès aux soins', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Accès aux crédit', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Amelioration du climat des affaires', 'Non déclaré', 'Subventions', 'Subventions', 'Je ne sais pas', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Sensibilisation', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Vaccination', 'Je ne sais pas', 'Subventions', 'Je ne sais pas', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Je ne sais pas', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Sensibilisation', 'Sensibilisation', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Je ne sais pas', 'Je ne sais pas', 'Subventions', 'Sensibilisation', 'Subventions', 'Non déclaré', 'Vaccination', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Mesures de securité barrières', 'Je ne sais pas', 'Subventions', 'Vaccination', 'Mesures de securité barrières', 'Subventions', "Création d'emplois", 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Emplois', 'Subventions', 'Je ne sais pas', 'Amelioration du climat des affaires', 'Non déclaré', 'Subventions', 'Allègement fiscal', 'Je ne sais pas', 'Allègement fiscal', 'Je ne sais pas', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Non déclaré', 'Dons de kit de protection', 'Je ne sais pas', "Création d'emplois", 'Non déclaré', 'Non déclaré', 'Subventions', 'Vaccination', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Vaccination', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'accès aux soins', 'Dons de kit de protection', 'Je ne sais pas', 'Accès aux crédit', 'Allègement fiscal', 'faite votre choix', 'Subventions', 'Subventions', 'Je ne sais pas', 'Subventions', 'Subventions', 'Je ne sais pas', 'Allègement fiscal', 'Subventions', 'Subventions', 'Réduction du coût de la vie', 'Non déclaré', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Test de depistage de la virus', 'Subventions', 'Non déclaré', 'Subventions', 'Non déclaré', 'Allègement fiscal', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Vaccination', 'Subventions', 'Subventions', 'Subventions', 'Vaccination', 'Sensibilisation', 'Subventions', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Accès aux crédit', 'Subventions', 'Non déclaré', 'Non déclaré', 'Non déclaré', 'Subventions', 'Non déclaré', 'Subventions', 'Vaccination', 'Subventions', 'Subventions', 'Allègement fiscal', 'Non déclaré', 'Dons de kit de protection', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Vaccination', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Réduction du coût de la vie', 'Allègement fiscal', 'Accès aux crédit', 'Subventions', 'Allègement fiscal', 'Dons de kit de protection', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Réduction du coût de la vie', 'Réduction du coût de la vie', 'Accès aux crédit', 'Réduction du coût de la vie', 'Amelioration du climat des affaires', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Accès aux crédit', 'Subventions', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Sensibilisation', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Vaccination', 'Subventions', 'Subventions', 'Subventions', 'Sensibilisation', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Test de depistage de la virus', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Subventions', 'Accès aux crédit', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Mesures de securité barrières', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Non déclaré', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Accès aux crédit', 'Subventions', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Vaccination', 'Dons de kit de protection', 'Subventions', 'Accès aux crédit', 'Subventions', 'Dons de kit de protection', 'Mesures de securité barrières', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Mesures de securité barrières', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Non déclaré', 'Subventions', 'Réduction du coût de la vie', 'Subventions', 'Subventions', 'Je ne sais pas', 'Amelioration du climat des affaires', 'Vaccination', 'Allègement fiscal', 'Allègement fiscal', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Accès aux crédit', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Subventions', 'Adoptions des mesures coercitives', 'Accès aux crédit', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'accès aux soins', 'Non déclaré', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Vaccination', 'Subventions', 'Allègement fiscal', 'Amelioration du climat des affaires', 'Amelioration du climat des affaires', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Je ne sais pas', 'Accès aux crédit', 'Amelioration du climat des affaires', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Je ne sais pas', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Accès aux crédit', 'Réduction du coût de la vie', 'Je ne sais pas', 'Amelioration du climat des affaires', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Sensibilisation', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Sensibilisation', 'Non déclaré', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Je ne sais pas', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Test de depistage de la virus', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Accès aux crédit', 'Subventions', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Subventions', 'Test de depistage de la virus', 'Je ne sais pas', 'Subventions', 'Je ne sais pas', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Mesures de securité barrières', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Je ne sais pas', 'Subventions', 'Accès aux crédit', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Je ne sais pas', 'Dons de kit de protection', 'Subventions', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Non déclaré', 'Non déclaré', 'Non déclaré', 'Accès aux crédit', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Drapeau MTN', 'Subventions', 'Accès aux crédit', 'Subventions', 'Non déclaré', 'Subventions', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Subventions', 'Non déclaré', 'Non déclaré', 'Amelioration du climat des affaires', 'Subventions', 'Non déclaré', 'Accès aux crédit', 'Subventions', 'Non déclaré', 'Non déclaré', 'Allègement fiscal', 'Accès aux crédit', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Dons de kit de protection', 'Je ne sais pas', 'Subventions', 'Non déclaré', 'Subventions', 'Vaccination', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Je ne sais pas', 'Non déclaré', 'Non déclaré', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Je ne sais pas', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Accès aux crédit', 'Subventions', 'Non déclaré', 'Je ne sais pas', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Vaccination', 'Subventions', 'Non déclaré', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Non déclaré', 'Je ne sais pas', 'Non déclaré', 'Amelioration du climat des affaires', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Sensibilisation', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Je ne sais pas', 'Subventions', 'Vaccination', 'Non déclaré', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Non déclaré', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Accès aux crédit', 'Non déclaré', 'Subventions', 'Subventions', 'Non déclaré', 'Allègement fiscal', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Subventions', 'Dons de kit de protection', 'Subventions', 'Non déclaré', 'suivi des entreprises', 'Subventions', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Subventions', 'Sensibilisation', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Sensibilisation', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Mesures de securité barrières', 'Allègement fiscal', 'Non déclaré', 'Sensibilisation', 'Subventions', 'Allègement fiscal', 'Non déclaré', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Réduction du coût de la vie', 'Non déclaré', 'Allègement fiscal', 'Subventions', 'Subventions', 'Non déclaré', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Allègement fiscal', 'Subventions', 'Subventions', 'Subventions', 'Subventions', 'Je ne sais pas', 'Sensibilisation', 'Subventions', 'Subventions', 'Non déclaré', 'Non déclaré', 'Sensibilisation', 'Subventions', 'Allègement fiscal', 'Subventions', 'Allègement fiscal', 'Allègement fiscal', 'Subventions', 'Subventions', 'Ouverture des frontières', 'Subventions', 'Subventions', 'Je ne sais pas', 'Je ne sais pas', 'Subventions', 'Je ne sais pas', 'Subventions', 'Je ne sais pas', 'Je ne sais pas', 'Allègement fiscal', 'Je ne sais pas', 'Sensibilisation', 'Allègement fiscal', 'Je ne sais pas', 'Je ne sais pas', 'Je ne sais pas', 'Amelioration du climat des affaires', 'Subventions', 'Non déclaré', 'Je ne sais pas']
fichier_en='D:\\a.bin'
w2v = KeyedVectors.load_word2vec_format( fichier_en,binary=True)
import re

def nlp_pipeline(text):

    text = text.lower() # mettre les mots en minuscule

# Retirons les caractères spéciaux :

    text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
    text = re.sub(r"\&\S*\s", "", text)
    text = re.sub(r"\-", "", text)
    
    return text
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
 model = KMeans(n_clusters=4,random_state=0) 
model.fit(X) 
KMeans(n_clusters=4, random_state=0)
predictions = model.predict(X) 
 model.score(X)
-7415.179330113408
for index, phrase in zip(new_phrases,predictions):    
    print ( index + ":" +   str(phrase) )
    
Allègement fiscal:1
Je ne sais pas:2
Non déclaré:2
Subventions:0
Subventions:0
Je ne sais pas:2
Subventions:0
Subventions:0
Allègement fiscal:1
Non déclaré:2
Non déclaré:2
Non déclaré:2
Non déclaré:2
Allègement fiscal:1
Vaccination:3
Subventions:0
Subventions:0
Non déclaré:2
Allègement fiscal:1
Subventions:0
Je ne sais pas:2
Digitalisation:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Dons de kit de protection:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Dons de kit de protection:2
Vaccination:3
Subventions:0
Subventions:0
Sensibilisation:2
Allègement fiscal:1
Non déclaré:2
Allègement fiscal:1
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Je ne sais pas:2
Réduction du coût de la vie:2
Non déclaré:2
Vaccination:3
Vaccination:3
Accès aux crédit:2
Subventions:0
Vaccination:3
Non déclaré:2
Subventions:0
Subventions:0
Non déclaré:2
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Accès aux crédit:2
Subventions:0
Allègement fiscal:1
Non déclaré:2
Subventions:0
Subventions:0
Je ne sais pas:2
Je ne sais pas:2
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Vaccination:3
Dons de kit de protection:2
Non déclaré:2
Dons de kit de protection:2
Je ne sais pas:2
Subventions:0
Subventions:0
Non déclaré:2
Sensibilisation:2
Subventions:0
Allègement fiscal:1
Je ne sais pas:2
Sensibilisation:2
Subventions:0
Subventions:0
Sensibilisation:2
Allègement fiscal:1
Subventions:0
Accès aux crédit:2
Non déclaré:2
Non déclaré:2
Subventions:0
Subventions:0
Réduction du coût de la vie:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Non déclaré:2
Allègement fiscal:1
Vaccination:3
Allègement fiscal:1
accès aux soins:2
Non déclaré:2
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Non déclaré:2
Accès aux crédit:2
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Amelioration du climat des affaires:2
Non déclaré:2
Subventions:0
Subventions:0
Je ne sais pas:2
Allègement fiscal:1
Non déclaré:2
Subventions:0
Sensibilisation:2
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Vaccination:3
Je ne sais pas:2
Subventions:0
Je ne sais pas:2
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Je ne sais pas:2
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Sensibilisation:2
Sensibilisation:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Je ne sais pas:2
Je ne sais pas:2
Subventions:0
Sensibilisation:2
Subventions:0
Non déclaré:2
Vaccination:3
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Mesures de securité barrières:2
Je ne sais pas:2
Subventions:0
Vaccination:3
Mesures de securité barrières:2
Subventions:0
Création d'emplois:2
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Emplois:2
Subventions:0
Je ne sais pas:2
Amelioration du climat des affaires:2
Non déclaré:2
Subventions:0
Allègement fiscal:1
Je ne sais pas:2
Allègement fiscal:1
Je ne sais pas:2
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Subventions:0
Non déclaré:2
Non déclaré:2
Dons de kit de protection:2
Je ne sais pas:2
Création d'emplois:2
Non déclaré:2
Non déclaré:2
Subventions:0
Vaccination:3
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Vaccination:3
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
accès aux soins:2
Dons de kit de protection:2
Je ne sais pas:2
Accès aux crédit:2
Allègement fiscal:1
faite votre choix:2
Subventions:0
Subventions:0
Je ne sais pas:2
Subventions:0
Subventions:0
Je ne sais pas:2
Allègement fiscal:1
Subventions:0
Subventions:0
Réduction du coût de la vie:2
Non déclaré:2
Non déclaré:2
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Test de depistage de la virus:2
Subventions:0
Non déclaré:2
Subventions:0
Non déclaré:2
Allègement fiscal:1
Non déclaré:2
Allègement fiscal:1
Subventions:0
Vaccination:3
Subventions:0
Subventions:0
Subventions:0
Vaccination:3
Sensibilisation:2
Subventions:0
Allègement fiscal:1
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Accès aux crédit:2
Subventions:0
Non déclaré:2
Non déclaré:2
Non déclaré:2
Subventions:0
Non déclaré:2
Subventions:0
Vaccination:3
Subventions:0
Subventions:0
Allègement fiscal:1
Non déclaré:2
Dons de kit de protection:2
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Vaccination:3
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Réduction du coût de la vie:2
Allègement fiscal:1
Accès aux crédit:2
Subventions:0
Allègement fiscal:1
Dons de kit de protection:2
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Réduction du coût de la vie:2
Réduction du coût de la vie:2
Accès aux crédit:2
Réduction du coût de la vie:2
Amelioration du climat des affaires:2
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Accès aux crédit:2
Subventions:0
Allègement fiscal:1
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Sensibilisation:2
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Vaccination:3
Subventions:0
Subventions:0
Subventions:0
Sensibilisation:2
Allègement fiscal:1
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Test de depistage de la virus:2
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Subventions:0
Accès aux crédit:2
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Mesures de securité barrières:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Non déclaré:2
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Accès aux crédit:2
Subventions:0
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Vaccination:3
Dons de kit de protection:2
Subventions:0
Accès aux crédit:2
Subventions:0
Dons de kit de protection:2
Mesures de securité barrières:2
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Mesures de securité barrières:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Non déclaré:2
Subventions:0
Réduction du coût de la vie:2
Subventions:0
Subventions:0
Je ne sais pas:2
Amelioration du climat des affaires:2
Vaccination:3
Allègement fiscal:1
Allègement fiscal:1
Non déclaré:2
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Accès aux crédit:2
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Subventions:0
Adoptions des mesures coercitives:2
Accès aux crédit:2
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
accès aux soins:2
Non déclaré:2
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Non déclaré:2
Allègement fiscal:1
Subventions:0
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Vaccination:3
Subventions:0
Allègement fiscal:1
Amelioration du climat des affaires:2
Amelioration du climat des affaires:2
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Je ne sais pas:2
Accès aux crédit:2
Amelioration du climat des affaires:2
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Je ne sais pas:2
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Accès aux crédit:2
Réduction du coût de la vie:2
Je ne sais pas:2
Amelioration du climat des affaires:2
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Sensibilisation:2
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Sensibilisation:2
Non déclaré:2
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Je ne sais pas:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Test de depistage de la virus:2
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Accès aux crédit:2
Subventions:0
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Subventions:0
Test de depistage de la virus:2
Je ne sais pas:2
Subventions:0
Je ne sais pas:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Mesures de securité barrières:2
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Je ne sais pas:2
Subventions:0
Accès aux crédit:2
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Je ne sais pas:2
Dons de kit de protection:2
Subventions:0
Allègement fiscal:1
Non déclaré:2
Subventions:0
Non déclaré:2
Non déclaré:2
Non déclaré:2
Accès aux crédit:2
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Drapeau MTN:2
Subventions:0
Accès aux crédit:2
Subventions:0
Non déclaré:2
Subventions:0
Allègement fiscal:1
Subventions:0
Non déclaré:2
Subventions:0
Non déclaré:2
Non déclaré:2
Amelioration du climat des affaires:2
Subventions:0
Non déclaré:2
Accès aux crédit:2
Subventions:0
Non déclaré:2
Non déclaré:2
Allègement fiscal:1
Accès aux crédit:2
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Dons de kit de protection:2
Je ne sais pas:2
Subventions:0
Non déclaré:2
Subventions:0
Vaccination:3
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Je ne sais pas:2
Non déclaré:2
Non déclaré:2
Allègement fiscal:1
Non déclaré:2
Subventions:0
Je ne sais pas:2
Allègement fiscal:1
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Accès aux crédit:2
Subventions:0
Non déclaré:2
Je ne sais pas:2
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Vaccination:3
Subventions:0
Non déclaré:2
Non déclaré:2
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Non déclaré:2
Je ne sais pas:2
Non déclaré:2
Amelioration du climat des affaires:2
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Sensibilisation:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Je ne sais pas:2
Subventions:0
Vaccination:3
Non déclaré:2
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Non déclaré:2
Allègement fiscal:1
Non déclaré:2
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Accès aux crédit:2
Non déclaré:2
Subventions:0
Subventions:0
Non déclaré:2
Allègement fiscal:1
Allègement fiscal:1
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Subventions:0
Dons de kit de protection:2
Subventions:0
Non déclaré:2
suivi des entreprises:2
Subventions:0
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Allègement fiscal:1
Subventions:0
Subventions:0
Sensibilisation:2
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Sensibilisation:2
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Mesures de securité barrières:2
Allègement fiscal:1
Non déclaré:2
Sensibilisation:2
Subventions:0
Allègement fiscal:1
Non déclaré:2
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Réduction du coût de la vie:2
Non déclaré:2
Allègement fiscal:1
Subventions:0
Subventions:0
Non déclaré:2
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Allègement fiscal:1
Subventions:0
Subventions:0
Subventions:0
Subventions:0
Je ne sais pas:2
Sensibilisation:2
Subventions:0
Subventions:0
Non déclaré:2
Non déclaré:2
Sensibilisation:2
Subventions:0
Allègement fiscal:1
Subventions:0
Allègement fiscal:1
Allègement fiscal:1
Subventions:0
Subventions:0
Ouverture des frontières:2
Subventions:0
Subventions:0
Je ne sais pas:2
Je ne sais pas:2
Subventions:0
Je ne sais pas:2
Subventions:0
Je ne sais pas:2
Je ne sais pas:2
Allègement fiscal:1
Je ne sais pas:2
Sensibilisation:2
Allègement fiscal:1
Je ne sais pas:2
Je ne sais pas:2
Je ne sais pas:2
Amelioration du climat des affaires:2
Subventions:0
Non déclaré:2
Je ne sais pas:2
 labels={}
   
   
 for index, phrase in zip(new_phrases,predictions):
         labels[phrase]=index
 
print(labels   )
{1: 'Allègement fiscal', 2: 'Je ne sais pas', 0: 'Subventions', 3: 'Vaccination'}
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
     


   
    
    
  
    
