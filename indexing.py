#!/usr/bin/env python
# coding: utf-8

# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import matplotlib.pyplot
files = glob.glob("/Users/karthi/Desktop/*.json")
len(files)


# In[31]:


import json


# In[32]:


texts = []
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
    segment = data["segments"]
    #print(segment)
    
    for segments in segment:
        answer = segments["answer"]
        texts.append(answer)
        


# In[33]:


texts


# In[34]:


len(texts)


# In[35]:


corpus = []
for i in range(len(texts)):
    ans = texts[i].split()
    for i in range(len(ans)):
        corpus.append(ans[i])


# In[36]:


corpus


# In[38]:


len(corpus)


# In[39]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
print(X.shape)


# In[40]:



#Vector Space representation
import pandas as pd
vector = X
df1 = pd.DataFrame(vector.toarray(), columns=vectorizer.get_feature_names())
df1


# In[41]:


#loading basic packages
import nltk

nltk.download('punkt')
nltk.download('stopwords')


# In[42]:


from nltk.corpus import stopwords
print(stopwords.words('english'))
stop_words = set(stopwords.words('english'))


# In[43]:


# this function returns a list of tokenized and stemmed words of any text
def get_tokenized_list(doc_text):
    tokens = nltk.word_tokenize(doc_text)
    return tokens

# This function will performing stemming on tokenized words
def word_stemmer(token_list):
  ps = nltk.stem.PorterStemmer()
  stemmed = []
  for words in token_list:
    stemmed.append(ps.stem(words))
  return stemmed


# In[44]:


# Function to remove stopwords from tokenized word list
def remove_stopwords(doc_text):
  cleaned_text = []
  for words in doc_text:
    if words not in stop_words:
      cleaned_text.append(words)
  return cleaned_text


# In[49]:


#Check for single document
tokens = get_tokenized_list(corpus[1])
print("WORD TOKENS:")
print(tokens)
doc_text = remove_stopwords(tokens)
print("\nAFTER REMOVING STOPWORDS:")
print(doc_text)
print("\nAFTER PERFORMING THE WORD STEMMING::")
doc_text = word_stemmer(doc_text)
doc_text


# In[50]:


doc_ = ' '.join(doc_text)
#doc_


# In[51]:


cleaned_corpus = []
for doc in corpus:
  tokens = get_tokenized_list(doc)
  doc_text = remove_stopwords(tokens)
  doc_text  = word_stemmer(doc_text)
  doc_text = ' '.join(doc_text)
  cleaned_corpus.append(doc_text)
cleaned_corpus


# In[52]:


vectorizerX = TfidfVectorizer()
vectorizerX.fit(cleaned_corpus)
doc_vector = vectorizerX.transform(cleaned_corpus)
print(vectorizerX.get_feature_names())

print(doc_vector.shape)


# In[53]:


df1 = pd.DataFrame(doc_vector.toarray(), columns=vectorizerX.get_feature_names())
df1


# In[55]:


query = 'karthi'
query = get_tokenized_list(query)
query = remove_stopwords(query)
q = []
for w in word_stemmer(query):
    q.append(w)
q = ' '.join(q)
q
query_vector = vectorizerX.transform([q])


# In[56]:


# calculate cosine similarities
from sklearn.metrics.pairwise import cosine_similarity
cosineSimilarities = cosine_similarity(doc_vector,query_vector).flatten()


# In[57]:


related_docs_indices = cosineSimilarities.argsort()[:-10:-1]
print(related_docs_indices)

for i in related_docs_indices:
    data = [cleaned_corpus[i]]
    print(data)


# In[58]:


data = {}


# In[ ]:




