#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download_shell()


# In[3]:


messages = [line.rstrip() for line in open('SMSSpamCollection')]
print(len(messages))


# In[4]:


for mess_no, messages in enumerate(messages[:10]):
    print(mess_no , messages)
    print('\n')


# In[5]:


import pandas as pd


# In[6]:


messages = pd.read_csv("SMSSpamCollection" , sep = '\t' , names = ['label','message'])


# In[7]:


messages.describe()
messages.groupby("label").describe()


# In[8]:


messages['length'] = messages['message'].apply(len)
messages


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


messages.groupby('length').describe()


# In[11]:


messages['length'].plot.hist(bins = 70)


# In[12]:


messages[messages['length']== 910]['message'].iloc[0]


# In[13]:


messages.hist(column = 'length' , by  = 'label' , bins = 60)


# In[14]:


import string


# In[15]:


mes = 'nihit is cool@ nihit is hot $$.'
nopunc = [c for c in mes if c not in string.punctuation]


# In[16]:


nopunc


# In[17]:


x = ''.join(nopunc)


# In[18]:


from nltk.corpus import stopwords


# In[19]:


stopwords.words("english")


# In[20]:


x


# In[21]:


x.split()


# In[22]:


clean = [words for words in x.split() if words.lower() not in stopwords.words("english") ]


# In[23]:


clean


# In[24]:


# defining for function for completing above tasks
def total_process (mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = "".join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[25]:


messages['message'].head(5)


# In[26]:


# normalization of messages using vectorization
from sklearn.feature_extraction.text import CountVectorizer
#convert string in to marix
bow_tranformer  = CountVectorizer(analyzer= total_process).fit(messages['message'])


# In[27]:


print(len(bow_tranformer.vocabulary_))


# In[28]:


mess4 = messages['message'][3]
mess4


# In[29]:


bow4 = bow_tranformer.transform([mess4])
print(bow4)
print(bow4.shape)


# In[30]:


#to get whcih word is repeated twice etc
bow_tranformer.get_feature_names()[7186]


# In[34]:


#bag fo words to all messages
messages_bow = bow_tranformer.transform(messages['message'])


# In[35]:


print('shape of sparse :' , messages_bow.shape)


# In[38]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))


# In[39]:


print('sparsity: {}'.format(sparsity))


# In[41]:


#inverse frequency and inverse documented frequency
from sklearn.feature_extraction.text import TfidfTransformer


# In[48]:


tfid_transformer = TfidfTransformer().fit(messages_bow)


# In[51]:


tdfidf4 = tfid_transformer.transform(bow4)
print(tdfidf4)


# In[54]:


tfid_transformer.idf_[bow_tranformer.vocabulary_['shall']]


# In[56]:


messages_tfidf = tfid_transformer.transform(messages_bow)


# In[57]:


#for prediction we will use naive based classifier
from sklearn.naive_bayes import MultinomialNB


# In[59]:


spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# In[65]:


spam_detect_model.predict(tdfidf4)[]


# In[68]:


all_pred = spam_detect_model.predict(messages_tfidf)


# In[69]:


all_pred


# In[70]:


# doin ouur train and test split 
from sklearn.model_selection import train_test_split


# In[71]:


msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)


# In[72]:


#doing pipelining by using scikit basically converting into vector than itf all through again but by scikit
from sklearn.pipeline import Pipeline


# In[74]:


pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=total_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[75]:


pipeline.fit(msg_train , label_train)


# In[76]:


predictions = pipeline.predict(msg_test)


# In[77]:


from sklearn.metrics import classification_report
print(classification_report(label_test , predictions))


# In[ ]:




