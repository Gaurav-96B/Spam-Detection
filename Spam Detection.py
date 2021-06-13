#!/usr/bin/env python
# coding: utf-8

# In[119]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# # Importing the dataset

# In[121]:


data=pd.read_csv('SMSSpamCollection',sep='\t',names=["label", "message"])
data


# In[122]:


data.head(10)


# In[123]:


data.tail(10)


# In[124]:


data.shape


# In[125]:


data.isna().sum()


# In[126]:


len(data)


# # Data Cleaning and Preprocessing

# In[127]:


nltk.download('stopwords')
ps = PorterStemmer()
corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['message'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # list comprehension 
    review = ' '.join(review)
    corpus.append(review)


# In[128]:


corpus[5570]


# # Creating the Bag of Words model

# In[129]:


cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values


# In[130]:


len(X)


# In[131]:


len(y)


# In[134]:


X.shape


# # Splitting the dataset into the Training set and Test set

# In[135]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# # Training model using Naive bayes classifier
# 
#  # 1.GaussianNB

# In[138]:


GN_classifier = GaussianNB()
GN_classifier.fit(X_train, y_train)


# In[139]:


GN_score = GN_classifier.score(X_test,y_test)
GN_score


# In[140]:


y_GN_pred = GN_classifier.predict(X_test)
y_GN_pred


# # 2.MultinomialNB

# In[141]:


MN_classifier = MultinomialNB()
MN_classifier.fit(X_train, y_train)


# In[142]:


MN_score = MN_classifier.score(X_test,y_test)
MN_score


# In[143]:


y_MN_pred = MN_classifier.predict(X_test)
y_MN_pred


# # Making the Confusion Matrix

# In[144]:


GN_cm = confusion_matrix(y_test, y_GN_pred)
print(GN_cm)


# In[145]:


MN_cm = confusion_matrix(y_test, y_MN_pred)
print(MN_cm)


# # Compare Both models

# In[146]:


models = pd.DataFrame({"GaussianNB": GN_score,"MultinomialNB": MN_score},index=[0])
models.T.plot.bar(title="Compare different models",legend=False)
plt.xticks(rotation=0);

