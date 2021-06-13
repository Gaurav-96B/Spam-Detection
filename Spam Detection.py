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


# Importing the dataset
data=pd.read_csv('SMSSpamCollection',sep='\t',names=["label", "message"])

# Data Cleaning and Preprocessing
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
    
# Creating the Bag of Words model
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training model using Naive bayes classifier
# 1.GaussianNB
GN_classifier = GaussianNB()
GN_classifier.fit(X_train, y_train)
GN_score = GN_classifier.score(X_test,y_test)
y_GN_pred = GN_classifier.predict(X_test)


# 2.MultinomialNB
MN_classifier = MultinomialNB()
MN_classifier.fit(X_train, y_train)
MN_score = MN_classifier.score(X_test,y_test)
y_MN_pred = MN_classifier.predict(X_test)


# Making the Confusion Matrix
GN_cm = confusion_matrix(y_test, y_GN_pred)
print(GN_cm)

MN_cm = confusion_matrix(y_test, y_MN_pred)
print(MN_cm)


# Compare Both models
models = pd.DataFrame({"GaussianNB": GN_score,"MultinomialNB": MN_score},index=[0])
models.T.plot.bar(title="Compare different models",legend=False)
plt.xticks(rotation=0);

