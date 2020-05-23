# -*- coding: utf-8 -*-
"""
In NLP , we use tsv file(tab seperated value) because the model is about telling a review is either positive or negative, so a tab seperated valueis appropriate delimiter . as no review contain tab in it, comma can be seen in the text, so taking comma as delimiter is a bad idea
"""
#importing the packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk # for stop words
import re # regular expression package to do operations on the reviews
#import the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the text

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# quoting is to remove double quotes
# download stopwords such as 'this' 
#perform stemming , the process is that it gives root values, loved is stemmed to love
# remove unecessary words
 #making all this as a list
N = 1000
corpus = []
for i in range(0,N):
    # review = dataset['Review'][i]
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# creating bag of words ( creating a sparse matrix of words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # reduce unnecessary words by using max_length
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1]

# mostly used models are naive bayes, random forests and Decision trees
#naive bayes
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)



# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy = (55+91)/200
Precision = 55/(55+42)
Recall = 55/(55+12)
F1_Score = 2 * Precision * Recall / (Precision + Recall)