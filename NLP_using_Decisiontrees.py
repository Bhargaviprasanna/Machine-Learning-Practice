# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:02:16 2020

@author: jyoth
"""
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
corpus.append(review)
# creating bag of words ( creating a sparse matrix of words)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # reduce unnecessary words by using max_length
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1]

# mostly used models are naive bayes, random forests and Decision trees
#naive bayes
# Splitting the dataset into the Training set and Test set
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[:-1,:], y, test_size = 0.25, random_state = 0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = classifier.predict([X_test[2]]) # for a single value
y_prediction = classifier.predict([X[-1]]) # for a new string

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
# accuracy 
# accuracy 
Accuracy = (91+26)/200
Precision = 91/(91+26)
Recall = 91/(91+43)
F1_Score = 2 * Precision * Recall / (Precision + Recall)
"""
Evaluate the performance of each of these models. Try to beat the Accuracy obtained in the tutorial. But remember, Accuracy is not enough, so you should also look at other performance metrics like Precision (measuring exactness), Recall (measuring completeness) and the F1 Score (compromise between Precision and Recall). Please find below these metrics formulas (TP = # True Positives, TN = # True Negatives, FP = # False Positives, FN = # False Negatives):

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Precision = TP / (TP + FP)

Recall = TP / (TP + FN)

F1 Score = 2 * Precision * Recall / (Precision + Recall)


"""
"""
to use a single review to check whether positive or negative , add the review to the dataset  and while training escape it and fit the model then while prediction use that row(review added row)
"""

test_string = 'This is an awful and very bad restaurant!!'
# N = len(test_string)
new_test_string = []
review = re.sub('[^a-zA-Z]',' ', test_string)
review = review.lower()
review = review.split()
ps = PorterStemmer()
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
review = ' '.join(review)
