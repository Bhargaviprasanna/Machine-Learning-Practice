# -*- coding: utf-8 -*-
"""
Created on Sat May  9 21:12:14 2020

@author: jyoth
"""

# Apriori ARL Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transaction = []
for i in range(0,7501):
    transaction.append([str(dataset.values[i, j]) for j in range(20)])

# Training the model with transactions
from apyori import apriori
rules = apriori(transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
# the rules are sorted in apriori function, no need to sort them as in R, they are sorted by support, confidence and lift.
# min_length is 2 becoz atleast 2 items in a set
#Visualize the results
results = list(rules)
results_list = []
for i in range(0, len(results)):
 results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nInfo:\t' + str (results[i][2]))