# -*- coding: utf-8 -*-
"""
Created on Fri May 15 11:32:01 2020

@author: jyoth
"""
# Implementation of Upper Confidence Bound Algorithm
""" step 1: for each round n, find no of times ad got selected and the sum of rewards
    step2: find the average no of times and the bound interval delta i and find the upper_bound value
    step3: calucate the maximum upperbound at each round and ad got selected
"""
#Import the packages
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
#Import the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implemet the algorithm
d = 10
N = 10000 #no of rows
no_of_selections = [0] * d
sums_of_rewards =[0] * d
ads_selected = []
total_rewards = 0
for i in range(0,N):
    ad = 0
    max_ucb = 0
    for j in range(0,d):
        if no_of_selections[j] > 0 :
            aver_reward =  sums_of_rewards[j] / no_of_selections[j] 
            deltai = math.sqrt(3/2 * math.log(i + 1)/no_of_selections[j])
            ucb = aver_reward + deltai
        else: 
            ucb = 1e400
        if ucb > max_ucb:
            max_ucb = ucb
            ad = j
    rewards = dataset.values[i, ad]
    sums_of_rewards[ad]= sums_of_rewards[ad] + rewards
    no_of_selections[ad] = no_of_selections[ad] + 1
    ads_selected.append(ad)
    total_rewards = total_rewards + rewards
print(total_rewards)

#visualising which add is most viewed
plt.hist(ads_selected, color = 'magenta')
plt.title('Different versions of an Ad')
plt.show()