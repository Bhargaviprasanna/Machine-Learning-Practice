# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:50:19 2020

@author: jyoth
"""
# Thompson sampling
""" 
step1: at each round n, we consider two numbers for each ad i :
     N(i,1) - the no of times the ad i got reward 1 upto round n.
     N(i,0) - the no of times the ad i got reward 0 upto round n.

step2: For each ad i, we take a random draw from the distribution below:
     theta(n) = beta(N(i,1)+1, N(i,0)+1)

step3: we select the ad that has the highest theta value

it is better than ucb rewards, in ucb we have to take prev round results at each round , but here there can be delay in that information, here past info is taken batch wise 
"""

# import the packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# implementing the algorithm

d = 10
N = 10000 #no of rows
numbers_of_selected_1 = [0] * d
numbers_of_selected_0 = [0] * d
ads_selected = []
total_rewards = 0
for i in range(0,N):
    ad = 0
    max_random = 0
    for j in range(0,d):
        random_theta = random.betavariate(numbers_of_selected_1[j]+1, numbers_of_selected_0[j]+1)
        if random_theta > max_random:
            max_random = random_theta
            ad = j
        
    rewards = dataset.values[i, ad]
    if rewards == 0 :
        numbers_of_selected_0[ad] = numbers_of_selected_0[ad] + 1
    else:
        numbers_of_selected_1[ad] = numbers_of_selected_1[ad] + 1
    ads_selected.append(ad)
    total_rewards = total_rewards + rewards
print(total_rewards)

#visualising which add is most viewed
plt.hist(ads_selected, color = 'magenta')
plt.title('Different versions of an Ad')
plt.show()