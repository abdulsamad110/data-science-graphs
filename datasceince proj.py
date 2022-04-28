import numpy as np
import pandas as pd

# reading data from dataset
dataset = pd.read_csv("C:/Users/walee/Desktop/meta.csv")

from numpy.random import seed
import matplotlib.pyplot as plt
import seaborn as sns
# seed random number generator
seed(1)

# prepare data
data1 = dataset.values[:,2] # colunm 2
data2 = dataset.values[:,3] # column 3

# calculate the Pearson's correlation between two variables

import math

# calculates the mean
def mean(x):
    sum = 0.0
    for i in x:
         sum += i
    return sum / len(x) 

# calculates the sample standard deviation
def sampleStandardDeviation(x):
    sumv = 0.0
    for i in x:
         sumv += (i - mean(x))**2
    return math.sqrt(sumv/(len(x)-1))

# calculates the PCC using both the 2 functions above
def pearson(x,y):
    scorex = []
    scorey = []

    for i in x: 
        scorex.append((i - mean(x))/sampleStandardDeviation(x)) 

    for j in y:
        scorey.append((j - mean(y))/sampleStandardDeviation(y))

# multiplies both lists together into 1 list (hence zip) and sums the whole list   
    return (sum([i*j for i,j in zip(scorex,scorey)]))/(len(x)-1)

corr = pearson(data1, data2)
print('Pearsons correlation: %.3f' % corr)

# Density Histogram
plt.figure(1)
sns.distplot(data1, hist=True, kde=False, 
             bins=int(180/5), color = 'red',
             hist_kws={'edgecolor':'orange'})
sns.distplot(data2, hist=True, kde=False, 
             bins=int(180/5), color = 'green',
             hist_kws={'edgecolor':'black'})   
plt.title('Density Histogram')      
plt.show()


# Bar Plot
plt.figure(2)
y_pos = np.arange(len(data1))
plt.bar(y_pos, data2, align='center', alpha=0.5)
plt.xticks(y_pos, data1)
plt.title('Bar Plot')
plt.show()


# Bubble Chart
plt.figure(3)
size = 500
plt.scatter(data1, data2, s=size)
plt.title('bubble chart')
plt.show()