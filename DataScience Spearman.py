import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# reading data from dataset
dataset = pd.read_csv("G:/dataset.csv")


#spearman corelation for 2 attributes

data1 = dataset.values[:,6] # colunm 6
data2 = dataset.values[:,3] # column 3

# calculate spearman's correlation
corr, _ = spearmanr(data1, data2)
print('Spearmans correlation: %.3f' % corr)

#Applying KNN



X = dataset.drop(columns=['abc124'])

         
#separate target values
y = dataset['abc124'].values
#view target values
y[0:5]


#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=15)
classifier.fit(X_train.astype(int), y_train.astype(int))

y_pred = classifier.predict(X_test)


#confusion matrixxxx

print("confusion matrixxx----------->",confusion_matrix(y_test.astype(int), y_pred.astype(int)))
#classification report

print("class report",classification_report(y_test.astype(int), y_pred.astype(int)))


#accuracy count--

print("Accuracy before validation:",metrics.accuracy_score(y_test.astype(int), y_pred.astype(int)))

#cross-10 validation



scores = cross_val_score(classifier, X_train.astype(int), y_train.astype(int), cv=10, scoring='accuracy')
print("cross val results on training data",scores)


k_range = range(1, 31)
# empty list to store scores
k_scores = []

# 1. we will loop through reasonable values of k
for k in k_range:
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(classifier, X_train.astype(int), y_train.astype(int), scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())


print(k_scores)

print('Length of list', len(k_scores))
print('Max of list', max(k_scores))

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
print("accuracy after validation",scores.mean())

#check for future KNN errors
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train.astype(int), y_train.astype(int))
    pred_i = knn.predict(X_test.astype(int))
    error.append(np.mean(pred_i != y_test))
    

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')