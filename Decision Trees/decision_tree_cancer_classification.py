import numpy as np
from sklearn.tree import DecisionTreeClassifier  # ML algorithm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate  # find best model
from sklearn import datasets  # load iris dataset

cancer_data = datasets.load_breast_cancer()
# print(cancer_data.data) #2D array, and each subarray has something to do with 1 datapoint in the set. 30 values because 30 features
# for eachprint(cancer_data.target) #all 0 and 1, integer represtiation

features = cancer_data.data
labels = cancer_data.target

# print(features.shape) #569 rows (569 samples in dataset), 30 features
# print(labels.shape) #1D array with 569 values

feature_train, feature_test, target_train, target_test = train_test_split(
    features, labels, test_size=0.2)

model_gini = DecisionTreeClassifier(max_depth=3)  # uses gini approach by default
predicted_gini = cross_validate(model_gini, features, labels, cv=10)
print("Gini Accuracy: ",  np.mean(predicted_gini['test_score']))

model_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3)
predicted_entropy = cross_validate(model_entropy, features, labels, cv=10)
print("Entropy Accuracy: ",  np.mean(predicted_entropy['test_score']))

# changing the max depth can change the accuracy, use grid search
