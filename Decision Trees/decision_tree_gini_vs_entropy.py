import numpy as np
from sklearn.tree import DecisionTreeClassifier  # ML algorithm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate  # find best model
from sklearn import datasets  # load iris dataset


iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

feature_train, feature_test, target_train, target_test = train_test_split(
    features, targets, test_size=0.2)

# has several paramerters, define if you eant to use entropy or gini index; default is gini
model_entropy = DecisionTreeClassifier(criterion='entropy')
predicted_entropy = cross_validate(model_entropy, features, targets, cv=10)
print("Entropy Accuracy:", np.mean(predicted_entropy['test_score']))

model_gini = DecisionTreeClassifier(criterion='gini')
predicted_gini = cross_validate(model_gini, features, targets, cv=10)
print("Gini Accuracy:", np.mean(predicted_gini['test_score']))
# notive gini is working a little bit better on this dataset
