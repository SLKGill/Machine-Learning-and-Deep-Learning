import numpy as np
from sklearn.tree import DecisionTreeClassifier  # ML algorithm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate  # find best model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets  # load iris dataset
from sklearn.model_selection import GridSearchCV

iris_data = datasets.load_iris()

features = iris_data.data
targets = iris_data.target

# with grid search you can find an optimal parameter "parameter tuning"

param_grid = {'max_depth': np.arange(1, 10)}
# print(np.arange(1, 10))  # note 10 is not present in the array

feature_train, feature_test, target_train, target_test = train_test_split(
    features, targets, test_size=0.2)

# in ebery iteration we split the data randomly in cross validation + DecisionTreeClassifier
# initializes the tree tandomly (get different results)
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(feature_train, target_train)
print("Best parameter with Grid Search when: ", tree.best_params_)

grid_predictions = tree.predict(feature_test)

print(confusion_matrix(target_test, grid_predictions))
print("Accuracy Score: ", accuracy_score(target_test, grid_predictions))
