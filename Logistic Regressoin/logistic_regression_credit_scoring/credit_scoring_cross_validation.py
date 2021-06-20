import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

credit_data = pd.read_csv("credit_data.csv")

# this has indices and stuff, needs to be reshaped
features = credit_data[["income", "age", "loan"]]
target = credit_data.default  # this is already a 1D array, doesn't need to be reshaped

# machine learning handle arrays not data-frames so have to reshape
# means Python will figure out number of rows according to last row, and 3 columns
X = np.array(features).reshape(-1, 3)
y = np.array(target)

model = LogisticRegression()  # underlying ML algorithm we are using
# cv indicated number of folds, this predicted variable will be a dictionary
predicted = cross_validate(model, X, y, cv=5)

# test_score is a key in the dictionary, other keys are train_score, fit_time, score_time, estimator
print(predicted['test_score'])
# 5 values for each cv, first iteration, 91.5% is the accuracy of the algorithm, next 91.75%, etc.

print(np.mean(predicted['test_score']))
# take mean of each cv fold
