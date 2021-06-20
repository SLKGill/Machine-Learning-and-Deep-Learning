# multinomial logistic regression
# inputs: clientID, income, age, loan = features (input parameters), these are numbers
# output: default (target variable we are going to predict),  2  values (0-can payback loan or 1-failed to payback loan)

import pandas as pd  # get csv file
from sklearn.linear_model import LogisticRegression  # machine learning approach
from sklearn.model_selection import train_test_split  # split data into training and testing
from sklearn.metrics import confusion_matrix, accuracy_score  # analyze results

credit_data = pd.read_csv("credit_data.csv")


# print(credit_data.head()) #fetch data, print first 5 lines
# print(credit_data.describe()) #gives general information, how many items (2000), mean of features, sd, min/max
# print(credit_data.corr())  #symmetric matrix, diagonal values are 1 and off diagonal are the correlation between the given random variables

# input parameters (multinomial), we will have a b0, b1 x1=age, b2 x2=income, b3 x3=amount of loan
features = credit_data[["income", "age", "loan"]]
target = credit_data.default  # taget values

# split features and targets
# 70% is for Training (finding the b values)
# 30% of dataset is for testing (test whether logistic regerssion model is working or not with the model)
feature_train, feature_test, target_train, target_test = train_test_split(
    features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(feature_train, target_train)

predictions = model.fit.predict(feature_test)

print(confusion_matrix(target_test, predictions))  # target_test are the certain true results
# 503 items classified right in terms of 0
# 39 correct classified items for 1

print(accuracy_score(target_test, predictions))
# 94% probability, logistic model predicted the output correctly
# can calculate this probability from the confusion matrix (509+56)/(509+56+15+20)

# print(model.fit.coef_) #these will give b1 b2 b3 values
# print(model.fit.intercept_) #b0 value

#income, age, loan
print("Prediction: ", model.predict([[66155.9251, 59.01701507, 8106.532131]]))


# NOTE: everytime run confusion matrix can be slightly different
