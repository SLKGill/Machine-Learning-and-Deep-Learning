import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

# considering 2 outcomes, 0 or 1. some overlapping between the datasets
# logistic regression will assign a probability of what class a value is assigned to

# x value is small result is 0
x1 = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])
y1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# x value is large result is 1
x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])
y2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# values of x1 and x2, then y1 and y2
X = np.array([[0], [0.6], [1.1], [1.5], [1.8], [2.5], [3], [3.1], [3.9], [4], [4.9], [5], [5.1], [
             3], [3.8], [4.4], [5.2], [5.5], [6.5], [6], [6.1], [6.9], [7], [7.9], [8], [8.1]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

plt.plot(x1, y1, 'ro', color="blue")
plt.plot(x2, y2, 'ro', color="red")
# plt.show()
# see we have 2 classes, blue class = output 0, and red class = ouput 1, 1,
# notice there is overlapping values in the middle. need a sigmoid function


model = LogisticRegression()
model.fit(X, y)

print("b0 is: ", model.intercept_)
print("b0 is: ", model.coef_)
# this is the formula regerssion uses under the hood
# during procedure, gradient descent will find optimal values


def logistic(classifier, x):
    return 1/(1+np.exp(-(model.intercept_ + model.coef_*x)))


for i in range(1, 120):
    plt.plot(i/10.0 - 2, logistic(model, i/10.0), 'ro', color='green')

plt.axis([-2, 10, -0.5, 2])
# plt.show()

pred = model.predict([[1]])  # need 2D array input, will give definitive value, 0 or 1
print("Prediction: ", pred)


pred = model.predict_proba([[1]])  # gives probability belongs to 0 or 1 class [0, 1]
print("Prediction: ", pred)
# 97% chance input value 1 belongs to 0 class, and 3% it belongs to 1
