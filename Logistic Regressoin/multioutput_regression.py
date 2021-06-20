from sklearn.datasets import make_regression
# Linear Regression for Multioutput regression
from sklearn.linear_model import LinearRegression
# Decision Tree for Multioutput regression
from sklearn.tree import DecisionTreeRegressor

# create dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5,
                       n_targets=2, random_state=1, noise=0.5)
# n_targets = how many dependent variables I have

print(X.shape, y.shape)  # shape of input and output arrays

# define models
model_LR = LinearRegression()
model_DT = DecisionTreeRegressor()

# fit model
model_LR.fit(X, y)
model_DT.fit(X, y)

# make predictions, 10 features
row = [0.21947749, 0.32948997, 0.81560036, 0.440956, -0.0606303, -
       0.29257894, -0.2820059, -0.00290545, 0.96402263, 0.04992249]
yhat_LR = model_LR.predict([row])
yhat_DT = model_DT.predict([row])


# summarize prediction
print(f"Linear Regression Output: {yhat_LR[0]}")
print(f"Decision Tree Output: {yhat_DT[0]}")
