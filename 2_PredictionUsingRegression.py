import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv(r"data\airfoil_self_noise.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, [-1]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

poly = PolynomialFeatures(4)
poly.fit(X_train)
X_train = poly.transform(X_train)
X_test = poly.transform(X_test) 
reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
print("RMSE: ", mean_squared_error(y_pred, y_test)**0.5)