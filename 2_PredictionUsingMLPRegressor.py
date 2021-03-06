import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, Normalizer

dataset = pd.read_csv(r"data\airfoil_self_noise.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, [-1]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

reg = MLPRegressor(hidden_layer_sizes = (64, 32, 16), activation='relu', solver = 'adam', max_iter = 500)
reg.fit(X_train, y_train.ravel())

y_pred = reg.predict(X_test)
print("RMSE: ", mean_squared_error(y_pred, y_test)**0.5)