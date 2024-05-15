mport numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


df = pd.read_csv('/content/Real-estate-dataset.csv')
X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
Y = df['Y house price of unit area']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 20)
model = LinearRegression().fit(X_train, Y_train)
r_sq = model.score(X,Y)
print(f"coefficient of determination: {r_sq}")
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_}")

print(f"Prediction: {model.predict(X)}")