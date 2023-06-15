import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
df = pd.read_csv("/content/Housing.csv")
x = df[["lotsize", "bathrms"]]
y = df.price
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=2)
mlr = LinearRegression()
mlr.fit(x_train, y_train)
print(mlr.coef_)
print(mlr.intercept_)
ypred = mlr.predict(x_test)
r2 = r2_score(y_test, ypred)
print("R^2 score:", r2)
x_new = np.array([[5000,2]])
prediction=mlr.predict(x_new)
print("prediction:", prediction)

