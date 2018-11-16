import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


diabetes = datasets.load_diabetes()

diabetes.data.shape

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

model.score(X_test, y_test)


y_pred = model.predict(X_test) 
plt.plot(y_test, y_pred, '.')
x = np.linspace(0, 330, 50)
y = x
plt.plot(x, y)
plt.show()
