import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

X=2*np.random.rand(2000,1)
y=6 + 3*X + np.random.rand(2000,1)

plt.plot(X,y,'_')


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)



model.score(X_test, y_test)


y_pred = model.predict(X_test) 
