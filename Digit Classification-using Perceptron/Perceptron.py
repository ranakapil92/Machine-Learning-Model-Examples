import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('mnist_test.csv')
X_test = dataset.iloc[:, 1:].values
y_test = dataset.iloc[:, 0].values

dataset = pd.read_csv('mnist_train.csv')
X_train = dataset.iloc[:, 1:].values
y_train = dataset.iloc[:, 0].values


from sklearn.linear_model import Perceptron


clf=Perceptron()
clf.fit(X_train,y_train)


clf.score(X_train,y_train)

y_pred = clf.predict(X_test) 
plt.plot(y_test, y_pred, '.')
x = np.linspace(0, 330, 50)
y = x
plt.plot(x, y)
plt.show()

error=0
for i in range(0,len(y_pred)):
    if(y_pred[i]!=y_test[i]):
        error+=1
 
print("Total Incorrect Prediction",error)       

