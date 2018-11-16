import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']


dataset = pd.read_csv(url,names=names)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)



y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test,y_pred))


