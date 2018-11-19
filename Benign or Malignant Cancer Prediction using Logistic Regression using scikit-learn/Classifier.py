import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('wisc_bc_data.csv')

dataset.isnull().sum()

print(dataset['diagnosis'])

S = {'M': 0,'B': 1} 
dataset.diagnosis = [S[item] for item in dataset.diagnosis]


dataset=dataset.drop(['id'], axis=1)


y = dataset.iloc[0:, 0].values
X = dataset.iloc[0:, 1:].values    

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train,y_train)



error=0
pre=0
for i in range(0,len(y_train)):
    pre=clf.predict(X_train[i])
    if(y_train[i]==pre):
        error+=0
    else:
        error+=1
        
print("Accuracy on Training Set",float((len(X_train)-error))/float(len(X_train))*100) 

y_pred=clf.predict(X_test)

error=0
for i in range(0,len(y_test)):
    if(y_test[i]==y_pred[i]):
        error+=0
    else:
        error+=1
 
print("Accuracy on Training Set",float((len(X_train)-error))/float(len(X_train))*100)     
    
       