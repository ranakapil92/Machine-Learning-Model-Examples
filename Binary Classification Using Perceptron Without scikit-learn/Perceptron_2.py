import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('input.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


W=[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
W_0=0.0



# We are considering alpha as equal to 1

for i in range(0,1000):
    for j in range(0,len(X_train)):
        activation=0.0
        activation = np.dot(W,X_train[j])+W_0
        if(activation<0 and y_train[j]==1):
            W=np.add(W,X_train[j])
            W_0+=1
        if(activation>=0 and y_train[j]==0):
            W=np.subtract(W,X_train[j])
            W_0-=1
            
Error=0
for i in range(0,len(X_test)):
    activation= np.dot(W,X_test[i])+W_0
    if(activation<0 and y_test[i]==1):
        Error+=1
    if(activation>=0 and y_test[i]==0):
        Error+=1
        
Accuracy=(2000-Error)/20 
print("Total Accuracy",Accuracy)       
               
           
