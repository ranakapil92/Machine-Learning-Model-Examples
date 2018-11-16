import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets


dataset = pd.read_csv('wavesurge.csv')
X = dataset.iloc[1:, 1].values
y = dataset.iloc[1:, 2].values

X=X.reshape(-1,1)
y=y.reshape(-1,1)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


plt.plot(X_train, y_train, '.')
plt.show()


w =1.0
b =0.0
alpha = 0.001
iteration=1000

def gradient_descent(initial_w, initial_b, alpha, iteartions,X_input,y_output):
    b = initial_b
    w=initial_w
    for i in range(0,iteartions):
         
         b_grad=0.0
         w_grad=0.0
         N = float(len(X_input))
         for j in range(0,len(X_input)):
             x=X_input[i]
             y=y_output[i]
             w_grad+=2*x*(y-(w*x+b))*(-1)
             b_grad+=2*x*(y-(w*x+b))*(-1)
            
         w_grad=w_grad/N
         b_grad=b_grad/N
        
         b=b-(alpha*b_grad)
         w=w-(alpha*w_grad)
    return [w,b]

def calculate_error(w,b,X_input,y_output):
    error =0.0
    for i in range(0,len(X_input)):
        error+= (y_output[i] - (w*X_input[i] + b))**2
    return error/float(len(X_input))    

intial_error= calculate_error(w,b,X_train,y_train)
print(intial_error)

[w,b]= gradient_descent(w,b,alpha,iteration,X_train,y_train)

error_after_training= calculate_error(w,b,X_train,y_train)
print(error_after_training)



y_pred = b+np.array(w*np.array(X_test))

plt.plot(y_test, y_pred, '.')
plt.show()
