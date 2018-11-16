import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import operator


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']	



dataset = pd.read_csv(url,names=names)
X = dataset.iloc[:, :].values

from sklearn.cross_validation import train_test_split
X_train, X_test = train_test_split(X,test_size = 0.25, random_state = 0)

def euclideanDistance(point_a, point_b,dim ):
	dist = 0.0
	for x in range(0,dim):
		dist += (point_a[x] - point_b[x])**2
	return math.sqrt(dist)

#d=euclideanDistance(X_train[0], X_train[1], len(X_train[0])-1)
                    

def K_Neighbors(train, x, k):
    distances = []
    dim=len(x)-1
    for i in range(0,len(train)):
         dist = euclideanDistance(x, train[i], dim)
         distances.append((train[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neigh = []
    for i in range(0,k):
        neigh.append(distances[i][0])
        
    return neigh
 
def get_prediction(neigh):
	class_count = {}
	for i in range(len(neigh)):
		response = neigh[i][-1]
		if response in class_count:
			class_count[response] += 1
		else:
			class_count[response] = 1
	class_count_sort = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
	return class_count_sort[0][0]
 
def Accuracy(Test, predictions):
	correct = 0
	for x in range(len(Test)):
		if Test[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(Test))) * 100.0


k=1
acc=[0.0]*100
for k in range(1,100): 
    predictions=[]
    for x in range(len(X_test)):
        kneighbors = K_Neighbors(X_train, X_test[x], k)
        result = get_prediction(kneighbors)
        predictions.append(result)
    acc[k]=Accuracy(X_test, predictions)
print(acc)


x = np.linspace(1, 100, num=100)
plt.xlabel("Value of K")
plt.ylabel("Total Accuracy")
plt.plot(x, acc)
plt.show()


