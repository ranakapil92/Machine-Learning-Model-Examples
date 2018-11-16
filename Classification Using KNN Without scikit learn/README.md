About:
1. This is classification problem solved using K-Nearest Neighbors(KNN) Algorithm.

Prerequisites:
1. math
2. numpy
3. matplotlib
4. pandas
5. operator
6. sklearn : Only for train_test_split , for spilling dataset into test and training sets. 



Files:

1. knn.py :  This is primary file defining K-Nearest Neighbors algorithm.

Input : Input is IRIS dataset downloaded from
        https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
        
        It contains 4 features and 1 class ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']. 

Aim of the problem is predict the class using values of 4 features using KNN algorithm.

2. knn_diabetes.py: This is same algorithm with diffrent dataset 'diabetes.csv'. which is downloaded from https://www.kaggle.com/uciml/pima-indians-diabetes-database. It is "Pima Indians Diabetes Database"

3. KNN_IRIS_graph.png: It is graph showing accuracy vs value of K for knn.py code

4. KNN_Diabetes_Graph.png: It is graph showing accuracy vs value of K for knn_diabetes.py code 
