#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()
X = iris.data
y = iris.target
#print(X)
#print(y)
print(iris.feature_names)
print(iris.target_names)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)
print(f'{accuracy:.4f}')
print(y_predict)
new = [[1.2,2.3,4.2,2.3]]
new_predict = knn.predict(new)
print(iris.target_names[new_predict])


# In[30]:


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
#print(X)
#print(y)
print(cancer.feature_names)
print(cancer.target_names)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_predict = knn.predict(X_test)
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)
print(f'{accuracy:.4f}')
print(y_predict)
new = [[17.99, 10.38, 122.80, 1001.00, 0.11840, 0.27760, 0.30010, 0.14710, 0.24190, 0.07871, 1.0950, 0.9053, 8.5890, 153.40, 0.006399, 0.04904, 0.05373, 0.05373, 0.03003, 0.006193, 25.38, 17.33, 184.60, 2019.00, 0.16220, 0.66560, 0.71190, 0.26540, 0.46010, 0.11890]]
new_predict = knn.predict(new)
print(cancer.target_names[new_predict])


# In[ ]:




