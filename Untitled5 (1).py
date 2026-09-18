#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['target'] = iris.target
df

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train ,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy) 
print("Accuracy percentage:",accuracy * 100,"%")


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
data = pd.read_csv('food.csv') 
print(data)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X = pd.get_dummies(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train ,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy) 
print("Accuracy percentage:",accuracy * 100,"%")


# In[ ]:




