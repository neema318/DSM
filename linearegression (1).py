#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data = load_iris()
X= data.data[:, 2].reshape(-1, 1)
y= data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
print(f'Linear Regression MSE: {lr_mse}')
r2_score = r2_score(y_test, lr_predictions)
print('R2 score:',r2_score)
new_pred = [[5.1]]
prediction = lr_model.predict(new_pred)
predicted_class = round(prediction[0])
print(f'Predicted:{data.target_names[predicted_class]}')


# In[3]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
data = load_iris()
X= data.data
y= data.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
print("MSE:",mse)
print("R2 Score:", r2)
new_pred = [[5.1,3.5,1.4,0.2]]
predict = model.predict(new_pred)
print("Predicted:",data.target_names[round(predict[0])])


# In[ ]:




