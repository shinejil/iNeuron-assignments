#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics


# In[3]:


boston = load_boston()


# In[4]:


bos = pd.DataFrame(boston.data, columns=boston.feature_names)
bos['PRICE'] = boston.target


# In[5]:


bos


# In[6]:


#column names of the dataset
boston.feature_names


# In[8]:


bos.head()


# In[9]:


X= bos.drop("PRICE",axis=1)
y= bos["PRICE"]


# In[13]:


#Checking for null values if any
sns.heatmap(bos.isnull())


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[15]:


model=LinearRegression()


# In[16]:


model.fit(X_train,y_train)


# In[20]:


# print the intercept
print(model.intercept_)


# In[21]:


coeff_df = pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[23]:


y_pred = model.predict(X_test)
print(y_pred)


# In[34]:


plt.scatter(y_test,y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()


# In[24]:


df1=pd.DataFrame({'Actual':y_test, 'predicted':y_pred})
df2=df1.head(10)
df2


# In[27]:


print(f'MAE:{metrics.mean_absolute_error(y_test,y_pred)}')
print(f'MSE:{metrics.mean_squared_error(y_test, y_pred)}')
print(f'RMSE:{np.sqrt(metrics.mean_squared_error(y_test, y_pred))}')

