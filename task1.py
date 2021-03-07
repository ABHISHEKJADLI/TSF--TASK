#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('http://bit.ly/w-data')
df.head()
#printing first 5 rows of dataframe


# In[3]:


df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()


# In[4]:


# we can also use .corr to determine the corelation between the variables 
df.corr()


# In[5]:


df.head()


# In[6]:


X = df.iloc[:, :1].values  
y = df.iloc[:, 1:].values


# In[7]:


X


# In[8]:


y


# In[9]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0)


# In[11]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[12]:


lr.fit(X_train,y_train)


# In[13]:


lr.coef_


# In[14]:


lr.intercept_


# In[15]:


# Plotting the regression line
line = lr.coef_*X+lr.intercept_

# Plotting for the test data
plt.show()
plt.scatter(X_train, y_train, color='red')
plt.plot(X, line, color='Blue');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# In[16]:


print(X_test)
y_pred = lr.predict(X_test)


# In[17]:


comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred] })
comp


# In[18]:


hours = 9.25
own_pred = lr.predict([[hours]])
print("Number of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[19]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




