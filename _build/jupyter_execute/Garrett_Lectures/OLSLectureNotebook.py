#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm


# # Omitted Variable Bias #

# In[2]:


shark = pd.read_csv('http://uclspp.github.io/PUBLG100/data/shark_attacks.csv')
sns.scatterplot(shark['IceCreamSales'], shark['SharkAttacks'])
x1 = shark['IceCreamSales']
y1 = shark['SharkAttacks']
m1, b1 = np.polyfit(x1, y1, 1)
plt.plot(x1, m1*x1+b1, color = 'red')


# In[3]:


X = sm.add_constant(shark[['IceCreamSales']])
y = shark['SharkAttacks']
model = sm.OLS(y, X).fit()
model.summary()


# In[4]:


shark.corr()


# In[5]:


X = sm.add_constant(shark[['IceCreamSales', 'Temperature']])
y = shark['SharkAttacks']
model = sm.OLS(y, X).fit()
model.summary()


# The coefficient of ice cream sales decreased and the pvalue increased to above .05 indicating that the IceCreamSales predictor is not statistically signficant with 95% confidence

# # Perfect Multicollinearity #
# 

# In[6]:


shark['SalesNTax'] = shark['IceCreamSales'] * 1.25
shark.corr()


# In[7]:


#just ice cream sales
X = sm.add_constant(shark[['IceCreamSales']])
y = shark['SharkAttacks']
model = sm.OLS(y, X).fit()
model.summary()


# In[8]:


#just sales and tax
X = sm.add_constant(shark[['SalesNTax']])
y = shark['SharkAttacks']
model = sm.OLS(y, X).fit()
model.summary()


# In[9]:


X = sm.add_constant(shark[['IceCreamSales', 'SalesNTax']])
y = shark['SharkAttacks']
model = sm.OLS(y, X).fit()
model.summary()


# # Heteroskedasticity #

# In[10]:


n = 100
x = 25*np.random.randn(n)
x = x[x > 0]
sigma = 2.5
noise = sigma*np.random.randn(len(x))

y = 280 + 3*x + noise*x
#y = 280 + 3*x + noise


# In[11]:


sns.scatterplot(x, y)


# In[12]:


X = sm.add_constant(x)
model = sm.OLS(y, X).fit()

model.summary()


# In[13]:


y_pred = model.predict(X)
sns.scatterplot(y,y-y_pred)


# In[14]:


model = sm.RLM(y, X).fit()

model.summary()


# In[15]:


y_pred = model.predict(X)
sns.scatterplot(y,y-y_pred)


# # Nonlinearity #

# In[16]:


#True form of exponential data
n = 100
x = np.random.randn(n)
ytrue = 5 + 0.5*x**2

sns.scatterplot(x, ytrue)


# In[17]:


#noisy form of exponential data
noise = 1.5*np.random.randn(n)
ynoise = ytrue+noise

sns.scatterplot(x, ynoise)


# In[18]:


#True form linear regression
sns.scatterplot(x, ytrue)
m, b = np.polyfit(x, ytrue, 1)
plt.plot(x, m*x+b, color = 'r')


# In[19]:


#noisy form linear regression
sns.scatterplot(x, ynoise)
m, b = np.polyfit(x, ynoise, 1)
plt.plot(x, m*x+b, color = 'r')


# In[20]:


X = sm.add_constant(x)
model = sm.OLS(ynoise, X).fit()

model.summary()


# In[21]:


#check for nonlinearity using residual plot
y_pred = model.predict(X)
sns.scatterplot(ynoise,ynoise-y_pred)
#if there is a pattern then there's a problem!

