#!/usr/bin/env python
# coding: utf-8

# #Last Lecture!
# 
# # NOTE: This is the code I used in class. So, its not commented, but maybe you can see learn something from it!
# 
# Announcements:
# 
# 1. CAPEs due: Monday, June 7 at 8:00am
# 
#   1.   If >50% submit, everyone gets a bonus points
#   2.   If >75% submit, everyone gets 2 bonus points
# 
# 2. Discussion board 4
# 3. Final project is due Tuesday June 8th at midnight
#   - Any 2 analysis techniques
#   - Descriptive stats in analysis section -> mode for categorical
#   
# 
# Topics:
# 
# 1. Review!

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# lets get some data

# In[2]:


data = pd.read_csv('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/LAozone.data')


# In[3]:


data.head()


# In[4]:


sns.pairplot(data)


# In[5]:


data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/credit.csv')


# In[6]:


data.head()


# In[7]:


sns.pairplot(data)


# In[8]:


data = pd.read_csv('https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/whitewines.csv')


# In[9]:


sns.pairplot(data)


# In[10]:


data.head()


# let's try to predict ozone

# In[11]:


sns.distplot(data['ozone'],bins=20)


# is this normally distributed?

# In[12]:


data.columns


# In[13]:


365-91


# In[14]:


var = np.where((data['doy']>=91) & (data['doy']<=274),1,0)


# In[15]:


data.insert(len(data.columns),'YearPoint2',var)


# In[16]:


data.head()


# In[17]:


sns.jointplot(data['temp'],data['ozone'],hue=data['YearPoint2'])


# In[18]:


sns.scatterplot(data['temp'],data['ozone'],hue=data['YearPoint'],alpha=.5)
plt.title('Cool Plot - give me extra credit')
plt.show()


# wind does not appear related to ozone
# 
# Temp does

# In[19]:


data.isnull().any() # this uses isnull method and then the any method, which looks for tru


# In[20]:


X = np.array(data['temp']).reshape(data.shape[0],1)
y = data['ozone']


# In[21]:


sns.scatterplot(X[:,0],y)


# In[22]:


data.corr()


# Can I predict the ozone based on temperature?

# What kind of model should we use here?

# In[23]:


model = LinearRegression()
kfold = model_selection.KFold(n_splits=5,shuffle=True,random_state=1)
results1 = model_selection.cross_val_score(model,X,y,cv=kfold)
ymodel1 = model_selection.cross_val_predict(model,X,y,cv=kfold)
print(results1.mean(),results1.std())


# In[24]:


sns.scatterplot(ymodel1,ymodel1-y)
plt.xlabel('residuals')


# In[25]:


p = PolynomialFeatures(2) # create the polynomial object we are interested in
X_p = p.fit_transform(X) # get the transformed features

model = LinearRegression()
kfold = model_selection.KFold(n_splits=5,shuffle=True,random_state=1)
results2 = model_selection.cross_val_score(model,X_p,y,cv=kfold)
ymodel2 = model_selection.cross_val_predict(model,X_p,y,cv=kfold)
print(results2.mean(),results2.std())


# In[26]:


sns.scatterplot(ymodel2,ymodel2-y)
plt.xlabel('residuals')


# In[27]:


sns.set_context('notebook')


# In[28]:


sns.scatterplot(x= X[:,0],y =y,label='raw')
sns.scatterplot(x = X[:,0],y=ymodel1,color='r',label='linear')
sns.scatterplot(x = X[:,0],y= ymodel2,color='g',label='non-linear')
#sns.scatterplot(x = X[:,0],y= ymodel_whoa,color=[0,.5,1],label='non-linear')
plt.xlabel('temperature')
plt.legend()
plt.show()


# We want to use cross validation to evaluate our prediction
# 
# Kfold is a good method
# 
# our data is ordered

# In[29]:


data.head(20)


# can we use the ozone to predict whether its hot or cold outside?

# In[30]:


X = np.array(data['ozone']).reshape(data.shape[0],1)
y = data['YearPoint2']


# In[31]:


sns.boxplot(y=X[:,0],x=y)
plt.ylabel('ozone')


# In[32]:


model = LogisticRegression()
kfold = model_selection.KFold(n_splits=5,shuffle=True,random_state=1)
results1 = model_selection.cross_val_score(model,X,y,cv=kfold)
ymodel1 = model_selection.cross_val_predict(model,X,y,cv=kfold)
print(results1.mean(),results1.std())


# In[33]:


model = tree.DecisionTreeClassifier()
kfold = model_selection.KFold(n_splits=5,shuffle=True,random_state=1)
results2 = model_selection.cross_val_score(model,X,y,cv=kfold)
ymodel2 = model_selection.cross_val_predict(model,X,y,cv=kfold)
print(results2.mean(),results2.std())


# In[34]:


fpr, tpr, thresholds = metrics.roc_curve(y,ymodel1)


# In[35]:


fpr2, tpr2, thresholds2 = metrics.roc_curve(y,ymodel2)


# roc plot

# In[36]:


plt.plot(fpr,tpr,marker='^',label='logit')
plt.plot(fpr2,tpr2,marker='^',label='tree')
plt.plot([0,1],[0,1],linestyle=':',color=[0,0,0])

plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend()

plt.show()


# In[37]:


data.columns


# In[38]:


y = data['ozone']


# In[39]:


model = LinearRegression()
kfold = model_selection.KFold(n_splits=5,shuffle=True,random_state=1)
results_whoa = model_selection.cross_validate(model,data[['vh', 'wind', 'humidity', 'temp', 'ibh', 'dpg', 'ibt', 'vis']],y,cv=kfold,return_train_score=True)
ymodel_whoa = model_selection.cross_val_predict(model,data[['vh', 'wind', 'humidity', 'temp', 'ibh', 'dpg', 'ibt', 'vis']],y,cv=kfold)
print(results_whoa)


# In[40]:


results_whoa['train_score']


# In[41]:


results_whoa['test_score']

