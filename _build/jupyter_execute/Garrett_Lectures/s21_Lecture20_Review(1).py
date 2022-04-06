#!/usr/bin/env python
# coding: utf-8

# Importing a bunch of libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Importing our data

# In[2]:


data = pd.read_excel('NBA_playing_around_1972to2019.xlsx',index_col=0)


# Checking out our data

# In[3]:


data.head(15)


# In[4]:


data.tail()


# Seems strange that our index ends at 15? Probably just need to redfine the index

# In[5]:


data.index = range(0,data.shape[0])


# In[6]:


data.tail()


# Looks check out the describe now
# 
# 

# In[7]:


data.describe()


# strange that GB (games behind) doesn't show up. Oh look. In the head(), we see that some are labelled as '-'. We should fix that

# In[8]:


data['GB'] = np.where(data['GB']!='—',data['GB'],np.nan)


# In[9]:


data.head()


# In[10]:


data['GB'] = data['GB'].astype('float64')


# In[11]:


data.describe()


# In[12]:


data.info()


# Much better

# Now, let's check out the data with a pairplot!

# In[13]:


plt.figure(figsize=[40,40])
sns.pairplot(data)


# Out of curiosity, I wanted to check out the Team names, in case that was something we could play around with. But not so much. Too many teams and too many instances where the team name is a bit weird

# In[14]:


data['Team'].unique()


# So, what should we do? What data looks interesting to you guys?
# 
# People want to fit a curvy line to points scored per game (PS/G) and the year

# In[15]:


X = data[['Year']]
y = data['PS/G'] 


# It certainly looks like the points scored per game when up in the 80s, down in the 00s, and is now back on the rise

# In[16]:


sns.scatterplot(X['Year'],y)


# Is this linear or non-linear?
# 
# Can we predict PS/G using year?

# In[17]:


sns.boxplot(X['Year'])


# In[18]:


sns.boxplot(y)


# In[19]:


data.duplicated().any() # probably should have check earlier = but you can trust Garrett's web scraping


# any other feature engineering?
# 
# Oh yeah. Missing data. Info told us there weren't any, but let's check the heatmap anyways

# In[20]:


sns.heatmap(data.isnull())


# check R^2 value for testing set between linear and different polynomial fits

# In[21]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)


# Code taken from a previous lecture that we adapted

# In[22]:


sns.scatterplot(X_test[X_test.columns[0]],y_test)

training_score = []
testing_score = []
polys = [1,2,3,4,5]
for degree in polys:
  
  print(degree)
  p = PolynomialFeatures(degree)
  X_p = p.fit_transform(X_train) 
  model = LinearRegression() 
  model.fit(X_p, y_train) 

  sns.lineplot(X_test[X_test.columns[0]],model.predict(p.fit_transform(X_test)),label = str(degree))

  training_score.append(model.score(X_p,y_train))
  testing_score.append(model.score(p.fit_transform(X_test),y_test))

plt.legend()


# Interesting that the 3, 4, and 5th polynomials all overlap. But, so it goes I guess!
# 
# When you check the training adn testing scores, the data is best predicted when the polynomial is at least 3, which is what we see in the figure above

# In[23]:


training_score


# In[24]:


testing_score


# Let's see points allowed per game (PA/G) 
# 
# Looks very similar to PS/G

# In[25]:


sns.scatterplot(data['Year'],data['PA/G'])


# Based on this finding, we may have the following logic:
# 
# If PS/G is a wavy line
# If PA/G is also a wavy line
# 
# and, these wavy lines basically overlap
# 
# We've expect that year and Diff (i.e., PS/G - PA/G) probably has no relationship
# 
# And that is what we do see! 

# In[26]:


sns.scatterplot(data['Year'],data['Diff'])

