#!/usr/bin/env python
# coding: utf-8

# # Classification
# 
# ## Problem / Goal
# 
# **Goal:** Predict a categorical label based on some associated features.
# 
# **Examples:**
# 
# - Predict which party will win the general election in a particular county.
# - Predict which class a given student will enroll in.
# - Predict whether an image contains a cat.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Example
# 
# Let's say we want to predict whether an NFL field goal attempt will succeed, based on how far away the kick is coming from, and which week of the season it is.
# 
# ### Features and labels

# In[2]:


fg = pd.read_csv('../datasets/fieldgoals.csv')
fg


# The `yardage` and `week` are the *features* of our data, which we are using to predict the `success` *label*/*outcome*.
# 
# We will often separate the features into a variable called `X` and the labels into a variable called `y`:
# 

# In[3]:


X = fg[['yardage', 'week']]
y = fg['success']


# ### Evaluating predictions
# 
# There are many ways we might try to do this prediction.  For instance, we could use our intuitions to pick a rule.  One intuitive rule would be to predict that a fieldgoal will be successful if it came from a distance of less than 30 yards.

# In[4]:


def predict(X):
    return (X['yardage'] < 30).astype(int)

y_predicted = predict(X)


# How good are these predictions?  We could simply ask "how often did we get the right answer"?  What fraction of the time did the success value that we predicted correspond to reality?
# 

# In[5]:


np.mean(y_predicted == y)


# 

# In[6]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y, y_predicted)


# 
