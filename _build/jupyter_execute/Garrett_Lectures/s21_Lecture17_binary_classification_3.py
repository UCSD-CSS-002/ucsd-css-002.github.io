#!/usr/bin/env python
# coding: utf-8

# Binary Classification 3
# 
# 1. Binary classification with multiple predictors
# 2. Support vector machines
# 
# As always let's import our libraries

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import tree


# And just like last class, let's focus on the  MPG dataset and see if we can predict whether a car has an origin from the USA or Not USA using mpg as the predictor

# In[2]:


mpg = sns.load_dataset('mpg')


# In[3]:


dummy_origin =  pd.get_dummies(mpg['origin'])
mpg.insert(len(mpg.columns),'usa',dummy_origin['usa'])


# In[4]:


mpg.head()


# As we saw yesterday, mpg by itself seems like a better good predictor for whether the car is USA or not usa

# In[5]:


sns.boxplot(mpg['usa'],mpg['mpg'])


# In[6]:


Xtrain,Xtest,ytrain,ytest = train_test_split(mpg[['mpg']],mpg['usa'],random_state=1)


# In[7]:


logit_model = LogisticRegression(random_state =1)
logit_model = logit_model.fit(Xtrain, ytrain)
print(logit_model.score(Xtrain,ytrain))
print(logit_model.score(Xtest,ytest))


# In[8]:


tree_model = tree.DecisionTreeClassifier(max_depth=1)
tree_model = tree_model.fit(Xtrain, ytrain)
print(tree_model.score(Xtrain,ytrain))
print(tree_model.score(Xtest,ytest))


# With a single predictor, we do a pretty decent job. >75% is pretty good!

# But we aren't always dealing with one predictor and we can probably do even better with multiple predictors!

# In[9]:


mpg.head()


# Let's look at the scatterplot of total_phenols and proline

# In[10]:


sns.scatterplot(mpg['displacement'],mpg['mpg'],hue=mpg['usa'])


# Using both predictors, we can probably do an even better job of separating groups 0 and 1!

# In[11]:


X = mpg[['displacement','mpg']]


# In[12]:


Xtrain,Xtest,ytrain,ytest = train_test_split(X,mpg['usa'],random_state=1)


# And now let's this new data with a logistic regression

# In[13]:


logit_model2 = LogisticRegression()
logit_model2 = logit_model.fit(Xtrain, ytrain)


# And let's see the score of the train and test

# In[14]:


print(logit_model.score(Xtrain,ytrain))
print(logit_model.score(Xtest,ytest))


# 
# Just like with linear regression, its hard to see what this fit looks like in logistic regression when there are multiple predictors
# 
# Let's take a peek with a tree diagram

# In[15]:


tree_model2 = tree.DecisionTreeClassifier(max_depth=1)
tree_model2.fit(Xtrain, ytrain)
print(tree_model2.score(Xtrain,ytrain))
print(tree_model2.score(Xtest,ytest))


# Try playing around with the max_depth and you will see at what level the decision tree uses both dfeatures.

# In[16]:


tree_model3 = tree.DecisionTreeClassifier(max_depth=3) 
tree_model3.fit(Xtrain, ytrain)
print(tree_model3.score(Xtrain,ytrain))
print(tree_model3.score(Xtest,ytest))


# In[17]:


plt.figure(figsize=[12,12])
tree.plot_tree(tree_model3)


# With max_depth 1 or 2, it only considered X[1] (i.e., the second column of data) and basically splits with just that column. Accuracy doesn't change much either
# 
# With a max_depth of 3, you can see it using X[0] in some of the branches. And our accuracy is better! Nice!

# But there is another really useful method for binary classification
# 
# Support vector classification 
# 
# https://en.wikipedia.org/wiki/Support_vector_machine#Linear_SVM
# 
# And let's import the library

# In[18]:


from sklearn.svm import SVC


# Here, we will create an instance of the model with a linear kernal, meaning it will separate the data using just a line

# In[19]:


svc_model = SVC(kernel='linear')
svc_model.fit(Xtrain, ytrain)


# Let's check the accuracy of this SVC model.

# In[20]:


print(svc_model.score(Xtrain,ytrain))
print(svc_model.score(Xtest,ytest))


# Pretty good!
# 
# We can check the predictions. Again, it returns a 1 or 0 on whether it thinks the data will belong to group 1 or 0

# In[21]:


svc_model.predict(Xtrain)


# We can view our support vectors, which are the datapoints on the margin

# In[22]:


svc_model.support_vectors_


# And plot them onto our data. This gives us a quick look into how ou data was separate

# In[23]:


Xtrain


# In[24]:


sns.scatterplot(Xtrain['displacement'], Xtrain['mpg'], hue=ytrain)
sns.scatterplot(svc_model.support_vectors_[:,0],svc_model.support_vectors_[:,1],color='k')


# Alterantively, we can plot our margin using the coef_ and intercept_ of the model.
# 
# Note: this is specific to linear SVC

# In[25]:


svc_model.coef_


# In[26]:


svc_model.intercept_


# To use the coef_ and intercept, we need to do a few things. First, let's pull out the coefficents, which are basically weights

# In[27]:


w = svc_model.coef_[0]
print(w)


# Dividing them basically gets us a slope

# In[28]:


slope = -w[0] / w[1] 
print(slope)


# Our intercept needs to be adjusted by the weight of the y-axis

# In[29]:


intercept = -svc_model.intercept_[0]/w[1]
print(intercept)


# Then, using Xtrain as the X in our y = m*x + b equation, we can derive our line

# In[30]:


yy = slope * Xtrain['displacement'] + intercept


# Let's put this all on the plot

# In[31]:


sns.scatterplot(Xtrain['displacement'], Xtrain['mpg'], hue=ytrain)
plt.plot(Xtrain['displacement'], yy)
sns.scatterplot(svc_model.support_vectors_[:,0],svc_model.support_vectors_[:,1],color='k')
plt.ylim([0,50])


# Looks good!
# 
# We can do this with the testing data too to get a feel for how well it separated it

# In[32]:


yy = slope * Xtest['displacement'] + intercept

sns.scatterplot(Xtest['displacement'], Xtest['mpg'], hue=ytest)
plt.plot(Xtest['displacement'], yy)
plt.ylim([0,50])


# Here is another method to generate the margin. Its a bit more complex

# In[33]:


sns.scatterplot(Xtrain['displacement'], Xtrain['mpg'],hue=ytrain)

# create grid to evaluate model
xx = np.linspace(0, 500, 30)  # works best if you know the xlim[0] and xlim[1]
yy = np.linspace(0,60, 30)  # works best if you know the ylim[0] and ylim[1]
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svc_model.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])


# That's pretty much most of what I wanted to cover of binary classification
# 
# There are other types of classification techniques (Random forest, perceptron, k-nearest neighbors) and we could certainly spend a lot more time talking about the naunces of Decision Trees and Support Vector machines.
# 
# For now, I think this should give you a basic idea of how to conduct classification, what is happening, and how to evaluate the classification
# 
# Next, we will cover what cross-validation is because it concerns how we conduct our machine learning
