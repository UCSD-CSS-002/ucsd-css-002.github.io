#!/usr/bin/env python
# coding: utf-8

# # Classification, again
# 
# Apologies.  Last class went *way* too fast.  I'm going to back up and try again from first principles, bit by bit.
# 
# Consequences: 
# - more classification today.   
# - lab 7 due by sunday  
# - lab 8 due by sunday  
# 

# # Classification
# 
# **Basic problem:**  Define a rule that predicts labels based on input features.  
# 
# **ML formulation:** Given some prior examples with correct labels, *learn* a rule to do this.

# ## Manual Example
# 
# Classifying legendary pokemon.

# In[1]:


import matplotlib.pyplot as plt
plt.rc('font', size=16)


# In[2]:


import pandas as pd
import numpy as np
poke = pd.read_csv('Pokemon.csv')
poke


# In[3]:


X = poke['Total']
y = poke['Legendary']


# ### Manual classification
# 
# Let's pick a rule based on the single variable called `Total`.  
# That rule amounts to a threshold: everything above that threshold we will call Legendary, everything below, we will call not legendary

# In[4]:


threshold = 550
y_predicted = X > threshold
(y == y_predicted).mean()


# In[5]:


y.mean()


# ### Manual calculation of confusion matrix, etc.
# 
# rows: label in reality
# cols: label we predict
# 
# 

# In[6]:


# how many things are actually legendary and *predicted* as legendary?

# True & True
print(sum((y==True) & (y_predicted == True)))
# True & False: legendary pokemon that we labeled as not legendary
print(sum((y==True) & (y_predicted == False)))
# False & True
print(sum((y==False) & (y_predicted == True)))
# False & False
print(sum((y==False) & (y_predicted == False)))


# In[7]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y, y_predicted)


# True positive Rate:  what fraction of things that were truly positive, did we label as positive.   
# 
# False positive rate:  what fraction of things that were not positive in reality, did we label as positive

# In[8]:


tpr = sum((y==True) & (y_predicted == True)) / sum(y==True)
tpr


# In[9]:


fpr = sum((y==False) & (y_predicted == True)) / sum(y==False)
fpr


# In[10]:


def tpr_fpr(threshold):
    y_predicted = X > threshold
    positive_rate = (y_predicted==True).mean()
    tpr = sum((y==True) & (y_predicted == True)) / sum(y==True)
    fpr = sum((y==False) & (y_predicted == True)) / sum(y==False)
    return {'tpr':tpr, 'fpr':fpr, 'ppr':positive_rate}


# In[11]:


threshold = 700.5
print(tpr_fpr(threshold = threshold))
_ = plt.hist(X[y==True], color='green', alpha=0.5)
_ = plt.hist(X[y==False], color='blue', alpha=0.5)
_ = plt.legend(['legendary', 'not'])
_ = plt.plot([threshold,threshold], [0, 150], 'r--')


# ### Manual plotting of Receiver Operating Characteristic

# In[12]:


X = poke['Total']
# X = np.random.randint(150, 900, 800)
y = poke['Legendary']

tprs = []
fprs = []

for threshold in range(150, 900, 10):
    evaluation = tpr_fpr(threshold = threshold)
    #print(f"{threshold=}, tpr: {evaluation['tpr']}, fpr: {evaluation['fpr']}")
    tprs.append(evaluation['tpr'])
    fprs.append(evaluation['fpr'])


# In[13]:


_ = plt.plot(fprs, tprs, 'ko-')
_ = plt.xlabel('false positive rate')
_ = plt.ylabel('true positive rate')

_ = plt.plot([0,1], [0,1], 'r-')


# ### Manual optimization of decision rule
# 
# What is our overall loss/gain function?

# In[14]:


# loss function: how much do we care about different kinds of errors:
# false positive  -1
# false negatives: 1-tpr  -1

def loss(y, y_predicted):
    # of false positives:
    false_positives = sum((y == False) & (y_predicted == True))
    # of false negatives
    false_negatives = sum((y == True) & (y_predicted == False))
    # conf_mat = confusion_matrix(y, y_predicted)
    # false_positives = conf_mat[0,1]
    # false_negatives = conf_mat[1,0]
    return false_positives + 5*false_negatives


# In[15]:


threshold = 6000
y_predicted = X > threshold
loss(y, y_predicted)


# In[16]:


min_loss = 800
min_loss_threshold = None
for threshold in range(150, 900):
    y_predicted = X > threshold
    cur_loss = loss(y, y_predicted)
    if cur_loss < min_loss:
        min_loss = cur_loss
        min_loss_threshold = threshold

print(min_loss_threshold, min_loss)


# In[17]:


y_predicted = X > min_loss_threshold
confusion_matrix(y, y_predicted)


# In[18]:


poke


# ## Scikit learn classification
# 
# 

# ## K nearest neighbors

# ### Labels and features

# In[19]:


X = poke.loc[:,'HP':'Generation']
y = poke['Legendary']


# In[20]:


X


# ### Train and test split

# In[21]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size = 0.75, 
                                                    random_state=0)
# note, random_state provided to yield consistent behavior over runs


# ### fit classifier

# In[22]:


from sklearn.neighbors import KNeighborsClassifier

nn_3 = KNeighborsClassifier(3)
nn_3.fit(X_train, y_train)


# ### Evaluate performance (on test)

# In[23]:


from sklearn.metrics import accuracy_score
y_test_prediction = nn_3.predict(X_test)
accuracy_score(y_test, y_test_prediction)


# In[24]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_test_prediction)


# ### **Decision tree**
# 
# Build a tree of binary decisions of the form "feature X >= threshold", so as to separate the classes.

# In[25]:


from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

# max_depth

dt = DecisionTreeClassifier(max_depth = 8)
dt.fit(X_train, y_train)


# In[26]:


fig = plt.figure(figsize = (50,50))
_ = tree.plot_tree(dt, 
                   feature_names = poke.loc[:,'HP':'Generation'].columns,
                  class_names = ['Not Legendary', 'Legendary'],
                  filled = True)


# ## Evaluating what a classifier is doing

# In[27]:


X = poke[['Defense','Attack']]
y = poke['Legendary']


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size=0.75,
                                                   random_state=0)


# In[29]:


dt = DecisionTreeClassifier(max_depth = 8)
dt.fit(X_train, y_train)


# In[30]:


nn_3 = KNeighborsClassifier(3)
nn_3.fit(X_train, y_train)


# In[31]:


defense = np.linspace(X['Defense'].min(),X['Defense'].max(), 50)
attack = np.linspace(X['Attack'].min(),X['Attack'].max(), 50)


# In[32]:


longdefense = []
longattack = []
predictions = []

for d in defense:
    for a in attack:
        longdefense.append(d)
        longattack.append(a)
        predictions.append(nn_3.predict([[d,a]])[0])

longdefense = np.array(longdefense)
longattack = np.array(longattack)
predictions = np.array(predictions)


# In[33]:


plt.scatter(longdefense[predictions==True], 
            longattack[predictions==True], s=1, 
            color='green')
plt.scatter(longdefense[predictions==False], 
            longattack[predictions==False], s=1, 
            color='red')
plt.legend(['Legendary', 'not legendary'])
plt.xlabel('defense')
plt.ylabel('attack')

#plt.plot(X_train['Defense'][y_train], 
#         X_train['Attack'][y_train], 'go', alpha=0.3)

#plt.plot(X_train['Defense'][y_train==False], 
#         X_train['Attack'][y_train==False], 'ro', alpha=0.3)


# In[34]:


fig = plt.figure(figsize = (10,10))
_ = tree.plot_tree(dt, 
                   feature_names = X_train.columns,
                  class_names = ['Not Legendary', 'Legendary'],
                  filled = True)


# ## Linear / Quadratic discriminant analysis
# 
# Model the classes as multivariate gaussians either with constant covariances (linear) or with different covariances (quadratic).  Draw the resulting boundary based on posterior probability.
# 
# ```
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# ```
# 
# ![linear quad disc](sphx_glr_plot_lda_qda_001.png)

# ## Logistic Regression
# 
# Log odds of positive choice increases as a linear function of *features*.  Log-odds yields probability via logistic transform.
# 
# ![logistic](linear_vs_logistic_regression_edxw03.webp)

# ## Support vector machines
# 
# Find a maximum margin separating boundary.  Boundary determined by closest instances -- support vectors.
# 
# ![max margin](600px-SVM_margin.png)

# ### Kernel trick
# 
# Intuition: project features into a higher dimensional space via non-linear transformations.  Trick is that instead of representing the new features, replace feature representation of data points with data-data similarity, and replace similarity between feature vectors via a kernel, rather than just a dot-product.
# 
# ![kernel trick](440px-Kernel_trick_idea.svg.png)

# ## Simple neural networks
# 
# ```from sklearn.neural_network import MLPClassifier```
# 
# 
# ![perceptron](multilayerperceptron_network.png)

# ### Ensemble models
# 
# **Random forests**
# 
# ```sklearn.ensemble.RandomForestClassifier```
# 
# **Gradient Boosted trees**
# 
# ```sklearn.ensemble.GradientBoostingClassifier```

# ## Fairness
# 
# Problem: machine learning algorithms make predictions that we do not like.  E.g.:
# 
# - predict higher recidivism rates for black vs white parolees,  
# 
# - predict higher click-through rates on ads for executive jobs presented to male viewers  
# 
# - systematically misidentify black faces in photos   
# 
# - translate 'the surgeon ate her lunch' into 'the surgeon ate his lunch'
# 
# We do not like this, and we would like machines to behave better.
# 
# ### Why do machines do this? and what can be done?
# 
# machines do not know, or care about, our particular notions of race, gender, etc.  
# 
# these behaviors arise from fitting models to optimize prediction on some training data.  
# 
# Some possible reasons / solutions:
# 
# - **the bias is in the data selection/sampling process**.  Careful stratified, unbiased, random sampling of data (rather than convenience, natural-world samples).  Generate synthetic data without bias, etc.
# 
# - **the bias is in the training labels**.  Throw out those labels, and get unbiased labels.
# 
# - **the bias is in the feature representation**.  Hopefully better features are available, or features may be transformed.
# 
# - **the bias is in the world**.  This is unfortunately a very common, and most tricky, situation.  Various criteria for algorithm fairness try to deal with this problem.
# 
# 
# ### Defining fairness
# 
# Fairness is not easy to define, in the face of differences in the world.  There are multiple, intuitively appealing, definitions of fairness that cannot be simultaneously achieved.
# 
# Let's say we want to predict whether a given loan applicant will default on their loan, based on FICO score (and not give a loan in that case). 
# 
# ![Fico scores](fico-scores.png)
# 
# What is a fair way to predict default rates (and thus to decide on whether to give a loan to particular individuals, or to decide how high interest rate to charge?)
# 
# 

# In[ ]:




