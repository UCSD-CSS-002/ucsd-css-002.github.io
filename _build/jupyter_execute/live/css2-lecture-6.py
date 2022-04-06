#!/usr/bin/env python
# coding: utf-8

# # Modeling data part 2
# 
# ## From last time:
# 
# ### model: 
# 
# **what is the structure of our model?**
# 
# e.g., linear regression
# 
# ### model parameters: 
# 
# **what knobs does the model have?**
# 
# e.g., slope, intercept
# 
# ### loss function: 
# 
# **how badly does the model perform?**
# 
# e.g., mean squared error, mean absolute error, proportion imperfect match, etc.
# 
# ### fitting a model: 
# 
# **what parameter values minimize loss?**
# 
# done via optimization algorithms, 
# 
# e.g., brute force grid search, random search, gradient descent.

# ## Review live code example

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

n = 100

y = np.random.randint(48, 96, size=n)

cnt, bins, fig = plt.hist(y)


# Consider the following model/parameters/loss.  Let's write functions to do each.
# 
# **model**
# 
# $\hat y = m$
# 
# **parameters:** 
# 
# $m$
# 
# **loss:**
# 
# $\mathrm{loss}(\hat y, y) = \frac{1}{n} \sum_{i=0}^{n} (\hat y_i - y_i)^2$
# 
# $\mathrm{loss}(m) = \frac{1}{n} \sum_{i=0}^{n} (m - y_i)^2$
# 

# In[2]:


# what arguments should each function take?
# what should it do / return?

def predict(m):
    """returns predicted y given parameters"""
    return m
    
def loss(y_hat, y):
    """returns mse loss of predicted y relative to observed y"""
    return ((y_hat - y)**2).mean()
    
def evaluate_parameters(m, y):
    """returns mse loss for parameter m, relative to observed y"""
    return loss(predict(m), y)
    
def fit(y):
    """returns best fitting parameters given some observed y values"""
    
    best_loss = None
    best_m = None
    
    for m in np.linspace(min(y), max(y), 1000):
        current_loss = evaluate_parameters(m,y)
        
        if best_loss is None or current_loss < best_loss:
            best_loss = current_loss
            best_m = m
    
    return best_m
    
    


# In[3]:


fit(y)


# ## Gradient descent optimization

# ### What's a gradient?

# In[4]:


all_ms = np.linspace(48,96,1000)

plt.plot(all_ms, 
         np.vectorize(lambda m: ((y-m)**(2)).mean())(all_ms))
fig = plt.xlabel('m')
fig = plt.ylabel('loss')
fig = plt.title('MSE loss function for $\hat y = m$')


# ### Gradient
# 
# The partial derivative of the loss function with respect to the parameters.
# 
# $\mathrm{loss}(m) = \frac{1}{n} \sum_{i=0}^{n} (m - y_i)^2$
# 
# This is usually calculated via liberal use of the [chain rule](https://en.wikipedia.org/wiki/Chain_rule)
# 
# $\frac{\partial \mathrm{loss}(m)}{\partial m} = \frac{1}{n} \sum_{i=0}^{n}  2 (m - y_i)$

# In[5]:


def gradient(m,y):
    """partial derivative of mse loss wrt constant prediction"""
    return (2*(m-y)).mean()


# In[6]:


fig = plt.plot(all_ms, np.vectorize(lambda m: gradient(m,y))(all_ms))
fig = plt.axhline(y=0, color='r', linestyle='--')
fig = plt.xlabel('m')
fig = plt.ylabel('gradient of loss wrt m')
fig = plt.title('gradient of MSE loss function for $\hat y = m$')


# ### Gradient descent
# 
# 0. pick a starting value of parameters
# 
# 1. find gradient for current parameter values
# 
# 2. shift parameters in direction of gradient, by some step size (learning rate).
# 
# 3. repeat steps 1 and 2... until....
# 
# Optional variations:
# 
# ...until: some number of steps have been taken
# 
# ...until: improvement is small
# 
# learning rate reduces over time.
# 
# 

# In[7]:


def gradient_descent(y, learning_rate = 0.1):
    current_m = 0
    
    for step in range(100):
        current_gradient = gradient(current_m, y)
        
        current_m = current_m - learning_rate*current_gradient
    
    return current_m

gradient_descent(y)


# In[8]:


def loss(m,y):
    return ((y-m)**(2)).mean()


# In[9]:


ms = [50]
losses = [loss(ms[-1],y)]

for step in range(100):
    ms.append(ms[-1] - 0.1*gradient(ms[-1], y))
    losses.append(loss(ms[-1],y))


# In[10]:


plt.plot(all_ms, np.vectorize(lambda m: ((y-m)**(2)).mean())(all_ms))
plt.plot(ms, losses, 'ko-')
plt.xlabel('m')
plt.ylabel('loss')
plt.title('trace of gradient descent')


# ### Gradient descent and learning rate
# 
# Different regimes:  
# 
# - appropriate (rapidly dampening)
# 
# - too small (slow learning)
# 
# - too big (diverges!)
# 

# In[11]:


ms = [50]
losses = [loss(ms[-1],y)]

for step in range(100):
    ms.append(ms[-1] - 0.1*gradient(ms[-1], y))
    losses.append(loss(ms[-1],y))
    
    
plt.plot(all_ms, np.vectorize(lambda m: ((y-m)**(2)).mean())(all_ms))
plt.plot(ms, losses, 'ko-')
plt.xlabel('m')
plt.ylabel('loss')
plt.title('trace of gradient descent')


# ### Gradient descent for us:
# 
# Understand roughly what the gradient is.
# 
# Understand what the learning rate is, to be able to tweak it.
# 
# I will not ask you to do calculus.

# ## Overfitting
# 
# 

# In[12]:


def generate_y(x):
    return np.sin(x/50*np.pi)*100+np.random.randint(-10,10,x.shape)

x = np.random.randint(-100, 100, 10)
y = generate_y(x)

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')


# ### Model "complexity"
# 
# With polynomials
# 
# e.g., 
# 
# 0th order: $\hat y = b_0 x^0 = \hat y = b_0$
# 
# 1st order: $\hat y = b_0 x^0 +  b_1 x^1$
# 
# 2nd order: $\hat y = b_0 x^0 +  b_1 x^1 +  b_2 x^2$
# 
# 3rd order: $\hat y = b_0 x^0 +  b_1 x^1 +  b_2 x^2 +  b_3 x^3$
# 
# kth order: $\hat y = \sum_{i=0}^k b_k x^k$
# 
# 

# In[13]:


all_xs = np.linspace(-100,100, 100)

def poly_predict(x, coefs):
    prediction = np.zeros(x.shape)
    for order,coef in enumerate(reversed(coefs)):
        prediction += x**order * coef
    return prediction


# In[14]:


coefs = np.polyfit(x,y,9)
print(coefs)

f = plt.scatter(x,y)
f = plt.xlabel('x')
f = plt.ylabel('y')
f = plt.plot(all_xs, poly_predict(all_xs, coefs), 'r-')


# ### Training loss as a function of order

# In[15]:


def loss(predicted_y,y):
    """mean squared error"""
    return ((predicted_y-y)**2).mean()

ks = np.arange(0, 9)
training_loss = np.zeros(ks.shape)

for k in ks:
    coefs = np.polyfit(x,y,k)
    training_loss[k] = loss(poly_predict(x,coefs),y)

plt.plot(ks, training_loss, 'ko-')
plt.xlabel('polynomial order')
plt.ylabel('training loss')
plt.yscale('log')


# ### Loss on *new* data

# In[16]:


new_xs = np.random.randint(-100, 100, 10)
new_ys = generate_y(new_xs)


# In[17]:


coefs = np.polyfit(x,y,9)
plt.scatter(x, y, marker = 'o', color='black')
plt.scatter(new_xs, new_ys, marker = 'o', color='green')
plt.plot(all_xs, poly_predict(all_xs, coefs), 'r-')


# In[18]:


ks = np.arange(0, 9)
training_loss = np.zeros(ks.shape)
new_data_loss = np.zeros(ks.shape)

for k in ks:
    coefs = np.polyfit(x,y,k)
    training_loss[k] = loss(poly_predict(x,coefs),y)
    new_data_loss[k] = loss(poly_predict(new_xs,coefs),new_ys)


# In[19]:


fig, axs = plt.subplots(2,1, figsize = (4,8))
axs[0].plot(ks, training_loss, 'ko-')
axs[0].set_xlabel('polynomial order')
axs[0].set_ylabel('loss')
axs[0].set_title('training loss')
axs[0].set_yscale('log')
axs[1].plot(ks, new_data_loss, 'mo-')
axs[1].set_xlabel('polynomial order')
axs[1].set_ylabel('loss')
axs[1].set_title('new data loss')
axs[1].set_yscale('log')
fig.tight_layout()


# In[20]:


plt.scatter(training_loss, new_data_loss, marker = 'o', color='y')
for k in range(9):
    plt.text(training_loss[k], new_data_loss[k], str(k), color='k', fontsize = 16)
plt.plot([min(training_loss), max(new_data_loss)], [min(training_loss), max(new_data_loss)], 'r-')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('training loss')
plt.ylabel('new data loss')
plt.title('training and new data loss as a function of polynomial order')


# ### Overfitting
# 
# The thing happening above is called *overfitting*.  A complicated model has many degrees of freedom to wiggle through all the data points in the training data set.  But it will be fitting *noise* not *signal*.  And, as such, will generalize to new data badly.
# 
# 
# 

# ### Cross validation
# 
# To **evaluate** a model, we want to know how well it will do on *new* data.  But we don't have new data.  What are we to do?
# 
# The answer is called "cross validation".  We will use *part* of our data to train the model, and the rest of our data to evaluate the model.

# In[21]:


import pandas as pd

bf = pd.read_csv('bodyfat.csv')

bf = bf[bf['height'] > 30]

bf = bf[bf['age'] < 30]

bf


# In[22]:


plt.scatter(bf['abdomen'], bf['bf.percent'])
plt.xlabel('abdomen circumference (cm)')
plt.ylabel('bodyfat percentage')


# ### train test split
# 
# split the data into a *training* set and a *test* set.
# 
# let's say we will take 50% of the data for the training set, and 50% for the test set.
# 
# 

# In[23]:


bf


# - random sampling (with replacement) of indexes: might get overlap between training and test, and we might not include all the data.
# 
# - random sampling without replacement twice.  

# In[24]:


from random import sample

n = len(bf)


# In[25]:


# n = 35
35//2 # equivalent to int(35/2)


# In[26]:


n_train = n//2
n_test = n - n_train

print(n, n_train, n_test)


# In[27]:


train_idx = sample(list(range(n)), n_train)
test_idx = list(set(range(n)) - set(train_idx))


# In[28]:


print(set(train_idx) & set(test_idx)) # no overlap
print(set(range(n)) - (set(train_idx) | set(test_idx)) ) # full coverage


# We should not just take the first half for training, and second half for test, because the data appear to be sorted by age.  We want test and training data to be *random*

# In[29]:


n = len(bf)
n


# In[30]:


n_train = n//2
n_test = n - n_train
assert (n_test + n_train) == n


# In[31]:


all_idx = list(range(n))
print(all_idx)


# In[32]:


from random import shuffle

shuffle(all_idx)

print(all_idx)


# In[33]:


train_idx = all_idx[:n_train]
test_idx = all_idx[n_train:]

assert len(train_idx) == n_train  # right length
assert len(test_idx) == n_test    # right length
assert len(set(train_idx) & set(test_idx)) == 0 # zero overlap
assert len(set(train_idx) | set(test_idx)) == n # full coverage of all data


# In[34]:


train_data = bf.iloc[train_idx]
test_data = bf.iloc[test_idx]


# In[35]:


test_data


# In[36]:


train_data


# In[37]:


plt.scatter(train_data['abdomen'], train_data['bf.percent'], marker='o', color='k')
plt.scatter(test_data['abdomen'], test_data['bf.percent'], marker='x', color='b')


# In[38]:


polynomial_order = 2

coefs = np.polyfit(train_data['abdomen'], 
                   train_data['bf.percent'], 
                   polynomial_order)

all_xs = np.linspace(min(bf['abdomen']), max(bf['abdomen']), 200)

plt.scatter(train_data['abdomen'], train_data['bf.percent'], marker='o', color='k')
plt.scatter(test_data['abdomen'], test_data['bf.percent'], marker='x', color='b')
plt.plot(all_xs, poly_predict(all_xs, coefs), 'r-')
plt.xlabel('abdomen circumference (cm)')
plt.ylabel('bodyfat percentage')


# In[39]:


print('training loss:', 
      (
          (poly_predict(train_data['abdomen'], coefs) 
           - train_data['bf.percent'])**2).mean())

print('test loss:', ((poly_predict(test_data['abdomen'], coefs) - test_data['bf.percent'])**2).mean())


# In[40]:


poly_order = []
train_loss = []
test_loss = []

for k in range(len(train_data)):
    coefs = np.polyfit(train_data['abdomen'], train_data['bf.percent'], k)
    poly_order.append(k)
    train_loss.append(((poly_predict(train_data['abdomen'], coefs) - train_data['bf.percent'])**2).mean())
    test_loss.append(((poly_predict(test_data['abdomen'], coefs) - test_data['bf.percent'])**2).mean())
    


# In[41]:


l1 = plt.plot(poly_order, test_loss, 'mo-')
l2 = plt.plot(poly_order, train_loss, 'ko-')
plt.legend(['test data', 'training data'])
plt.xlabel('polynomial order')
plt.ylabel('loss')
plt.yscale('log')


# ## Summary
# 
# ### Gradient descent
# 
# Optimization is much more efficient if we can calculate the gradient, and follow it to find better parameters.
# 
# We must specify a learning rate, to dictate how large a step to take in the direction of the gradient.
# 
# Small learning rates create slow optimization.  Large learning rates may create *divergence*.  So these may need to be *tuned*.
# 
# ### Model complexity
# 
# Some models have more parameters, and more opportunities to wiggle than others.
# 
# A more complex model can *always* fit the data better, by using its extra wiggling power.
# 
# ### Overfitting
# 
# A very complex model fitted to a small set of data will *overfit* the data, and will wiggle through the noise, not just the signal.  
# 
# This means that the model will fit the *training* data well, but will *generalize* poorly to new data, meaning it will have a large error on new data it has not seen.
# 
# ### Cross-validation
# 
# To **evaluate** a model, we want to know how well it will do on *new* data, not how well it fit the existing data.
# 
# By definition, we don't have new data.  
# 
# So we pretend that we have only seen part of our data, and reserve the rest to be considered new data.
# 
# We split the data into a *training* subset, and a *test* subset.  We fit the model to the *training* data, and evaluate it on the *test* data, thus estimating how well the model would do on data it was not fitted to.  i.e., on new data.
# 
# 

# ## Cross validation subtleties:
# 
# How much data to use for training, and for testing?
# 
# ### Variations:
# 
# **holdout**
# 
# **k-fold**
# 
# - split the data into k parts.
# 
# - then for each part, use it as the test data once, and the remainder as the training data.
# 
# **repeat random subsampling**
# 
# **nested**

# In[ ]:




