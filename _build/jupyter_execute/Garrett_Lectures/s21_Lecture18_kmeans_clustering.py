#!/usr/bin/env python
# coding: utf-8

# # Topics to cover in Kmeans Clustering
# 1. What is unsupervised learning? What is kmeans?
# 2. Where is kmeans used? --> Show example on music bands and iris dataset
# 3. How to use sklearn function
# 4. Parameters to know when doing kmeans clustering
# 5. Pitfalls of kmeans
# 6. WCSS or Inertia
# 7. Application in robotics

# some helpful sites: (view towardsdatascience in incognito mode)
# 
# https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
# 
# https://jakevdp.github.io/PythonDataScienceHandbook/05.11-k-means.html
# 
# 
# this site has a lot of good stuff at the beginning:
# 
# https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
# 

# First, let's import our libraries

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# An easy way to visualize and learn about k-means clustering is to generate "blobs" of data that we will try to cluster together.
# 
# We can do that using a sklearn library called make_blobs

# In[2]:


from sklearn.datasets.samples_generator import make_blobs


# Here, we are specifing the blobs
# 

# In[3]:


X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)


# X corresponds to our 'features' and is 2-dimensional

# In[4]:


X.shape


# y are basically the clusters. Note: we won't actually use y to form those clusters

# In[5]:


y


# Let's look at our beautiful blob!

# In[6]:


plt.scatter(X[:,0], X[:,1])


# The k-means library can be imported from sklearn.cluster

# In[7]:


from sklearn.cluster import KMeans


# And here, we will create an instance of our model
# 
# Then fit the X data to it

# In[8]:


kmeans = KMeans(n_clusters=4,random_state=0)
kmeans.fit(X)


# ### Terms to understand
# 1. init = method of initalization
# 2. max_iter = max number of iterations
# 3. n_clusters = number of clusters
# 4. n_init = number of different centroid seeds
# 5. random_state
# 6. tol= tolerance

# Now, we will use predict to get the predicted clusters

# In[9]:


y_kmeans = kmeans.predict(X)


# In[10]:


y_kmeans


# In[11]:


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)


# We can compare this to our original clusters to see how well it did. Note that in general, you typically don't have the y values. 
# 
# Anyways, I think this helps demonstrate how well it did!
# 
# Note: the colors don't line up, but that is okay

# In[12]:


fig,axes=plt.subplots(1,2, figsize = (15,5))
axes[0].scatter(X[:, 0], X[:, 1], c=y)
axes[0].set_title('Actual clusters', fontsize=16, fontweight='bold')
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans)
axes[1].set_title('Predicted clusters', fontsize=16, fontweight='bold')


# You can use the code below to find the center of those clusters

# In[13]:


centers = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5); # make them big and transparent!
plt.title('Predicted clusters with centers', fontsize=16, fontweight='bold')


# Some potential limitations:
# 
# 1. need to worry about initial seeds
# 2. need to know the cluster number in advance. 
# 

# let's see what happens when we only look at when n_init is equal to 1.
# 
# n_init is defaulted to be 10, so this isn't something we normally have to worry about. But let's see that kmeans clsutering isn't always perfect

# In[14]:


kmeans = KMeans(n_clusters=4,random_state=2,n_init=1)
kmeans.fit(X)


# In[15]:


centers = kmeans.cluster_centers_
plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X))
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5); # make them big and transparent!


# So, while generally, things work out okay with the algorithm, its important to visualize the data. Because here, this is obviously not as good as before

# When visualizing this dataset, its pretty obvious we need four clusters, but that isn't always easy to detect!
# 
# We can use the Within Cluster Sum of Squares (WCSS) as a way to evaluate the number of clusters. Basically, WCSS will decrease as we add more clusters, but the amount it decreases decreases with the number of clusters. We will pick the cluster number that is at that elbow (i.e., near when WCSS plateaus)
# 
# Let's build a for loop to visualize this and also let's make a plot to see what this looks like. 
# 
# Note that WCSS is inertia in kmeans 
# 
# I'm first going to build a function that does the prediction and plotting for me. Note that this code is basically the same as we ran above. I'm also going to pass in the axes and the cluster number because I need them for my plotting

# In[16]:


sample_X = [2, 5, 8]
sample_Y = [2, 11, 2]
x_centroid, y_centroid = 5, 6
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 13)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Plot to show WCSS calculation for 1 cluster')

ax.scatter(sample_X, sample_Y, c='blue', s=40)
ax.scatter(x_centroid, y_centroid, c='red', s=50)

WCSS = 0
for x, y in zip(sample_X, sample_Y):
  WCSS += np.sqrt(np.abs((x - x_centroid)**2) + np.abs((y - y_centroid)**2)) #using pythagorean theorem c = sqrt(a^2 + b^2)
  ax.plot([x, x_centroid], [y, y_centroid], 'k-')

print(f'WCSS = {WCSS}')


# In[17]:


def plot_clusters(kmeans,X,axes,i):
  y_kmeans = kmeans.predict(X)
  centers = kmeans.cluster_centers_
  axes[i-1].scatter(X[:, 0], X[:, 1], c=y_kmeans)
  axes[i-1].scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# Then, I will create a for loop that goes over different cluster sizes

# In[18]:


fig,axes = plt.subplots(10,1,figsize=(6,30))

wcss = []
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i,random_state=0)
  kmeans.fit(X)
  plot_clusters(kmeans,X,axes,i)
  wcss.append(kmeans.inertia_)


# From these plots, you can see how things evolve with diffeent cluster numbers.
# 
# Next, let's plot wcss as a function of the number of clusters

# In[19]:


plt.plot(range(1, len(wcss)+1), wcss)
plt.xlabel('Number of clusters', fontsize=14)
plt.ylabel('WCSS', fontsize=14)
plt.title('WCSS vs Number of clusters', fontsize=16, fontweight='bold')


# From this, its pretty clear that once we get to 4, there is barely any reduction in wcss. At that point, wcss plateaus. We call that the elbow and we typically select the cluster at the elbow.

# # Kmeans applications in robotics
# 1. [Path planning](https://journals.sagepub.com/doi/full/10.5772/59992)
# 2. [Voronoi visualization](http://cfbrasz.github.io/VoronoiColoring.html)

# Let's try this with some real data! It may not work perfectly well, but let's just take a peek

# In[20]:


mpg = sns.load_dataset('mpg')


# In[21]:


sns.pairplot(data=mpg,hue='origin')


# maybe acceleration and mpg?

# In[22]:


X2 = mpg[['mpg','acceleration']]


# It sort of looks like there are clusters

# In[23]:


sns.scatterplot(X2['mpg'],X2['acceleration'],hue=mpg['origin'])


# It is maybe easier to see with this fancy jointplot that I adapted from someone else's code

# In[24]:


g = sns.JointGrid("mpg", "acceleration", mpg)
for day, day_tips in mpg.groupby("origin"):
    sns.kdeplot(day_tips["mpg"], ax=g.ax_marg_x, legend=False)
    sns.kdeplot(day_tips["acceleration"], ax=g.ax_marg_y, vertical=True, legend=False)
    g.ax_joint.plot(day_tips["mpg"], day_tips["acceleration"], "o", ms=5)


# Let's see what we get when we do the clustering!
# 
# Any predictions?

# In[25]:


kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(X2)
y_kmeans = kmeans.predict(X2)
centers = kmeans.cluster_centers_
sns.scatterplot(X2['mpg'], X2['acceleration'], hue=y_kmeans,palette='Accent')


# Very evenly split groups, which doesn't quite reflect the actual groups

# Here are some calculations to see how accurate we were

# In[26]:


print(sum((y_kmeans==1) & (mpg['origin']=='usa')))
print(sum((y_kmeans!=1) & (mpg['origin']=='usa')))


# In[27]:


print(sum((y_kmeans==0) & (mpg['origin']=='europe')))
print(sum((y_kmeans!=0) & (mpg['origin']=='europe')))


# In[28]:


print(sum((y_kmeans==2) & (mpg['origin']=='japan')))
print(sum((y_kmeans!=2) & (mpg['origin']=='japan')))


# Honestly, not too bad! 
