#!/usr/bin/env python
# coding: utf-8

# # This week
# 
# - **no lab assignments**.  Labtime spent on going over past assignments and problem sets as you need (i.e., one question from a while back was: can we go over previous assignments to find efficient ways to do things, that will be our agenda).
# 
# - one problem set, which is also the final, due Saturday at midnight (to be released later today)

# # Unsupervised learning
# 
# General formulation: we have complicated data, and we would like the computer to come up with a *simpler* representation of the data while retaining all the important information.
# 
# The most common varieties are dimensionality reduction and clustering.
# 
# Tradeoff: simpler representation vs more information retained.

# # Dimensionality reduction
# 
# Goal: find a low dimensional numeric representation of some originally high-dimensional, numeric data.
# 
# We will talk about PCA in particular, although there are many different models for dimensionality reduction.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Principal Component Analysis
# 
# Goal: find orthogonal directions in original feature space, in order of how much variance they can capture.

# In[2]:


from sklearn.decomposition import PCA


# In[3]:


bf = pd.read_csv('bodyfat.csv')
bf = bf[['height', 'weight']]


# In[4]:


bf = bf[bf['height']>30]
bf


# In[5]:


plt.scatter(bf['height'], bf['weight'])
plt.xlabel('height')
plt.ylabel('weight')


# In[6]:


pca = PCA()
pca.fit(bf)


# In[7]:


[f for f in dir(pca) if f[0].isalpha()]


# In[8]:


pca.n_components_


# In[9]:


pca.components_


# In[10]:


pca.explained_variance_


# In[11]:


pca.explained_variance_ratio_


# In[12]:


bf.var()


# ### Scaling
# 
# How do we compare variance across features?  Why do we assume lbs^2 are the same units as inches^2?  that doesnt make sense.
# 
# $z_i = (x_i - \bar x)/s_x$

# In[13]:


bf.mean()


# In[14]:


m = bf.mean()
s = bf.std()


# In[15]:


for c in ['height', 'weight']:
    newc = c + '_z'
    bf[newc] = (bf[c]-m[c])/s[c]
    


# In[16]:


bf


# In[17]:


plt.scatter(bf['height_z'], bf['weight_z'])

plt.xlabel('height (z-score)')
plt.ylabel('weight (z-score)')


# In[18]:


bf.var()


# In[19]:


features = ['height_z', 'weight_z']
pca = PCA()
pca.fit(bf[features])


# In[20]:


pca.components_


# In[21]:


pca.explained_variance_ratio_


# In[22]:


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(bf['height_z'], bf['weight_z'])

for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)

plt.xlabel('height (z-score)')
plt.ylabel('weight (z-score)')

plt.axis('equal')


# In[23]:


pca.transform(bf[features].iloc[0:3,:])


# In[24]:


bf.iloc[0:3,:]


# In[25]:


pca_locations = pca.transform(bf[features])
plt.scatter(pca_locations[:,0], pca_locations[:,1])
plt.xlabel('position on first PC')
plt.ylabel('position on second PC')


# ### null simulation for p values

# In[26]:


null_variance_explained = []
for sim in range(100):
    height = np.random.random(100)
    weight = np.random.random(100)
    fake_bf = pd.DataFrame({'h':height, 'w':weight})
    fake_pca = PCA()
    fake_pca.fit(fake_bf)
    null_variance_explained.append(fake_pca.explained_variance_ratio_[0])


# In[27]:


plt.hist(null_variance_explained)


# In[28]:


fake_pca.explained_variance_ratio_


# ### Pokemon

# In[29]:


poke = pd.read_csv('Pokemon.csv')
poke


# In[30]:


feature_list = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
poke[feature_list].var()


# In[31]:


pca = PCA()

pca.fit(poke[feature_list])


# In[32]:


plt.plot(pca.explained_variance_ratio_)
plt.xlabel('pc')
plt.ylabel('proportion of variance explained')


# In[33]:


plt.barh(feature_list, pca.components_[0])
plt.xlabel('weight from component')
plt.title('PC1')


# In[34]:


plt.barh(feature_list, pca.components_[1])
plt.xlabel('weight from component')
plt.title('PC2')


# In[35]:


pca_location = pca.transform(poke[feature_list])

pca_locations = pd.DataFrame(pca_location, columns = ['PC'+str(i+1) for i in range(6)])


# In[36]:


pca_locations['Name'] = poke['Name']


# In[37]:


pca_locations.sort_values('PC2')


# In[38]:


plt.scatter(pca_locations['PC1'], pca_locations['PC2'])


# ### measuring information retained

# In[39]:


poke[feature_list]


# In[40]:


poke[feature_list] - pca.inverse_transform(pca_locations.iloc[:,0:6])


# In[41]:


((poke[feature_list] - 
  pca.inverse_transform(pca_locations.iloc[:,0:6]))**2).mean()


# In[42]:


pca = PCA(n_components = 3)
pca.fit(poke[feature_list])
pca_locs = pca.transform(poke[feature_list])


# In[43]:


((poke[feature_list] - pca.inverse_transform(pca_locs))**2).mean()


# In[288]:





# In[44]:


((poke[feature_list] - pca.inverse_transform(pca_locs))**2).mean()


# In[45]:


((poke[feature_list] - pca.inverse_transform(pca_locs))**2).mean() / (poke[feature_list]**2).mean()


# In[46]:


poke[feature_list] - pca.inverse_transform(pca_locs)


# In[ ]:





# PCA: principal component analysis
# 
# we are learning new "directions" in the original feature space, and we can project our data points onto those directions.
# 
# 

# ## OSRI

# In[47]:


osri = pd.read_csv('osri-data.csv', sep = '\t')


# In[48]:


osri.columns


# In[49]:


features = ['Q'+str(n) for n in range(1, 45)]
features


# In[50]:


pca = PCA()
pca.fit(osri[features])


# In[51]:


plt.plot(pca.explained_variance_ratio_)


# In[52]:


plt.figure(figsize = (5, 12))
plt.barh(features, pca.components_[0])


# In[53]:


osri['gender'].unique()


# In[54]:


pc1 = pca.transform(osri[features])[:,0]


# In[55]:


colors = {0:'red', 1:'blue', 2:'yellow', 3:'green'}
for gender in osri['gender'].unique():
    plt.hist(pc1[osri['gender']==gender], color=colors[gender], 
             alpha=0.5, bins=100)
plt.xlabel('PC1')


# In[56]:


correlations = np.zeros((len(features), len(features)))
for f1 in features:
    for f2 in features:
        correlations[features.index(f1), 
                     features.index(f2)] = np.corrcoef(osri[f1], osri[f2])[0,1]


# In[57]:


pd.DataFrame(correlations, columns = features, index=features)


# In[58]:


plt.imshow(correlations, )


# In[59]:


get_ipython().run_line_magic('pinfo', 'plt.imshow')


# In[60]:


pca = PCA()
pca.fit(osri[features])


# In[61]:


pca.explained_variance_ratio_


# In[62]:


plt.figure(figsize=(5, 12), dpi=80)
plt.barh(features, pca.components_[0])

