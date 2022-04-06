#!/usr/bin/env python
# coding: utf-8

# Lecture 8 Missing data and duplicates
# 
# Announcements
# 1. Problem set 4 and quiz 4 due at the end of the week
# 2. Quiz 5 will be a check on (1) you picking a dataset and (2) you coming up with a question you want to explore
# 
# Today's topics
# 1. Finding missing data
# 2. Replacing/removing missing data
# 3. Removing duplicates 
# 
# #### Today, we will do some simple cleaning of some data by looking for missing data and duplicate entries
# 
# #### Let's first discuss finding missing data, then we can discuss what to do with it

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# #### Let's import our messed up mpg dataset

# In[2]:


filename = 'messed_up_mpg.xlsx'
data = pd.read_excel(filename)


# #### Let's take a quick look at the head and tail

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.describe()


# #### But what about the categorical columns?
# 
# #### Info tells us about those

# In[6]:


data.info()


# First thing we will do : take care of duplicate data
# 
# Sometimes, we may expect duplicate data. If this was a database on cars from a random sample, we may expect people to have the same car. 
# 
# However, let's assume this is a list of unique cars from different companies. Its less likely that one car has the same extra properties as another. 
# 
# So, duplicates could arise from bugs in the code or could be mistakes when manually coding data. In these cases we want to remove the data.
# 
# In this case, a duplicate thus overrepresents that combination of data
# 
# So, let's remove it

# In[7]:


data.duplicated()


# So duplicates returns boolean values for rows that are duplicated

# In[8]:


any(data.duplicated()) # any rows where all entries are the same


# Let's pull out the duplicates

# In[9]:


data[data.duplicated()]


# lets just look at some of the duplicates

# In[10]:


data[data['weight']==2979] # duplicates doesn't show you all of them


# Alternatively, you can set the input argument in duplicated to be keep=False, which means that all instances of duplicates are marked

# In[11]:


data[data.duplicated(keep=False)]


# We can drop them like so
# 
# Its important to use ignore_index as true in this situation. This reindexes the data to 0,1,2, etc. Otherwise, the dropped duplicates will have missing index values, which could cause problems if you forget about these missing indexes

# In[12]:


data_nodupes = data.drop_duplicates(ignore_index=True) # INDEXES


# Let's double check that that worked

# In[13]:


any(data_nodupes.duplicated())


# Just another quick check

# In[14]:


data_nodupes.shape


# We will focus on just the missing data now
# 
# #### Both the counts in describe and the non-null count in info tell us that we have missing data
# 
# #### We can visually see the null data using a heatmap
# 
# 

# In[15]:


sns.heatmap(data_nodupes.isnull(),cmap=['Green', 'Red']) # added a cmap to make it clearer


# #### We have two options regarding what to do with missing data:
#   1. drop it (remove the rows/columns)
#   2. fill it with another value
# 
# #### The decision to drop or fill may be based on a couple of factors. 
# 
# #### If the missing data is random, its generally okay to drop. 
# 
# #### However, if it is non-random, dropping the data may bias your sample. For example, if you wanted to figure out the average weight of a crowd of people, you may have lots of missing data from people who don't want to reveal their weight. Averaging only those willing to share their weight may be significantly different than what the actual weight is. In situations like this, there isn't much you can do, but perhaps filling in their data would help

# #### Let's talk about dropping it first

# In[16]:


data_nonans = data_nodupes.dropna() # dropna will take care of those pesky nans!


# In[17]:


sns.heatmap(data_nonans.isnull(),cmap=['Green','Green','Red']) 


# #### Looks great! Anyone think about a potential problem with this?

# In[18]:


data_nonans.shape


# #### Let's try to be more specific. Dropna dropped all of the rows with nans!
# 
# #### Let's just drop the rows that are all NaNs
# 
# #### The default is how='any'

# In[19]:


data_nonan_rows = data.dropna(how='all') # how? only drop those with all nans
data_nonan_rows.info()


# In[20]:


sns.heatmap(data_nonan_rows.isnull(),cmap=['Green','Red'])


# #### The Color column seems to be mostly missing data. We could create a 'color' or 'no color' column, but I don't think that would mean anything
# 
# #### Let's drop that column
# 

# In[21]:


data2 = data_nonan_rows.drop(['Color'],axis=1)


# In[22]:


data2.shape


# In[23]:


data2.head()


# #### Now let's back at our heatmap

# In[24]:


sns.heatmap(data2.isnull(),cmap=['Green','Red'])


# #### Those rows with mostly missing data are still pretty problematic
# 
# #### Some just having a single missing value, which probably isn't much of a problem. But some rows have lots of missing data. Let's drop those rows
# 
# #### Thresh says - it must have X many non-nans for it to be kept

# In[25]:


data2_test1 = data2.dropna(thresh = 9)
sns.heatmap(data2_test1.isnull(),cmap=['Green','Red'])


# In[26]:


data2_test1 = data2.dropna(thresh = len(data2.columns)-1) # this way, I dont have to count the number of columns
sns.heatmap(data2_test1.isnull(),cmap=['Green','Red'])


# #### Note that the default axis is 0. We could delete columns that have at least certain numbers of nan values
# 
# 

# In[27]:


data2_test2 = data2.dropna(thresh = len(data2.index)-3,axis=1)
sns.heatmap(data2_test2.isnull(),cmap=['Green','Red'])


# #### Let's just make sure there aren't more than two nans in a single row

# In[28]:


data3 = data2.dropna(thresh = len(data2.columns)-1)
sns.heatmap(data3.isnull(),cmap=['Green','Red'])


# #### Now, we are left with just a few nans. I think for these, let's keep them! 
# 
# #### As I said earlier, dropping/keeping them is dependent on the data. 
# 
# #### let's create an index of those remaining NaN data so we can check to make sure they have been filled in correctly

# In[29]:


is_NaN = data3.isnull() 
row_has_NaN = is_NaN.any(axis=1)


# In[30]:


rows_with_NaN = data3[row_has_NaN]
rows_with_NaN


# #### Let's first try filling in the data 
# 
# #### We can pass in a numerical value

# #### Check those with the index we created earlier

# In[31]:


data3.fillna(0)[row_has_NaN]


# #### But filling the empty spots with 0s is actually a pretty bad idea when 0 is meaningful
# 
# #### We have a couple of options we can fill with instead. See the documentation of fillna for some examples!

# #### One thing you can do that is pretty clever is to fill in the missing data with the average data
# 
# #### Note that this way of doing it applies the mean function individually to each column. So, its not just one mean value replacing all of the empteis

# In[32]:


data3.fillna(data.mean())[row_has_NaN]


# *See any problem with this?*
# 
# There are still NaNs in our categorical column! That makes some sense since what is the mean of coupe sedan. We will address this shortly

# #### If your data is skewed, then the mean will be pulled in the direction of the skew. Its better to use median in those cases

# In[33]:


data3.fillna(data.median())[row_has_NaN]


# just gonna create a df with this fill method

# In[34]:


data3_nans_median = data3.fillna(data.median())


# In[35]:


sns.heatmap(data3_nans_median.isnull(),cmap=['Green','Red'])


# Now that we are left with just those categorical columns, I'd probably just drop those rows. You could do something different (e.g., replace with the most common category, replace by looking at what that category is most likely to be based on weight or mpg or etc).

# In[36]:


data_nonans = data3_nans_median.dropna()


# In[37]:


sns.heatmap(data_nonans.isnull(),cmap=['Green','Green','Red'])


# One last thing - we need to reindex the data. As I mentioned before, when dropping rows, our index column is missing values. Its helpful when there aren't missing gaps in the index, so we will use the following to fix that

# In[38]:


data_nonans.index = np.arange(0,data_nonans.shape[0])


# In[39]:


data_nonans.to_csv('Data_no_nans.csv',sep=',')


# #### We are by no means finished with this dataset. There are a few more things we have to do with it, which we will do next week
