#!/usr/bin/env python
# coding: utf-8

# 
# Announcements:
# 1. Problem set 4 due Tuesday at midnight instead
# 2. Problem set 5 and quiz 5 due at the end of the week
# 3. Remember to attend sections!
# 4. If you are doing the github extra credit, please send me an email with your github account. 
# 
# #### Today, we will be dealing with transforming categorical and ordinal data into numerical data
# 1. Dummy coding
# 2. Binning continuous data into categories
# 3. Outliers
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# #### Load the dataset we saved from last class

# In[2]:


filename = 'Data_no_nans.csv'
inital_data = pd.read_csv(filename,delimiter=',')


# #### Check the data quick

# In[3]:


inital_data.head()


# #### Whoops. We need to take care of that. How do we do that again?

# In[4]:


data = pd.read_csv(filename,delimiter=',',index_col=0)


# In[5]:


data.head()


# #### This is the dataset with no nans, right?
# 
# #### Let's just double check that quickly. 

# In[6]:


data.info() # or describe


# #### Now, let's look at our string variables and convert them into boolean values
# 
# 

# In[7]:


data.head()


# #### let's look at type first
# 
# #### I like to use a countplot here just to visualize the different categories

# In[8]:


sns.countplot(data['Type'])


# #### Seems fine. We can easily convert these two 1s and 0s like so

# In[9]:


sedan_1 = np.where(data['Type']=='Sedan',1,0) # where is really helpful here! its like an if statement in excel


# In[10]:


sedan_1_df = pd.DataFrame(sedan_1,columns=['Sedans_1'])


# #### That seems to work out okay!

# In[11]:


print(sedan_1_df)


# #### Before we concat with our data, let's check out our other categorical variables
# 
# #### Let's do origin next
# 
# 

# In[12]:


sns.countplot(data['origin'])


# #### This is a problem. If we want to convert this to 1s and 0s, how do we do that with 3 categories!?!?!
# 
# We also don't want this to be 0, 1, and 2 because then, there is a "rank" to the data. This isn't ordinal or interval data, it is categorial. So, how do we handle this?
# 
# #### We use dummy coding!

# In[13]:


pd.get_dummies(data['origin'])


# #### What dummy coding has done is that now we have converted our origin column into numerical (boolean) values across multiple columns
# 
# #### but this is a little bit inefficient because we you don't need all of those columns
# 
# #### in this case, if europe = 0 and japan = 0, then USA must be 1
# 
# #### Instead of creating this dataframe and then dropping that column, we could drop_first = True

# In[14]:


origin_df = pd.get_dummies(data['origin'],drop_first=True)
print(origin_df)


# #### Now that we have done that for origin, let's move on to the names
# 
# #### We will need to dummy code these data as well, but let's look at it first

# In[15]:


sns.countplot(data['name'])


# #### Kind of hard to see. Let's increase the figure size

# In[16]:


plt.figure(figsize=[12,4])
sns.countplot(data['name'])


# #### Looks like some names have been inputted incorrectly!
# 
# #### Let's fix this with a for loop

# In[17]:


name_list = []
for name in data['name']:
  if name=='chevroelt':
    name_list.append('chevrolet')
  elif name=='toyouta':
    name_list.append('toyota')
  elif name=='vokswagen':
    name_list.append('volkswagen')
  else:
    name_list.append(name)


# Here is a different method using np.where which would be quicker than the for loops, especially when you have lots of data

# In[18]:


#data['name'] = np.where(data['name']=='chevroelt','chevrolet',data['name'])
#data['name'] = np.where(data['name']=='toyouta','toyota',data['name'])
#data['name'] = np.where(data['name']=='vokswagen','volkswagen',data['name'])


# #### Let's try this again

# In[19]:


plt.figure(figsize=[12,4])
sns.countplot(name_list)


# #### Much better!
# 
# #### Now, let's dummy code these

# #### Can someone tell me how to dummy code this?

# In[20]:


name_df = pd.get_dummies(name_list,drop_first=True)
print(name_df)


# #### Now, we are going to turn one of our continuous columns into binned data
# 
# #### Sometimes, this is called bucketing
# 
# #### This is especially useful if you have more possible data points than actual data. Alternatively, this basically smooths the data a bit, which means you have less precise data but it may help with analyses
# 
# #### Let's do horsepower

# In[21]:


sns.distplot(data['horsepower'])


# #### We want to bin this into low, medium, and high horsepower
# 
# #### We could bin the data in a couple different ways: 
# 1. bins are equally separate
# 2. bins have equal amounts of data inside of them

# In[22]:


equal_bin_size = pd.cut(data['horsepower'],3) # split data into 3 bins of equal separation
print(equal_bin_size)


# In[23]:


equal_bin_size.unique()


# #### but one big limitation of this is that look = there are very few data points in the largest bin

# In[24]:


equal_bin_size.value_counts()


# #### Let's try creating the bins such that they contain roughly the same number of data
# 
# #### For that, we will use qcut instead

# In[25]:


equal_bin_counts = pd.qcut(data['horsepower'],3) # split data into 3 bins of equal separation
print(equal_bin_counts)


# In[26]:


equal_bin_counts.value_counts()


# #### If you want to define your own bins, that is certainly an option
# 
# #### Maybe I want horsepower between 0 and 100, 100 to 150, and then anything above 150

# In[27]:


garrett_bins = pd.cut(data['horsepower'],[0,100,150,data['horsepower'].max()])
print(garrett_bins)


# #### I think I'm more of a fan of equally populated bins
# 
# #### Now, we could do the same dummy coding as before. Here, we will pass in our cut values!

# In[28]:


horsepower_df = pd.get_dummies(equal_bin_counts,drop_first=True)
print(horsepower_df)


# #### Just gonna rename these columns so that it makes sense later when I look at the data

# In[29]:


horsepower_df.columns = ['Medium_HP','High_HP']


# In[30]:


horsepower_df.head()


# #### At this point, what have we turned horsepower from ratio to what kind of data?
# 
# #### nominal! Categorical!
# 
# #### But these seem to be ordinal data, right?
# 
# #### Well, you could treat them as categorical. That is okay, but you lose information about the order. Alterantively, we could give each group a rank (1,2,3), but most models will just treat those ranks as interval data. So, there are tradeoffs.
# 

# 
# #### Outliers refer to data that are basically further away from the rest of the data
# 
# #### just like missing data or duplicates, these could be real. For example, if you looked at wealth, I'm sure Bezos and some other billionaries would show up as outliers. They could be errors. For example, if you accidentally added another 0 to the zero
# 
# 

# In[31]:


data.describe()


# So, we can see some outliers with just describe. It looks like we have some horsepower values less than 0, which doesn't make sense! 
# 
# You could selectly find these values using boolean logic and remove them. That would totally be justified. 
# 
# Instead, let's use some established methods for outlier removal that may catch some unexpected outliers 

# #### Visualizations will help us see if we have outliers. Especially boxplots

# In[32]:


fig,axes = plt.subplots(7,1,figsize=(12,12))

count = 0
for column in data.columns:
  if isinstance(data[column][0],float)==1:
    sns.boxplot(x = data[column],ax=axes[count])
    count = count + 1

plt.tight_layout()


# #### When we look at boxplots, we are checking for uni-variate outliers - which are outliers along one dimension
# 
# #### These are outliers because they are far away from the other values
# 
# #### You will typically see a couple different ways of finding these outliers
# 
# 1. multiple of the standard deviation
# 2. outside of IQR which is basically what the boxplot is doing 
# 
# #### Both methods work similar. Use dispersion of the data to figure out what is too far away from the other data points. We thus create thresholds. Outliers are data beyond those thresholds 

# In[33]:


temp_acc = data['acceleration'].copy()


# In[34]:


sns.boxplot(temp_acc)


# In[35]:


threshold1_upper = temp_acc.mean()+2*temp_acc.std()
threshold1_lower = temp_acc.mean()-2*temp_acc.std()
print(threshold1_upper,threshold1_lower)


# #### Now that I have my thresholds created with the standard deviation, I can remove data that is either less than or greater than those thresholds

# In[36]:


temp_acc_std = temp_acc[(temp_acc < threshold1_upper) & (temp_acc > threshold1_lower)]
temp_acc_std.shape


# In[37]:


fig,axes = plt.subplots(2,1)
sns.boxplot(temp_acc,ax=axes[0])
sns.boxplot(temp_acc_std,ax=axes[1])
axes[0].set_xlim([6,27])
axes[1].set_xlim([6,27])


# #### Let's try the second method with the IRQ

# In[38]:


Q1 = temp_acc.quantile(.25)
Q3 = temp_acc.quantile(.75)
IQR = Q3-Q1
print(IQR)


# In[39]:


temp_acc_iqr = temp_acc[(temp_acc > (Q1 - 1.5 * IQR)) & (temp_acc < (Q3 + 1.5 * IQR))]
temp_acc_iqr.shape


# In[40]:


fig,axes = plt.subplots(3,1)
sns.boxplot(temp_acc,ax=axes[0])
sns.boxplot(temp_acc_std,ax=axes[1])
sns.boxplot(temp_acc_iqr,ax=axes[2])
axes[0].set_xlim([6,27])
axes[1].set_xlim([6,27])
axes[2].set_xlim([6,27])
plt.tight_layout()


# #### First thing to note - 2 std takes away more data than the IQR method. You could do 2.5 std or 3 std. The other thing to note is that the median is pretty much the same in all cases.

# #### Note that the outlier procedure is done once. So while there may be some "new" outliers that appear after removing outliers, that is okay! 

# #### Now, let's remove the outliers from the other columns
# 
# #### Any guesses on how we can do that? 
# 
# #### I'm suggesting a for loop like we made before!

# In[41]:


for column in data.columns:
  if isinstance(data[column][0],float)==1:
    Q1 = data[column].quantile(.25)
    Q3 = data[column].quantile(.75)
    IQR = Q3-Q1

    index = (data[column] > (Q1 - 1.5 * IQR)) & (data[column] < (Q3 + 1.5 * IQR))
    data[column] = np.where(index, data[column],np.nan)


# In[42]:


sns.boxplot(data['acceleration'])
axes[0].set_xlim([6,27])


# In[43]:


sns.heatmap(data.isnull(),cmap=['Green','Red'])


# #### So now, our outliers have been replaced with nans. looks good!
# 
# #### just to reiterate, we could have set those outliers to the median or mean values

# #### We can check for bi-variate outliers by plotting scatterplots
# 
# #### Instead of showing all scatterplots at once, let's just focus on one scatterplot

# In[44]:


sns.scatterplot(x= 'weight',y = 'displacement',data=data)


# #### The datapoint on the far right appears to be pretty far from the others. We can quickly see this with the sns.lmplot

# In[45]:


sns.lmplot(x= 'weight',y = 'displacement',data=data)
plt.xlim([1500,5500])


# #### Here, seaborn is plotting a best fit line. If that data point is far from others, it may be considered an outlier with high leverage. If that outlier is actually pulling that line towards it, then it may be considered influential.
# 
# #### For now, we will leave these datapoints, but it is certainly something to be aware of and something we may touch on once we are discussing regression
# 
# 

# #### Now, let's just fill our data with medians
# 
# #### Anyone remember how?

# In[46]:


data2 = data.fillna(data.median())


# Now, let's concatenate this with our dummy coded columns

# In[47]:


final_data = pd.concat([data2,sedan_1_df,origin_df,name_df,horsepower_df],axis=1)


# Let's check to make sure that works

# In[48]:


final_data.head()


# And let's describe() and info() as well to double check some things

# In[49]:


final_data.describe()


# In[50]:


final_data.info()


# #### So, just to recap. Here was our procedure for engineering our features (columns)
# 
# 1. Remove mostly empty column = given that with so much missing data, the column appeared useless
# 2. Remove mostly empty rows = given that with so much missing data, the rows appeared useless
# 3. Replaced missing data with median = since median is less affected by outliers
# 4. Removed duplicates = since we shouldn't have any in this dataset
# 5. Dummy coded categorical columns = so we can analyze them later
# 6. Binned Horsepower = to examine how binning affects analyses
# 7. Removed clearly erroneous data = since we know our data is ratio and data below 0 cannot occur
# 8. Removed outliers with IQR = since we had some outliers in the data

# #### This is just one example of how to clean and transform the data. We could have done these steps in a slightly different order. We could have done more than this (e.g., transforming skewed data, remove bivariate outliers, bucket horsepower then make it numerical/ordinal, etc)

# In[51]:


final_data.to_csv('fixed_mpg.csv')


# #### Now that the data is saved, we can being to analyze it next week or so!
