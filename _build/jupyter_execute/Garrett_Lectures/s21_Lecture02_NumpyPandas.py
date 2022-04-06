#!/usr/bin/env python
# coding: utf-8

# #Using NumPY and Pandas
# 
# Today will mostly be a refresher in using some basic Python and NumPy, which is useful for scientific computing and creating arrays
# 
# Due this week:
# 1.   Quiz 1 due on Sunday (released after class on Canvas)
# 2.   Problem set 1 due on Sunday
#     
#     a. question -> should the best problem set from the students be uploaded?
# 
# We will discuss:
# 
# 1.   Creating arrays
# 2.   Indexing arrays
# 3.   Applying functions and methods to those arrays
# 4.   Series in Pandas
# 5.   Dataframes in Pandas
# 
# 
# We start with importing numpy as np, which is the conventional way to import the numpy library. 
# 
# Numpy adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays

# In[1]:


import numpy as np


# First thing we will do is create an array. An array is just a structure that has the same value types. Typically, we will deal with integers or floating point numbers

# In[2]:


list1 = [1, 2, 3, 4, 5]
arr1 = np.array(list1)
arr1 = np.array([1,2,3,4,5]) # same thing


# We can see what that array looks like using print
# 
# 

# In[3]:


print(arr1)


# We can access different parts of this array by indexing into it
# 
# Just remember that Python is 0-indexed, which means the first item in the array is the 0th item

# In[4]:


arr1[0]


# In[5]:


arr1[1]


# *What would arr1[3] show us? *

# Here, we are slicing, which just accesses multiple portions of that array. Kind of like slicing a pie

# In[6]:


arr1[0:2]


# In[7]:


arr1[0:3]


# *Will the next line of code break?*
# 
# Negative indexing is helpful if you don't know how large your array is

# In[8]:


arr1[-1]


# In[9]:


arr1[-5:]


# It may also be helpful to pass in an index as a list

# In[10]:


arr1[[-1,-2,-3]]


# In[11]:


arr1[[2,4,3]]


# In[12]:


arr1[[-2,3]]


# In case indexing seems kind of strange, check out this link
# 
# https://www.quora.com/What-is-negative-index-in-Python
# 
# It is worth your time to become familiar with slicing and indexing because often many of bugs simply result from incorrect slicing/indexing

# With an array, there are various methods/functions we can use to interact with our data

# In[13]:


print(arr1.shape) # doesn't need parentheses because it is a tuple, which is like a list that cannot be changed (immutable)
print(np.shape(arr1))


# In[14]:


print(arr1.mean())
print(np.mean(arr1))


# In[15]:


arr1.max()
np.max(arr1)


# As I mentioned before, numpy has mathematical functions we can use on our data
# 
# It may be helpful to log tranform or sqrt the values within your array

# In[16]:


np.log(arr1)


# In[17]:


np.sqrt(arr1)


# We can also reshape the array, which will may be useful when combining arrays that have different dimensions

# In[18]:


arr1.reshape(5,1)
np.reshape(arr1,[5,1])


# In[19]:


arr1.transpose() # transpose is another option, though this doesn't work on 1-dimensional arrays


# We can add the shape method after we reshape to see the new size of our dimensions
# 
# 5 rows, 1 column

# In[20]:


arr1.reshape(5,1).shape


# The shape of your data is something to be mindful of as it tells you how much data you have and the orientation of that data

# Of course, you can also do operations to that array, presuming the array contains integers or floating point numbers

# In[21]:


arr1 * 2


# In[22]:


arr1 ** 2 ## squaring it


# In[23]:


arr1+1


# Let's take that last code cell and see what happens if I index into it.
# 
# *If I do arr1[0], will it return a 1 or 2?*

# In[24]:


arr1[0]


# Of course, adding to that array does not replace that array with the current values
# 
# If I do my operation, I can create a new array with the new values

# In[25]:


new_arr1 = arr1+1
print(new_arr1)


# Let's create a slightly more complicated array
# 
# We are basically passing it multiple lists that are treated as separate rows

# In[26]:


arr2 = np.array([[1,2,3],[4,5,6]])
print(arr2)


# *What shape will arr2 be?*

# In[27]:


print(arr2.shape)


# And you can index into the tuple to retrieve the number of items in the row or column

# In[28]:


arr2.shape[1]


# We can then apply those previous functions, but along different axes

# In[29]:


arr2.sum()


# In[30]:


arr2.sum(axis=0) # axis 0 = rows


# In[31]:


arr2.sum(axis=1) # axis 1 = columns


# Numpy is also useful for creating empty or random arrays
# 

# In[32]:


np.zeros((1,2))


# In[33]:


np.ones((2,2))


# You can also make arrays of random integers

# In[34]:


np.random.randint(0,10,[2,2]) # input arguments = low, high, size of the array


# And arrays of random numbers

# In[35]:


np.random.randn(2,3) # input arguments = size of the array


# We can perform operations on these arrays
# 
# I'll just create a deep copy of arr2
# 

# In[36]:


arr3 = arr2.copy()
print(arr3)


# In[37]:


print(np.add(arr2,arr3))
print(arr2+arr3)


# In[38]:


print(np.multiply(arr2,arr3))


# What you do think will happen here?

# In[39]:


arr2+np.ones((2,3))


# When performing operating using two arrays, they need to have at least one dimension in commmon
# 
# This takes advantage of something called "broadcasting" where python is basically doing an extra step of basically repeating the smaller array until it matches the dimension shape of the larger array

# In[40]:


print(arr2+np.ones((2,1)))
print(arr2+1)


# One thing that will come in handy often is the ability to turn our data into boolean values (i.e., either 0 or 1)

# In[41]:


1 > 0


# In[42]:


1 < 0


# In[43]:


1 == 0


# In[44]:


1 != 0


# We can make a boolean array from an array

# In[45]:


arr5 = np.array([1,2,3,4,5,6,7,8])
print(arr5)


# In[46]:


boolean = arr5 > 4
print(boolean)


# In[47]:


boolean = arr5 >= 4
print(boolean)


# When we slice the original array with the boolean array, it will return just the portions of the boolean labelled True

# In[48]:


arr5[boolean] # >= 4


# Slicing via a boolean array is a very useful way of accessing data and will be especially important in this class

# Now, that was a basic crash course in numpy. Let's quickly discuss pandas now. Pandas is a library for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series
# 
# Let's import pandas as pd

# In[49]:


import pandas as pd


# In[50]:


arr5 = np.array([8.3,3.9,2.7,2.3,1.6])
print(arr5)


# This new aray is nice, but each of these values has some meaning

# Let's also create a list with these items

# In[51]:


city_list = ['NYC','LA','Chicago','Houston','Phoenix']


# And let's create a series with that array and use the city list as the index
# 
# These numbers correspond to the city's population (millions)

# In[52]:


ser = pd.Series(data=arr5,index=city_list)
print(ser)


# This looks pretty similar to arrays, except that on the left, you have those city names. This certainly helps us understand our data a bit better

# Can index into the series just like the array

# In[53]:


print(ser['NYC']) # can index by the index name or by its number
print(ser[0])


# But let's add a couple new cities - including the best city - San diego
# 
# I'm going to create a new series with the number cities and number

# In[54]:


ser2 = pd.Series(data=[1.4],index=['SD']) # important that there are brackets around what you set index to


# In[55]:


print(ser2)


# We can now append or concatenate this series with ser

# In[56]:


ser.append(ser2)


# Of course, just like before, this method does not replace the original one
# 
#  So, let's create a new series
# 

# In[57]:


ser3 = ser.append(ser2)
print(ser3)


# Instead of append, we could have used concat. Many times in this class, we will see multiple ways of performing the same action. I prefer pd.concat to append because you can concat lots of objects instead of just one

# In[58]:


pd.concat([ser,ser2])


# To label this as population, let's put it in a dataframe and make the columns "population"
# 
# A dataframe is just a way to concatenate multiple Series next to each other to create a table
# 
# See the second image (the one with apples and oranges as column headers) here:https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/

# In[59]:


dframe1 = pd.DataFrame(ser3,columns=['Population'])


# In[60]:


print(dframe1)


# You can access specific values from this dataframe by first putting in the column name, then the index

# In[61]:


print(dframe1['Population'])
print(dframe1['Population']['NYC'])
print(dframe1['Population'][0])
print(dframe1[0][0]) # will this last one work? <- this is one difference between a 2d array and a dataframe


# Let's add another column to our dataframe. We can do this in multiple different ways. Let's try concatenating multiple dataframes first. We will first create an array
# 

# In[62]:


arr6 = np.array([301,468,227,637,517,301]) # SD should be 325, but I'm putting it as 301 for demonstration
print(arr6)


# Before, we typed out our list of cities, but we can quickly get them from the other dataframe
# 
# Let's create a new list, but this time, we will make sure its the same as the first dataframe. You can do that by calling the index of the dataframe
# 
# Index, like shape, is an attribute of the dataframe, so it doesn't need parentheses

# In[63]:


city_index = dframe1.index
print(city_index)


# Now, let's create our second data frame

# In[64]:


dframe2 = pd.DataFrame(arr6,index = city_index, columns=['Area']) # data, then index, then columns


# In[65]:


print(dframe2)


# Let's put these dataframes together using concat.
# 

# In[66]:


cities = pd.concat([dframe1,dframe2],axis=1) # axis is defaulted to 0. Important to specify it as 1
print(cities)


# You could do this with the insert method
# 
# Note that when using insert, it DOES take the place of the original array, so I would recommend using concat 

# In[67]:


dframe1.insert(0,'Area',arr6) # (place, name, then the array)


# *Why did 'Area' show up before population?* 

# Let's do a few things with this data frame

# In[68]:


cities.head() ## Really useful for getting a snap shot of the data


# In[69]:


cities.tail() ## bottom 5 indexes


# In[70]:


cities['Area']


# In[71]:


cities['Area']['NYC']


# We can do similar functions to what we did with arrays

# In[72]:


np.sum(cities)
cities.sum()


# In[73]:


np.average(cities)
cities.mean()


# In[74]:


np.max(cities)
cities.max()


# You could also specify axis here if you want to sum over the columns, but that would just be weird

# In[75]:


np.sum(cities,axis=1)


# #### We can find the rank of the values within the different columns. 

# In[76]:


cities.rank()


# #### We can sort the indices

# In[77]:


cities.sort_index()


# #### Describe is also extremely useful and will generally be one of the first things you do when you get a new dataset

# In[78]:


cities.describe()


# In[79]:


cities.info()


# #### It may also be useful to see how many unique values there are

# In[80]:


cities['Area'].unique()


# In[81]:


cities['Area'].nunique()


# In[82]:


cities['Area'].value_counts()


# #### Just like in the arrays, we can turn these dataframes into boolean values using logical operations

# In[83]:


cities['Area']>400


# In[84]:


cities['Population']>2


# #### This may be useful if, for example, you just want to see the population of the cities with large areas
# 
# *What will come out here?*

# In[85]:


cities[cities['Area']>400]


# #### Use can of course change the values of things in the dataframe. DataFrames are mutable 
# 
# #### Here, I'm setting that value equal to the column I'm changing, that way, it changes in the original dataframe

# In[86]:


cities['Population'] = cities['Population']*1000000
print(cities)


# We can pull out just two of the columns in the following way. This will be useful later when we load in a large dataset with many columns, but we only care about a few of those columns

# In[87]:


cities[['Area', 'Population']] 


# Finally, let's remove LA from our dataframe. You could do that in a couple ways
# 
# Drop is probably the easiest

# In[88]:


cities.drop(index='LA')


# You could do something like this, where you create boolean values and find the indices you want to keep

# In[89]:


boolean_index = cities['Area']!=468
print(boolean_index)


# In[90]:


cities2 = cities[boolean_index]
print(cities2)


# One last thing that may be useful is to create a pivot table. Pivot tables are extremely helpful for quickly looking at data in excel and we can create them here as well

# In[91]:


cities3 = cities2.copy()


# In[92]:


cities3.insert(0,'West Coast?',[0,0,0,1,1])


# In[93]:


cities3.head()


# In[94]:


new_city_index = cities3.index
pivot = pd.pivot_table(cities3,values='Population',index=new_city_index,columns='West Coast?')
print(pivot)


# A couple things are pretty obvious - we've keep the indices, are columns are now west coast 0 or 1, and the values in the dataframe are the population. 
# 
# What's not so obvious is that it has taken the mean of these values
# 
# Let me show you by changing the aggfunc to 'count'

# In[95]:


pivot2 = pd.pivot_table(cities3,values='Population',index=new_city_index,columns='West Coast?',aggfunc='count')
print(pivot2)


# Below is an example of how we pivot tables would work with a more complex dataset. Below, I've made multiple copies of the cities

# In[96]:


cities_multi = pd.concat([cities3,cities3*.5,cities3*.2,cities3*.01])
cities_multi['West Coast?'][cities_multi['West Coast?']>0] = 1 # necessary because those multiplications apply to each number in the dataframe
print(cities_multi)


# #### Let's change the aggfunc to count to see that now, there are 4 values contributing to that cell in the dataframe

# In[97]:


new_city_index2 = cities_multi.index
pivot2 = pd.pivot_table(cities_multi,values='Population',index=new_city_index2,columns='West Coast?',aggfunc='count')
print(pivot2)


# Now we can change it to mean, median, etc to quickly look at how groupings affect our data
# 
# For example, imagine you ran a study where 10 participants complicated 50 trials of two different conditions. Using a pivot table, you could quickly examine the mean/median for each subject between the different groups

# In[98]:


pivot2 = pd.pivot_table(cities_multi,values='Population',index=new_city_index2,columns='West Coast?',aggfunc='median')
print(pivot2)
print(pivot2.mean())


# Side note - because these are dataframes, you can do all the stuff above with the pivot table - eg, describe()

# In[99]:


pivot.describe()


# The last thing I want to discuss is that dataframes can do some simple plots

# In[100]:


cities3.plot() # ylim=[0,5]


# In[101]:


cities3.plot(kind='box') 


# In[102]:


cities3['Area'].plot(kind='hist') # histogram


# #### Tomorrow, we will discuss way better ways to visualize your data
