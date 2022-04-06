#!/usr/bin/env python
# coding: utf-8

# # CSS 2 Lecture 2
# 
# ## assignment feedback.
# 
# Sorry for delay: there is a slight technical problem that datahub admin needs to resolve for me, which is precluding me from releasing feedback on Lab 1.  However, I can confirm that I received 12 submissions (which is how many people are enrolled in the class, last I checked), and all got 5/5.  
# 
# ```
# a3xie  arnewlan  bedu  bzekeria  cpretori  jop007  juy003  lmesko  mbelon  miyin  talam  zyc004
# ```
# 
# If you submitted lab 1, and do not see your username on this list, please DM me so we can sort it out.
# 
# After the technical problem is resolved, I will be able to release feedback to you that you can view through the assignment interface

# ## Remainder of Monday
# 
# 
# 
# ### Control structures
# 
# - if statements
# 
# - for loops over (strings, lists, dictionaries)
#    

# In[1]:


x = 3
get_ipython().run_line_magic('whos', '')
y = 5
get_ipython().run_line_magic('whos', '')
print(x)
get_ipython().run_line_magic('whos', '')
x = y**2
get_ipython().run_line_magic('whos', '')
print(x)
get_ipython().run_line_magic('whos', '')


# In[2]:


height = 68
weight = 170
weight_units = 'lbs'

# change height to inches if it is in cm
if height > 120:
    # height is in cm if height > 120
    height = height / 2.54  # height /= 2.54

# change weight to lbs if it is in kgs
if weight_units == 'kgs':
    weight /= 2.2
    


# In[3]:


if condition 1:
    do something
elif condition 2:
    do B
elif condition 3:
    do C
else:
    do D
   

if A:
    do something
    
if B:
    do something else
    
if C:
    do something else


# In[4]:


names = ['ed', 'alice', 'bob', 'carol']

i = 0
for x in names:
    print(i, x)
    i += 1


# In[5]:


for character in 'sentence and stuff':
    print(character)


# In[6]:


heights = {'alice':68, 'bob':70, 'carol':72, 'dave':74}

#for x in heights.values():
#    print(x)


for key,value in heights.items():
    print(key, value)


# In[7]:


sentence = 'sentence and stuff'

print(f'{len(sentence)=}')

for i in range(len(sentence)):
    print(i, sentence[i])


# In[8]:


instructor = 'Ed'
course = 'css2'

print('The instructor of ' + course + ' is ' + instructor)


# In[9]:


print(f'The instructor of {course} is {instructor}')


# In[10]:


print(f'{course=}')


# In[11]:


print('course', course)


# In[12]:


print(f'{len(course)=}')


# ### Abstractions
# 
# - functions

# In[13]:


def addTwoThings(arg1, arg2):
    print(arg1 + arg2)
    return arg1+arg2


# In[14]:


addTwoThings(2, 3)


# In[15]:


x = 10

def addX(num):
    return num + x


# In[16]:


addX(3)


# In[17]:


def addX(num, x):
    return num + x


# In[18]:


addX(3, 11)


# In[19]:


x = 3
y = 10

z = addX(5, y)

print(z)


# In[20]:


addX(x = y, num = 5)


# In[21]:


def divide(a,b):
    return a/b

divide(b = 2, a = 4)


# In[22]:


def divide(a = 1, b=1):
    return a/b

divide()


# - classes

# In[23]:


class Dog:
    def __init__(self, name, weight, sound):
        self.name = name
        self.weight = weight
        self.sound = sound
    
    def speak(self, times):
        print(self.sound * times)

rover = Dog('rover', 50, 'arf')
gus = Dog('gustofer', 100, 'woof')


# In[24]:


rover.speak(3)
gus.speak(4)


# In[25]:


dir(rover)


# In[26]:


' huzzah '.join(['a', 'b', 'c'])


# In[27]:


x = 'sentence'


# In[28]:


x.count('en')


# ### Pythonic peculiarieties
# 
# - list/tuple/set/dictionary comprehension
# 
# - assignment unpacking
# 
# - lambda functions

# # Pandas and data frames
# 
# 
# 
# ## Foundations
# 
# ### Modules

# In[29]:


print(4)


# In[30]:


import math


# In[31]:


dir(math)


# In[32]:


math.cos(3.1415)


# In[33]:


import random as rn


# In[34]:


rn.random()


# In[35]:


from math import log


# In[36]:


log(2.7)


# ### numpy
# 
# - array / vector operations are fast

# In[37]:


import numpy as np


# In[38]:


x = [rn.random() for _ in range(10000)]


# In[39]:


# we want to calculate 1/x for all values of x

y = []
for value in x:
    y.append(1/value)
print(y[:10])


# In[40]:


array_x = np.asarray(x)


# In[41]:


array_x


# In[42]:


array_y = 1/array_x


# In[43]:


print(array_y)


# In[44]:


array_y + array_x


# In[45]:


x = [rn.random() for _ in range(10000)]

get_ipython().run_line_magic('timeit', 'y = [val**2 for val in x]')


# In[46]:


x = np.asarray(x)
get_ipython().run_line_magic('timeit', 'y = x**2')


# - also, super useful for linear algebra, matrices, but we are not going to cover that.
# 
# - supported array operations

# In[47]:


dir(x)


# ## Pandas
# 
# [tutorial](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)
# 
# [cheat sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
# 
# **Note: Committing all functions and their argument structure to memory is a lost cause.  That's not the goal.  The point is to know roughly how the system operates, and how to quickly look up the relevant functions**
# 
# data file we are using: Pokemon.csv.  Download here:
# 
# [https://raw.githubusercontent.com/UCSD-CSS-002/ucsd-css-002.github.io/master/datasets/Pokemon.csv](https://raw.githubusercontent.com/UCSD-CSS-002/ucsd-css-002.github.io/master/datasets/Pokemon.csv)

# In[48]:


import pandas as pd


# In[49]:


dir(pd)


# In[50]:


x = [rn.random() for _ in range(100)]

series_x = pd.Series(x)


# In[51]:


series_x


# In[52]:


series_x = pd.Series(x, index = [chr(i) for i in range(100, 200)])


# In[53]:


series_x


# In[54]:


series_x.iloc[0:4]


# ### Overview of data structures:
# 
# - series structure  (differences from... ndarray? list? dictionary?)  .values and .index
# 
# - dataframe structure  (index and columns, each column a series, kinda like dict, kinda like array)

# In[55]:


poke = pd.read_csv('Pokemon.csv')


# In[56]:


poke


# In[57]:


poke['Total']


# - index... why?

# ### Making a data frame
# 
# - from data already loaded in memory

# In[58]:


heights = [48 + int(36*rn.random()) for _ in range(100)]

def randomName():
    n = ''.join([chr(int(rn.random()*26)+ord('a')) for _ in range(5)])
    return n.title()

names = [randomName() for _ in range(100)]

df = pd.DataFrame({'name':names, 'height':heights})

df


# - from a csv file
# 
# `pd.read_csv('Pokemon.csv')`
# 
# - read_* functions

# ### browsing a data frame
# 
# - head, tail

# In[59]:


poke.tail(2)


# In[60]:


poke.head(2)


# - shape, columns

# In[61]:


poke.shape


# In[62]:


list(poke.columns)


# ### Subsets:
# 
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html

# In[63]:


poke


# #### Extracting columns (series)
# 
# - operations on columns
# 
# 

# In[64]:


poke['Attack']


# #### Extracting rows

# In[65]:


poke.iloc[5:8]


# #### Extracting cells

# In[66]:


poke


# In[67]:


poke['Defense'].iloc[3]


# In[68]:


poke.iloc[3]['Defense']


# In[69]:


poke.iloc[3,7]


# In[70]:


poke.loc[3, 'Defense']


# ### Indexing madness:
# 
# https://pandas.pydata.org/docs/user_guide/indexing.html#indexing-choice
# 
# 
# #### Filtering rows with booleans

# In[71]:


poke[poke['Type 1']=='Grass']


# In[72]:


poke[(poke['Type 1']=='Grass') & (poke['HP'] > 80)]


# In[73]:


poke[(poke['Type 1']=='Grass') | (poke['HP'] > 80)]


# #### loc: index/label based indexing.
# 
# #### iloc: integer position indexing
# 
# - we will avoid using indexes, and will use integer and boolean selection
# 
# 
# ### Grouping
# 
# https://pandas.pydata.org/docs/user_guide/groupby.html#groupby
# 
# - what does grouping do?
# 
# ### Summarizing
# 
# https://pandas.pydata.org/docs/getting_started/intro_tutorials/06_calculate_statistics.html
# 
# we will stick to using `.aggregate('newcol'=('oldcol','function'))`, although many other methods are available.

# In[ ]:




