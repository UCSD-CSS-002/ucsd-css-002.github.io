#!/usr/bin/env python
# coding: utf-8

# # CSS 2 Lecture 1
# 
# ## Preliminaries
# 
# ### CSS
# 
# ### CSS 2 objectives
# 
# ### Website
# 
# ### Syllabus
# 
# ### Campuswire
# 
# ### Datahub
# 
# ## Review of basics
# 
# ### Jupyter notebooks and jupyterhub environment
# 
# - Managing files.
# 
# - Jupyter notebooks
# 
# - Markdown and code cells
# 
# - Fetching assignments
# 
# - Completing assignments (practice assignment)

# # Title 1
# 
# ## Title 2
# 
# ### Title 3
# 
# ##### Title
# 
# kdjfghiouy
# 
# Whatever
# 
# These are words  **bold**  *italics* 
# 
# ### bulleted
# 
# - bullet 1
# 
# - bullet 2
# 
# ### numbered
# 
# 1. number one
# 
# 1. number two
# 
# 1. number three
# 
# [link to python website](https://www.python.org)
# 
# 

# ### Python
# 
# #### Expressions and interpreting
# 
# - what does python do with the stuff you type in?
# 
# function(argument-expression)
# something + something_else

# In[1]:


print(4 + 12)


# In[2]:


x = 'abracadabra' * 3  # assignment expression.  assign ('abracadabra' * 3 ) to x
                        # ('abracadabra' * 3 )  binary * expression  ('abracadabra')  *  3
                        # 'abracadabra', 3.  
                        # * 'abracadabra' 3
                        # ('abracadabra' * 3 ) -> 'abracadabraabracadabraabracadabra'
                        # x is assigned 'abracadabraabracadabraabracadabra'
print(x)


# - variable names

# In[3]:


this_variable_has_a_value = 39

print(this_variable_has_a_value)

# do not do this:
# print = 4


# In[4]:


print(35)


# - operations

# In[5]:


# arithmetic
print(45.0 + 4)
print(45 * 4)
print(45 - 4)
print(3**2)
print(4/2)
print(-3)

# boolean operations
print(True and False)
print(True or False)
print(not False)

# bitwise

# assignment operators

x = 4
x += 1  # shorthand for x = x + 1
# -=, /=, *=, 
print(x)


# In[6]:


# comparison operators

x = 5
print(x == 4)
print(x > 5)
print(x < 5)
print(x >= 5)
print(x <= 5)
print(x != 5)


# In[7]:


# containment / membership

'z' in 'this is a sentence'


# #### Basic data types
# 
# - ints/floats

# In[8]:


type(45)


# In[9]:


type(45.0)


# In[10]:


4/3


# In[11]:


4.0 == 4


# In[12]:


type(int(4.0))


# In[13]:


int(4.9)


# - strings

# In[14]:


x = 'this is a sentence'

'sent' in x


# In[15]:


dir(x)


# In[16]:


x.index('is')


# In[17]:


x


# In[18]:


x[2:4]


# In[19]:


x[0:4]


# In[20]:


x[-1]


# In[21]:


newstring = x.replace('i', 'u')

print(x)
print(newstring)


# In[22]:


list_of_words = x.split(' ')
print(x)
print(list_of_words)


# - lists

# In[23]:


x = [3, 4, 5, 'six', 'seven', 45.6]
print(x)


# In[24]:


x[0:3]


# In[25]:


dir(x)


# In[26]:


print(x)
x.append(8)
print(x)


# In[27]:


print(x)
x.extend(['a', 'b', 'c'])
print(x)


# In[28]:


print(x)
x.append(['a', 'b', 'c'])
print(x)


# In[29]:


x[10][1]


# In[30]:


'sev' in x


# In[31]:


print(x)
x.remove('b')
print(x)


# In[32]:


x.sort()


# In[33]:


list_of_integers = [5, 1, 7, 10, 12]

print(list_of_integers)
list_of_integers.sort()
print(list_of_integers)


# In[34]:


5 < 'alice'


# - dictionaries

# In[35]:


my_dictionary = dict()
my_dictionary['ed'] = 24
my_dictionary['bob'] = 40
my_dictionary['carol'] = 38

print(my_dictionary)


# In[36]:


my_dictionary['ed']


# In[37]:


my_dictionary['ed'] = 102
print(my_dictionary)


# In[38]:


my_dictionary.get('ed',0)


# In[39]:


my_dictionary.get('alice', 0)


# - sets

# In[40]:


string = 'this is a sentence'
list_of_string = list(string)
print(string)
print(list_of_string)
set_of_string = set(string)
print(set_of_string)


# In[41]:


'h' in set_of_string


# In[42]:


print(set_of_string)
set_of_string.add('z')
print(set_of_string)


# In[43]:


print(set_of_string)
set_of_string.remove('t')
print(set_of_string)


# In[44]:


char1 = set('this is a sentence') 
char2 = set('this string describes this class')

# set difference.
char2 - char1


# In[45]:


char1 | char2


# In[46]:


char1 & char2


# int
# float
# str
# list
# dict
# set
# 
# - overloaded operations

# In[47]:


4 + 3


# In[48]:


[1, 2, 3] + ['a', 'b']


# In[49]:


'abra' + 'cadabra'


# In[50]:


'abra' * 3


# In[51]:


type('abra')


# #### files
# 
# - reading in files

# In[52]:


fp = open('nonsense.txt', 'r') # append, write, read, 


# In[53]:


contents = fp.read()


# In[54]:


fp.close()


# In[55]:


contents


# In[56]:


with open('nonsense.txt', 'r') as fp:
    contents = fp.read()


# In[57]:


contents


# # Lab-01

# ### Handling new keys in dictionaries

# #### Option 1: check if key exists, and create if not

# In[58]:


x = dict()
if 'alice' not in x:
    x['alice'] = 0
x['alice'] += 1


# #### Option 2: use default value from get

# In[59]:


x = dict()
 
x['carol'] = x.get('carol',0) + 1


# #### Option 3: use try / except

# In[60]:


x = dict()

try:
    x['ed'] += 1
except KeyError:
    x['ed'] = 1


# ## Finding non-alphabetic characters.

# In[61]:


string = 'SFMNB GFH SDG(SYD (*FG HJG(&^SDFKBSDBFSD*&*SAGVJHAFS&^A FKJ DSF*^ SDJHV DSF^ DF)))'
string = string.lower()


# #### Option 1: set difference:
# 
# find the set of all characters, subtract the set of all characters you want to keep

# In[62]:


all_characters = set(string)
characters_i_like = set(' abcdefghijklmnopqrstuvwxyz')
characters_i_want_to_replace = all_characters - characters_i_like  # set difference
print(characters_i_want_to_replace)
# now just loop through all those characters that you dont like, and str.replace() them.


# #### Option 2: regular expressions
# 
# We have not (and will not) cover regular expressions, but this is an ideal regular expression problem:  we want everything that is not a letter or whitespace, so we use the `[^` pattern.  

# In[63]:


import re

re.sub(r'[^a-z ]', ' ', string)

