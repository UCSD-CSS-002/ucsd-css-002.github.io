#!/usr/bin/env python
# coding: utf-8

# # Web scraping 1
# 
# Announcements
# 1. Quiz 2 corrections
# 2. Problem set 3, Quiz 3, and discussion board 1 due on Sunday
# 
# Topics
# 1. pd.read_html
# 2. Beautiful soup - simple html 
# 3. Beautiful soup - Babish Recipes
# 4. Adnan speaking
# 
# #### Last class, we did some real simple web scraping using read_html. We are going to talk a little bit about what that function was doing and why it won't work in some cases

# In[1]:


import pandas as pd
import numpy as np


# We will web scrap using Beautiful Soup

# In[2]:


from bs4 import BeautifulSoup # now we get beautiful soup
import requests # need this to talk to a website


# #### Before we work with our website, let's see what soup does to some basic html

# In[3]:


html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""


# In[4]:


soup = BeautifulSoup(html_doc,'html.parser') # get the content


# In[5]:


type(soup)


# #### the beautiful soup object nests the html file for us. 

# In[6]:


print(soup.prettify()) 


# #### There are some specific attributes of the soup object we can look at quickly, such as the title

# In[7]:


print(soup.title)


# We can also call different tags. There's the "a" tag. Note that this only shows us the first instance of the a tag

# In[8]:


soup.a


# In[9]:


soup.p


# We can output the body of the html

# In[10]:


soup.body


# In[11]:


type(soup.body)


# #### In that tag, we can pull out specific parts. In this case, specifically the text

# In[12]:


print(soup.body.text)


# When we use print here, it is organizing the text according to some syntax in the string
# 
# Without print, we can see it is just a string

# In[13]:


soup.body.text


# Since its a string, we can look for certain things in the text

# In[14]:


soup.body.text.find('Lacie')


# If we search for something and it isn't there, it returns a -1

# In[15]:


soup.body.text.find('Garrett')


# But we can change that with some methods, like the replace method which finds the first input and replaces it when the second input :)

# In[16]:


print(soup.body.text.replace('Lacie','Garrett'))


# Let's get back to the html code.
# 
# We can also pull stuff out of the tags

# In[17]:


soup.a


# In[18]:


soup.a.text


# But as stated earlier, we have multiple instances of the a tag

# #### Use find_all to pull out all of the rows that have the a element

# In[19]:


soup.find_all('a')


# In[20]:


soup.find('a') # find by itself just pulls out one instance


# #### We could have also done this by trying to find all of the class "sister". Note that class is a special variable in python and cannot be used here, so we add an underscore

# In[21]:


soup.find_all(class_ = 'sister')


# We can also find all based on the href or id

# In[22]:


soup.find_all(href = 'http://example.com/elsie')


# In[23]:


soup.find_all(id = "link1")


# #### We can pull out all of the names by using 'a' element in a for loop
# 

# In[24]:


for names in soup.find_all('a'):
  print(names)


# In[25]:


for names in soup.find_all('a'):
  print(names.text)  


# We could also print that link that is also present in the html code
# 
# Let's also add the href, which is an example url, which may be useful
# 
# Note that in order to get stuff inside of the tag, we have to use get

# In[26]:


for names in soup.find_all('a'):
  print(names.text)
  print(names.get('href'))


# #### Now that we have a basic feel for it, let's pass in our babish link

# In[27]:


url = 'https://www.bingingwithbabish.com/recipes'; # passing in an html
response = requests.get(url)


# #### We can check to see if this worked by checking the status_code. If its 200, it worked okay

# In[28]:


response.status_code


# In[29]:


soup = BeautifulSoup(response.content,'html.parser') # get the content


# Let's use prettify to check out the html

# In[30]:


print(soup.prettify()) 


# 
# That's a lot of html! it should match what we see on the webpage

# #### Let's look at the html code and figure out what we need to look for to get text about our recipes

# In[31]:


recipes = soup.find_all('div')


# In[32]:


len(recipes)


# Over a thousand! That's a lot
# 
# Let's check the type 

# In[33]:


print(type(recipes))


# Now, let's just try some items in the list. Maybe the first one?

# In[34]:


print(recipes[1].text) # not a recipe. try another number?


# Maybe the 51st item?

# In[35]:


print(recipes[50].text) # here's a recipe


# #### Seems like just seaching by 'div' is going to take too long.

# 
# #### Looks like recipe-title-wrapper seems like a good place to simplify our web scraping
# 
# Note that sometimes, searching by class can be tricky, so its worth playing with things and spend the time to find something that works. In this case, I am not sure why recipe-col doesn't work (perhaps it is not specific enough)
# 

# In[36]:


recipes = soup.find_all(class_="recipe-title-wrapper")


# #### Let's see how many recipes come back when searching for that specific class
# 

# In[37]:


len(recipes)


# #### Cool. that seems more realistic

# #### Let's take a look at the first one to get a feel for what we have

# In[38]:


print(recipes[0].text)


# In[39]:


recipes[0]


# Let's pull out the href from the a tag

# In[40]:


recipes[0].find('a').get('href')


# #### Let's pull out the date, which appears to be under another class

# In[41]:


rec1 = recipes[0].find(class_='published') # not all of them have authors listed with this class!
print(rec1)


# Now we can get the date.... cool!

# In[42]:


print(rec1.text)


# #### Let's now go through all of the recipes and try to find some recipes for ourselves

# In[43]:


for recipes in soup.find_all(class_='recipe-title-wrapper'):
  if recipes.text.lower().find('chicken') > -1:
    print(recipes.text)
    print(recipes.a.get('href'))


# #### In this simple demonstration, we mostly tried to pull out text. Data works the same way. Next class, we will try to extract some data from different tables online to build a dataframe that we can analyze.
