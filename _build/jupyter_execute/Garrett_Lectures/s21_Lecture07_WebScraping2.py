#!/usr/bin/env python
# coding: utf-8

# Web Scraping 2
# 
# Announcements
# 1. Problem set 4 released after class
# 2. Discussion board 2 releasted after class
# 3. Problem set 4 and quiz 4 due at the end of the week
# 4. Problem set strict guide
#   - comments and format are important!
# 5. Github for extra credit
# 6. Final project
#   - Quiz 5 = progress check on final project
# 
# 
# Topics
# 1. More web scraping!
# 2. Quiz 3 review

# Goal for today: let's web scrap information from one source and then use it to web scrap from a different source
# 
# We will go back to babish.com to find some recipes. This time, however, we will start from the main recipe page and then go into each recipe specifically to pull out the ingredients. Maybe this is a good way to see which ingredients are most common?

# In[1]:


from bs4 import BeautifulSoup # now we get beautiful soup
import requests # need this to talk to a website


# We will start from our main recipe page

# In[2]:


url = 'https://www.bingingwithbabish.com/recipes'


# Let's get the html

# In[3]:


response = requests.get(url)


# And now, use beautifulsoup to parse that data

# In[4]:


soup = BeautifulSoup(response.content,'html.parser') # get the content


# Let's just check out the html to remind ourselves of the tags we want

# In[5]:


print(soup.prettify())


# Like last week, we used this class to find the recipes

# In[6]:


recipes = soup.find_all(class_="recipe-title-wrapper")


# Lets just double check things by checking out the first recipe

# In[7]:


recipes[0]


# Now, remember the goal - this is the first recipe on the main recipe page. And within that html, we can pull out the webpage path to go to that webpage
# 
# Let's just do some testing with the first recipe. Later, we will put this in a for loop

# In[8]:


recipes[0].find('a').get('href')


# This webpage is a bit different from our main recipe page. 
# 
# So, let's do some string processing to get our new url. We will use "replace" to replace the part that says "/recipes" with the new url portion
# 

# In[9]:


new_url = url.replace('/recipes',recipes[0].find('a').get('href'))
print(new_url)


# We can copy and paste to check. Looks good! 

# Let's go ahead and try it out

# In[10]:


response = requests.get(new_url)
print(response)


# In[11]:


soup = BeautifulSoup(response.content,'html.parser') # get the content


# And now, let's check out the prettify of this new webpage

# In[12]:


print(soup.prettify())


# It looks like we can pull out the ingredients with this class, but its not the first item in the resultset

# In[13]:


print(soup.find_all(class_ = 'sqs-block-content')[2].prettify())


# perhaps we can use something like this in an if statement?
# 
# Here, we are checking the items in the result set and trying to see if the text has the string 'Ingredients' in it. If it does, it returns the index. If not, then it returns a -1

# In[14]:


soup.find_all(class_ = 'sqs-block-content')[2].text.find('Ingredients')


# Let's just create a block of text with this ingredient list. We want to now find those ingredients

# In[15]:


ingred_block = soup.find_all(class_ = 'sqs-block-content')[2]


# In[16]:


print(ingred_block.prettify())


# We can take advantage of a new tag, in this, 'ul' which is an unordered list in html

# In[17]:


print(ingred_block.find_all('ul')[0].prettify())


# Cool. And inside of the "unordered list" we have a regular "list", which is li
# 
# Here, I am just using find_all to get all of the items in the ul and li. Then, I use the brackets to select specific items in the list. I can pull out the text using text at the end

# In[18]:


ingred_block.find_all('ul')[0].find_all('li')[0].text


# In[19]:


list1 = ingred_block.find_all('ul')[0].find_all('li')[0].text


# Now, we can take advantage of the method "split" to pull out all of the words separated by spaces
# 
# Is this the best way to get all of the ingredients individually? Meh. You could remove all of the numbers, but for now, we will keep it.

# In[20]:


list1.split()


# So now, we have everything we need
# 
# Here is the pseudocode:
# 
# 1. we create a for loop that goes through each recipe
# 2. we pull out the webpage
# 3. we create a new string
# 4. we pull out the html and do requests
# 5. then, we parse the data
# 6. we then create a for loop that goes through the different classes for the ingredients
# 7. we pull out the text
# 8. we then create a big list with all of the ingredients
# 
# Easy!
# 
# Let's see how this may look

# In[21]:


# first, let's initialize a list
ingredients = []

for i in range(0,5): # this loop is just going through the first 5. We will keep it simple

  # for each iteration in the loop, we create a new html string based on the recipe list we created before
  new_url = url.replace('/recipes',recipes[i].find('a').get('href'))

  # now, we get that text from the website
  response = requests.get(new_url)

  # let's check to make sure it worked by using this if statement
  if response.status_code == 200:

    # now, let's turn our text from the website into a format that we can interact with using beautifulsoup
    soup = BeautifulSoup(response.content,'html.parser') # get the content

    # now, let's create a resultset that includes all of the ingredients
    ingred_block_set = soup.find_all(class_ = 'sqs-block-content')

    # in case there is an error, we can check to make sure we have something here
    if len(ingred_block_set) > 0: 
      
      # if we do, cool! Now, because sqs-block-content isn't specific to the ingredients, we have to loop through until we find the ingredient list
      for j in range(0,len(ingred_block_set)):

        # here, if the text has "ingredients" right at the start, we should be good to go
        if ingred_block_set[j].text.find('Ingredients')==0:

          # some recipes have multiple sets of ingredients (why Babish?!), we need to loop through them. Let's pull out the resultset first
          ingred_group = ingred_block_set[j].find_all('ul')

          # again, just checking to make sure we actually have something in the result set
          if len(ingred_group) > 0:

            # okay. Cool. So, now, let's loop through the ingredient groups
            for k in range(0,len(ingred_group)):

              # and now, we should be able to pull out the specific list of ingredients from that group
              ingredient_list = ingred_group[k].find_all('li')

              # just another check
              if len(ingredient_list) > 0:

                # And now, finally, let's loop through all of the ingredients!
                for m in range(0,len(ingredient_list)):

                  # to pull out the specific things, I am using split. This isn't perfect, but its okay!
                  idv_ingredients = ingredient_list[m].text.split()

                  # and once we have split them, I'm just going to append them to our ingredient list
                  for n in range(0,len(idv_ingredients)):

                    ingredients.append(idv_ingredients[n])

                    # All good? Make sense?
              


# We can do a test here and just print the ingredients to make sure it worked.

# In[22]:


ingredients


# Cool! Looks good!

# Now, I want to see what is the most common. You could do it in a few ways. I'm actually going to put this list into a dataframe

# In[23]:


import pandas as pd


# In[24]:


df = pd.DataFrame(ingredients,columns=['ingredients'])


# And now, let's use the method value_counts(). Value_counts will tell me how much each value occurs. That way, I can see which value is most frequent

# In[25]:


df.value_counts().head(30)


# So, wow. That was a journey. That nested for loop is definitely not the most efficient way to get this information. And, we should probably do a bit more processing of the ingredient list themselves so we actually have meaningful information there
# 
# Its a good lesson in (1) how difficult it can be to get some types of data, and (2) that even with some basic techniques, we can extract some cool things from the internet!
