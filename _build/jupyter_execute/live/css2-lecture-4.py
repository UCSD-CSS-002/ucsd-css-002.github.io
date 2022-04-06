#!/usr/bin/env python
# coding: utf-8

# # Lecture 4: Many more complicated data things
# 
# - logarithms review
# 
# - bivariate summary statistics, and groupby.apply
# 
# - joining data frames
# 
# - melting/pivoting long-to-wide and wide-to-long

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


gapminder = pd.read_csv('../datasets/gapminder.csv')
# download here:
# https://raw.githubusercontent.com/UCSD-CSS-001/ucsd-css-001.github.io/main/datasets/gapminder.csv

gapminder = gapminder.drop(columns = ['Unnamed: 0'])


# In[3]:


gapminder


# ## Summary statistics
# 
# ### the wonders of logarithms.  

# In[4]:


f = plt.hist(gapminder[gapminder['year']==2007]['pop'])
f = plt.xlabel('population')
f = plt.ylabel('count of countries')
f = plt.title('Histogram of country populations')


# In[5]:


gapminder[gapminder['year']==2007]['pop'].median()


# Yikes!  Most standard statistics do not work well with data like this.  Standard statistics are designed for roughly normal (i.e., gaussian, bell-shaped) distributions.   
# 
# Fortunately, lots of variables, like population, wealth, income, gdp, etc. are roughly log-normally distributed.  Meaning their *logarithm* is normal.
# 
# So we can do standard statistics with them after log-transforming

# In[6]:


freq, bins, _ = plt.hist(gapminder[gapminder['year']==2007]['pop'],
                        bins = np.logspace(5, 9.2, num=15, base=10))
f = plt.xlabel('population')
f = plt.ylabel('count of countries')
f = plt.title('Histogram of country populations')
f = plt.xscale('log')


# In[7]:


np.logspace(5, 9, num=10, base=10)


# ### mean, median, mode, stdev, var, etc.

# In[8]:


populations_in_2007 = gapminder[gapminder['year']==2007]['pop']


# In[9]:


print(populations_in_2007.mean()) # sum / count
print(populations_in_2007.median()) # number such that 50% are higher and 50% are lower


# In[10]:


print(populations_in_2007.mode()) # does not make sense for numbers.


# In[11]:


gapminder['continent'].mode() # makes sense for categorical


# In[12]:


populations_in_2007.std()


# In[13]:


populations_in_2007.var()**(1/2)


# #### weighted statistcs

# In[14]:


populations_in_2007.mean()


# In[15]:


def custom_mean(nparray):
    return sum(nparray)/len(nparray)


# In[16]:


custom_mean(populations_in_2007)


# In[17]:


gm_2007 = gapminder[gapminder['year']==2007]
gm_2007


# In[18]:


gm_2007['gdpPercap'].mean()


# In[19]:


x = [1, 2, 3, 4]
print(len(x))
print(sum([1, 1, 1, 1]))


# In[20]:


def weighted_mean(values, weights):
    # works for numpy arrays.
    return sum(values * weights) / sum(weights)


# In[21]:


x = [1, 2, 3, 4]
w = [1, 1, 1, 10]
# sum: 46 / denominator: 13
# [1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
weighted_mean(np.array(x), np.array(w))


# In[22]:


# not the same
x = [1, 2, 3, 40]
w = [1, 1, 1, 1]
# sum = 46 denominator: 4
weighted_mean(np.array(x), np.array(w))


# In[23]:


# population-weighted average of gdp per capita
weighted_mean(gm_2007['gdpPercap'], gm_2007['pop'])


# ### bivariate statistics
# 
# consider life expectancy ~ gdp per capita

# In[24]:


gm_2007


# In[25]:


plt.scatter(gm_2007['gdpPercap'], gm_2007['lifeExp'])
plt.xscale('log')
plt.xlabel('gdp per capita')
plt.ylabel('life expectancy in 2007')


# In[26]:


# why we are log transforming gdp
f = plt.hist(np.log10(gm_2007['gdpPercap']))


# #### correlation
# 
# we want to calculate the correlation between log10(gdpPercap) and life Exp

# In[27]:


def cor(x,y):
    mx = x.mean()
    my = y.mean()
    sx = x.std()
    sy = y.std()
    zx = (x-mx)/sx
    zy = (y-my)/sy
    return (zx*zy).sum() / (len(zx)-1)


# In[28]:


cor(np.log10(gm_2007['gdpPercap']), gm_2007['lifeExp'])


# In[29]:


x = np.log10(gm_2007['gdpPercap'])
y = gm_2007['lifeExp']
np.corrcoef(x, y)[0,1]


# In[30]:


gm_2007['lifeExp'].corr(np.log10(gm_2007['gdpPercap']))


# In[31]:


# r^2, coefficient of determination, 
gm_2007['lifeExp'].corr(np.log10(gm_2007['gdpPercap']))**2


# #### regression line

# In[32]:


x = np.log10(gm_2007['gdpPercap'])
y = gm_2007['lifeExp']
fit = np.polyfit(x,y,1)
slope = fit[0]
intercept = fit[1]
print(f'{slope=}, {intercept=}')


# In[33]:


np.corrcoef(x, y)[0,1]


# In[34]:


# predictions from our fitted line
new_x = np.linspace(2, 5.2, 10)
predicted_y = new_x*slope + intercept


# In[35]:


plt.scatter(x, y)
plt.xlabel('log10(gdp per capita)')
plt.ylabel('life expectancy in 2007')
plt.plot(new_x, predicted_y,'r-')


# In[36]:


plt.scatter(gm_2007['gdpPercap'], gm_2007['lifeExp'])
plt.xscale('log')
plt.xlabel('gdp per capita')
plt.ylabel('life expectancy in 2007')
plt.plot(10**new_x, predicted_y, 'r-')

# log10 gdp per cap


# ## Advanced summarization via groupby.apply
# 
# consider life expectancy ~ gdp per capita by continent (by year)

# In[37]:


gapminder


# In[38]:


# calculate correlation of log10(gdpPercap) and lifeExp for each year

gapminder.groupby('year').agg(mean_life = ('lifeExp', 'mean'))


# In[39]:


def summary_function(df):
    return np.corrcoef(np.log10(df['gdpPercap']), df['lifeExp'])[0,1]

for grp_name, grp_df in gapminder.groupby('year'):
    print(grp_name, summary_function(grp_df))


# In[40]:


gapminder.groupby('year').apply(summary_function)


# In[41]:


def summary_function(df):
    r = np.corrcoef(np.log10(df['gdpPercap']), df['lifeExp'])[0,1]
    fit = np.polyfit(np.log10(df['gdpPercap']),df['lifeExp'],1)
    slope = fit[0]
    intercept = fit[1]
    return pd.Series({'correlation':r,
                      'slope': slope})


# In[42]:


summary_function(gapminder)


# In[43]:


gapminder.groupby('year').apply(summary_function)


# ## Joining data frames.
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html#database-style-dataframe-or-named-series-joining-merging
# 
# - inner, outer, left, right

# In[44]:


instructors = pd.DataFrame({'name':['Ed Vul', 'Drew Walker', 'Shannon Ellis', 'Erik Brockbank', 'Judy Fan'],
                            'email': ['evul@ucsd.edu','drew@fake.com', 'shannon@fake.com', 'erik@fake.com', 'judy@fake.com'],
                            'location': ['McGill 5137', 'COGS 563', 'COGS 123', 'Mandler 3509', 'McGill 5139?']})

classes = pd.DataFrame({'name':['CSS 1', 'CSS 2', 'COGS 18', 'PSYC 201', 'PSYC 60', 'COGS 9', 'PSYC 100'],
                        'location': ['mandler', 'mcgill', 'cog sci', 'mcgill', 'peterson', 'cog sci', 'auditorium'],
                       'instructor_name':['Ed Vul', 'Ed Vul', 'Shannon Ellis', 'Ed Vul', 'Judy Fan', 'Drew Walker', 'John Serences']})


# In[45]:


instructors


# In[46]:


classes


# In[47]:


pd.merge(instructors, 
         classes, 
         # either one 'on'
         left_on = 'name', 
         right_on='instructor_name', 
         how='inner', 
         suffixes=['_prof', '_class'])


# In[48]:


instructors = pd.DataFrame({'instructor_name':['Ed Vul', 'Drew Walker', 'Shannon Ellis', 'Erik Brockbank', 'Judy Fan'],
                            'email': ['evul@ucsd.edu','drew@fake.com', 'shannon@fake.com', 'erik@fake.com', 'judy@fake.com'],
                            'office': ['McGill 5137', 'COGS 563', 'COGS 123', 'Mandler 3509', 'McGill 5139?']})

classes = pd.DataFrame({'class_name':['CSS 1', 'CSS 2', 'COGS 18', 'PSYC 201', 'PSYC 60', 'COGS 9', 'PSYC 100'],
                        'location': ['mandler', 'mcgill', 'cog sci', 'mcgill', 'peterson', 'cog sci', 'auditorium'],
                       'instructor_name':['Ed Vul', 'Ed Vul', 'Shannon Ellis', 'Ed Vul', 'Judy Fan', 'Drew Walker', 'John Serences']})


# In[49]:


instructors


# In[50]:


classes


# In[51]:


pd.merge(classes, 
         instructors, 
         on='instructor_name',
         how='inner')


# In[52]:


pd.merge(classes, 
         instructors, 
         on='instructor_name',
         how='left')


# In[53]:


pd.merge(classes, 
         instructors, 
         on='instructor_name',
         how='right')


# In[54]:


pd.merge(classes, 
         instructors, 
         on='instructor_name',
         how='outer')


# ## Reshaping / pivoting / tidy / long-wide transforms
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-melt

# ### Wide data
# 
# ![wide](lotr-wide.png)
# 
# ### Long data
# 
# ![long](lotr-long.png)
# 
# 

# back to gapminder

# In[55]:


gapminder


# ### long to wide
# 
# - unstack (via index)
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-stacking-and-unstacking
# 
# - pivot (via column names)
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-pivoting-dataframe-objects
# 
# - need to deal with multi-index
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#
# 

# **Goal**: See how mean life expectancy in Europe and Asia has changed over the years.  
# 
# **Output:** A scatterplot over time, with x: mean life expectancy in europe, and y: mean life expectnacy in Asia, and each data point is a year.
# 
# **Process:**
# 
# - group by continent and year, and calculate mean life expectancy
# 
# - convert long to wide format, with one column per continent
# 
# 

# In[56]:


summary_data = (gapminder
   .groupby(['continent', 'year'])
   .agg(m_lifeExp = ('lifeExp', 'mean'),
        s_lifeExp = ('lifeExp', 'std')))
summary_data


# In[57]:


# via unstacking by index
summary_data.unstack('continent')


# In[58]:


summary_data['m_lifeExp']['Asia']


# In[59]:


summary_data = (gapminder.groupby(['continent', 'year'])
    .agg(m_lifeExp = ('lifeExp', 'mean'))
    .reset_index())
summary_data


# In[60]:


# via column name and .pivot

summary_data.pivot(index='year',
           columns='continent',
           values='m_lifeExp')


# In[61]:


life_exp = (gapminder.groupby(['continent', 'year'])
                    .agg(m_lifeExp = ('lifeExp', 'mean'))
                    .unstack('continent'))
life_exp


# In[62]:


plt.scatter(life_exp['m_lifeExp']['Europe'],
            life_exp['m_lifeExp']['Asia'])
plt.plot([45, 80], [45, 80], 'r-')
plt.xlabel('Europe life expectancy')
plt.ylabel('Asia life expectancy')


# ### wide to long
# 
# - stacking (with indexes)
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-stacking-and-unstacking
# 
# - melt (with column names)
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-melt

# #### stacking (with indexes)

# In[63]:


life_exp = (gapminder.groupby(['continent', 'year'])
                    .agg(m_lifeExp = ('lifeExp', 'mean'))
                    .unstack('continent'))


# In[64]:


life_exp


# In[65]:


# creates indexes
life_exp.stack()


# #### melting (column names)

# In[66]:


life_exp = (gapminder.groupby(['continent', 'year'])
                    .agg(m_lifeExp = ('lifeExp', 'mean'))
                    .unstack('continent')['m_lifeExp'].reset_index())


# In[67]:


life_exp


# In[68]:


life_exp.melt(id_vars = ('year'),
              value_vars = ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania'],
             var_name = 'continent',
             value_name = 'mean_life_expectancy')


# reshaping:
# 
# with indexes: unstack and stack
# 
# with column names: pivot and melt

# # Let's do something complicated together.

# **Goal**: we want to know how life expectancy has changed with gdpPercapita over time, for the different continents.
# 

# ### Complicated plot
# 
# First: let's make a plot of life expectancy as a function of gdpPerCapita, for each continent, for each year.  
# 
# So we want a years x continents subplot
# 
# and we want some nice labels.  
# 
# Perhaps we have size vary with country population?
# 
# label x, y, row, column

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Complicated analysis
# 
# Let's figure out how life expectancy has changed over time with gdp per capita.
# 
# We want to know how the mean life expectancy, and the relationship with gdp per capita, has changed over time, for each continent.
# 
# 

# In[ ]:





# In[ ]:




