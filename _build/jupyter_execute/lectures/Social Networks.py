#!/usr/bin/env python
# coding: utf-8

# # Social networks
# 
# ## Want to really learn about this?
# 
# http://networksciencebook.com/
# 
# 
# ## First, lets make a small social network
# 
# ### step 1: make a set of nodes (people)

# In[1]:


import random
from math import log


# In[2]:


n = 10
p = 1.1*(log(n)/n)


# In[3]:


import random

with open("../datasets/first_names.txt", "r") as f:
    names = f.read().splitlines()

names = sorted(random.sample(names, k=n))


# In[4]:


print(names)


# ### step 2: make edges
# 
# Here we will define an Erdős-Rényi graph (ER graph) random graph, where each pair of people has some probability $p$ of having a connection.

# In[5]:


edges = dict(zip(names, [[] for i in range(n)]))

for i in range(n-1):
    for j in range(i+1,n):
        if random.random() < p:
            # edge exists, add symmetrically
            edges[names[i]].append(names[j])
            edges[names[j]] += [names[i]]


# In[6]:


for k,v in edges.items():
    print(f'{k:10}', v)


# ## Basic effects
# 
# ### Your friends have more friends than you.
# 

# In[7]:


def mean(lst):  # could also `from statistics import mean`, but this is a bit faster, and thats all we need
    return(sum(lst)/len(lst))

nff_m_nf = []
for person,friends in edges.items():
    n_friends = len(friends)
    print(f'{person:10} has {n_friends} friends ', end='')
    if(n_friends > 0):
        n_friends_per_friend = [len(edges[fr]) for fr in friends]
        print(f'with an average of {mean(n_friends_per_friend):.2f} friends each', end='')
        nff_m_nf.append(mean(n_friends_per_friend) - n_friends)
    print('')
            

print(f'\nPeople have {mean(nff_m_nf):.2f} fewer friends than the average number of friends their friends have')


# ### Connectivity
# 
# One basic property of graphs is connectivity: is everyone in the graph connected to everyone else?  Let's figure out if our social graph is connected.  (This is Dijkstra's algorithm)

# In[8]:


can_reach = dict()

for person in names:
    can_reach[person] = {person:0}
    stack = [person] 
    while stack:
        friend = stack.pop(0) # pop from front to do breadth first search 
        # using set difference to find friends of friend who 
        # we have not already seen (which would make them included in can_reach[person])
        new_friends = set(edges[friend]).difference(can_reach[person])
        # add new friends to stack.
        stack.extend(new_friends)
        # calculate steps to new friends
        steps = can_reach[person][friend] + 1
        # add to can_reach[person] via dict.update and dict comprehension
        can_reach[person].update({f:steps for f in new_friends})


# In[9]:


for k,v in can_reach.items():
    print(f'{k:10} can reach {len(v)}/{n}')


# In[10]:


for k,v in can_reach.items():
    print(f'{k:10}', v)


# ## Small world
# 
# Milgram did a neat experiment, which has captivated folks' imagination: we are more collected than we think.
# 
# Let's get our expectations squared away: here we have N people, each with an average of $N p$ friends.  $p$ is fairly small, like 0.05.  So we have 100 people, each with about 5 friends, on average.  If we pick a random pair of people, how long is the path between them?
# 
# The small world network phenomenon is that for many different types of networks, the average shortest path is proportional to log(n) where n is the number of nodes.
# 
# Let's calculate our average shortest path.

# In[11]:


n_connected = 0
sum_min_path = 0

for i in range(n-1):
    for j in range(i+1, n):
        if names[j] in can_reach[names[i]]: # path exists from i to j
            sum_min_path += can_reach[names[i]][names[j]] # min path from i to j
            n_connected += 1 

# this is average min path among *connected* people.  disconnected pairs do not contribute.
sum_min_path / n_connected


# Such small world phenomena arise in many types of networks.  In fact, to avoid such a property, we must consider networks that are very regular, like a lattice, or a ring.  Such networks do arise, when we consider connections that are more stratified, such as the network defined by the relation "went to high school with", or "shook hands with" (when considering people in the past, and in the future).  

# ## Clustering / cliquishness
# 
# The degree of clustering or cliquishness of a social network amounts to asking whether friends of friends are likely to be friends.  We will calculate this as the proportion of triads that are close.
# 

# In[12]:


person_clustering = []
for person,friends in edges.items():
    k = len(friends)
    if k < 2:
        person_clustering += [None]
    else:
        n_triads = 0
        n_possible = 0
        for i in range(k-1):
            for j in range(i+1,k):
                n_possible += 1
                if friends[j] in edges[friends[i]]:
                    n_triads += 1
        person_clustering += [n_triads / n_possible]

print(f'{p=} mean={mean([c for c in person_clustering if c is not None])}')


# So with the random graph, we have no clustering -- pairs of people who share a friend are no more likely to be friends than average.  

# ## Different types of network
# 
# ### lattice networks
# 
# The premise of lattice networks is that all people have an underlying location, and people are connected only to those people they are close to.  There are many subtle variations of this: What is the location on?  Most simply, it would be a ring, but it could be a 2d space, or something more complicated.  How does the presence of edges decrease with distance?  Perhaps each node is connected to the closest k nodes?  Perhaps the probability of connection decreases with distance?  etc.  While these are important distinctions, lets not worry about it.

# In[13]:


max_dist = 2  # degree = max_dist*2

edges = dict(zip(names, [[] for i in range(n)]))

for i in range(n-1):
    for j in range(i+1,n):
        distance = min((i-j)%n, (j-i)%n)  # distance along ring defined by index from 0 to n-1 (alphabetical order)
        if distance <= max_dist:
            edges[names[i]].append(names[j])
            edges[names[j]] += [names[i]]


# In[14]:


for k,v in edges.items():
    print(f'{k:10}', v)


# ### Watts-Strogatz perturbed ring networks
# 
# Watts & Strogatz showed that lattice networks (of the sort we defined above) can gain small world properties with a very small number of randomly rewired edges, which create long-distance ties:

# In[15]:


n_perturbations = 4

for name in random.sample(names, n_perturbations):
    old_name = random.choice(edges[name])
    new_name = random.choice(names)
    # remove previous edges
    edges[name].remove(old_name)
    edges[old_name].remove(name)
    # add new edges
    edges[name].append(new_name)
    edges[new_name].append(name)


# ### Barabasi-Albert scale-free networks
# 
# Scale free networks are defined by their power-law degree distribution: most nodes have very few edges and a few nodes have very many edges.  This describes lots of social networks, such as twitter followers: most people have very few followers, and a small number of accounts have millions of followers.
# 
# Barabasi & Albert described an algorithm for generating such scale-free networks.
# 
# This is a preferential attachment model

# In[16]:


edges = dict(zip(names, [[] for i in range(n)]))

m0 = 5 # number of initial nodes
m = 2 # new edges per node

# initialize with connected graph of m0 nodes (here... ring lattice)
for i in range(m0):
    j = (i-1) % m0
    edges[names[i]].append(names[j])
    edges[names[j]].append(names[i])

# add new nodes, each with m preferential attachment.
for i in range(m0, n):
    degree = {k:len(v) for k,v in edges.items() if len(v)>0}
    j = 0
    while j < m:
        target = random.choices([x for x in degree], weights=[v for k,v in degree.items()], k = 1)[0]
        edges[names[i]].append(target)
        edges[target].append(names[i])
        degree.pop(target)
        j += 1

        
for k,v in sorted(edges.items(), key=lambda item: -len(item[1])):
    print(f'{k:>10} {len(v)}')


# In[17]:


degree.keys()


# In[ ]:




