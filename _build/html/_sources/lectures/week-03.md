# W3 problem

This week our ultimate goal is to run some simulations of an "agent based model" of segregation.  Specifically we want to simulate [Thomas Schelling's](https://en.wikipedia.org/wiki/Thomas_Schelling)  [Dynamic Models of Segregation](http://www.stat.berkeley.edu/~aldous/157/Papers/Schelling_Seg_Models.pdf).  A very nice illustration, and extension, of this model has been made by [Nicki Case](https://en.wikipedia.org/wiki/Nicky_Case) and [Vi Hart](https://en.wikipedia.org/wiki/Vi_Hart), called the [Parable of the Polygons](https://ncase.me/polygons/) (If you click on only one link in this paragraph, click the last one!)

## Agent based model

An [agent based model](https://en.wikipedia.org/wiki/Agent-based_model) works as follows:  
1. we define a model for an **agent** that *behaves* according to some simple *rules*    
2. we place a bunch of these agents in an **environment**
3. we have them behave for a while according to their behavioral rules and the environment rules.
4. we see what kind of aggregate behavior **emerges** from the interaction of these simple agents.

These kinds of models are used in a wide array of fields to predict the aggregate behavior of complex systems of interacting agents, including biology, epidemiology, evolutionary game theory, supply chain optimization, organizational behavior, sociology, political science, etc.  

To define our particular agent based model we need to specify:  
- **environment**: the rules / structure of the environment (what is the structure of the environment, what kinds of actions are available to an agent?)  
- **agents**: what are the properties of an agent, and what rules does it follow to choose among the actions available?  
- **measures**: what do we measure about the system as a whole to characterize the emergent behavior?  

### Environment

Following Schelling, we will assume a simple discrete-space, discrete-time 2d grid world:  
- the environment is an K x K grid (let's take K=10).  
- any given cell of the grid can be either empty, or occupied by an agent.  
- at any given time step an agent can choose to stay in their current location, or move randomly to another available (empty) location in the grid. 

### Agents

The agents have a categorical **type**, and they decide whether or not to move in any given time step according to one of the following rules.  
- *Aversion to being a minority*: move if less than p% of your neighbors are your type.
- *Group seeking*: move if you have fewer than k neighbors of your own type.  
- *Aversion to being a minority plus aversion to homogeneity*:  move if less than p% of your neighbors are your type *or* if more than q% of your neighbors are your type.

### Measures

We want to measure how segregated the spatial distribution of agents is.  The simplest measure is:  
- *segregation*: for each agent calculate the proportion of neighbors that are of the same type as the agent, and then average that same-type-proportion over all agents.

## Pseudo-code

This is the first structurally interesting problem we need to write.  

We want to do the following things:  
0. place agents in unique positions of the environment
1. for each agent, decide if it wants to move  
2. move each agent that wants to move to a random empty location  
3. if any agents moved, go back to step 1

A few things to notice here:  
- there are many repeated operations here, which are called at different times.  these should be abstracted into **functions** or **methods**.
- There are repeated instances of the same types of things (agents), these would be fruitfully abstracted into multiple **objects** of one **class**.  

This week we will learn how to create functions, and classes.

