#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install pymc3


# In[21]:


import numpy as np
import pymc3 as pm
import arviz as az


# ### (i) Noninformative prior

# In[46]:


from scipy.stats import bernoulli
n=[1,2,3,4,1000,4000]
###SAMPLER
def generator(x):
    data=bernoulli.rvs(size=x,p=0.60)
    return(data)

for i in n:
    with pm.Model() as model:
        theta=pm.Beta('theta',1,1)
        data=pm.Bernoulli('data',theta,observed=generator(i))
        trace=pm.sample(1000,random_seed=123)
        az.plot_posterior(trace)


# ### ii)Prior with peak at 0.25

# In[42]:


n=[1,2,3,4,1000,4000]
###SAMPLER
def generator(x):
    data=bernoulli.rvs(size=x,p=0.250)
    return(data)

for i in n:
    with pm.Model() as model:
        theta=pm.Beta('theta',2,4)
        data=pm.Bernoulli('data',theta,observed=generator(i))
        trace=pm.sample(1000,random_seed=123)
        az.plot_posterior(trace)


# ### iii)Prior with peak at 0.5

# In[47]:


from scipy.stats import bernoulli
n=[1,2,3,4,1000,4000]
###SAMPLER
def generator(x):
    data=bernoulli.rvs(size=x,p=0.5)
    return(data)

for i in n:
    with pm.Model() as model:
        theta=pm.Beta('theta',2,2)
        data=pm.Bernoulli('data',theta,observed=generator(i))
        trace=pm.sample(1000,random_seed=123)
        az.plot_posterior(trace)


# ### iv Prior with peak at p=0.75

# In[45]:


from scipy.stats import bernoulli
n=[1,2,3,4,1000,4000]
###SAMPLER
def generator(x):
    data=bernoulli.rvs(size=x,p=0.75)
    return(data)

for i in n:
    with pm.Model() as model:
        theta=pm.Beta('theta',4,2)
        data=pm.Bernoulli('data',theta,observed=generator(i))
        trace=pm.sample(1000,tune=2000,random_seed=123)
        az.plot_posterior(trace)


# In[61]:


from scipy import stats
from pymc3.distributions import Interpolated
def generator(x):
    data=bernoulli.rvs(size=x,p=0.35)
    return(data)
with pm.Model() as model:
    theta=pm.Beta('theta',2,2)
    data=pm.Bernoulli('data',theta,observed=generator(1))
    trace=pm.sample(1000,random_seed=123)
def update(p,samp):
    smin,smax=np.min(samp),np.max(samp)
    width=smax-smin
    x=np.linspace(smin,smax,100)
    y=stats.gaussian_kde(samp)(x)
    x=np.concatenate([[x[0]-3*width],x,[x[-1]+3*width]])
    y=np.concatenate([[0],y,[0]])
    return Interpolated(p,x,y)
for _ in range(3):
    def generator(x):
        data=bernoulli.rvs(size=x,p=0.35)
        return(data)
    with pm.Model() as model:
        theta=update("theta",trace["theta"])
        data=pm.Bernoulli('data',theta,observed=generator(1))
        trace=pm.sample(1000,random_seed=123)
az.plot_posterior(trace)
    


# In[53]:


def generator(x):
    data=bernoulli.rvs(size=x,p=0.35)
    return(data)
with pm.Model() as model:
    theta=pm.Beta('theta',2,2)
    data=pm.Bernoulli('data',theta,observed=generator(4))
    trace=pm.sample(1000,random_seed=123)
    az.plot_posterior(trace)


# # Question 2

# In[ ]:




