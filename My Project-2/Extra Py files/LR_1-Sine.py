#!/usr/bin/env python
# coding: utf-8

# In[149]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[150]:


time= np.arange(0, 10, 0.1)
y = np.sin(time)
n = np.size(time)


# In[151]:


plt.scatter(time,y, color = 'red')
plt.xlabel("Time")
plt.ylabel("Signal")
plt.show()


# In[152]:


#initialize the parameters
a0 = 0                  #intercept
a1 = 0                  #Slope
a2 = 0
lr = 0.0001 #Learning rate
k = 0
f = k*np.pi
iterations = 4000       # Number of iterations
error = []              # Error array to calculate cost for each iterations.
for itr in range(iterations):
    error_cost = 0
    cost_a0 = 0
    cost_a1 = 0
    cost_a2 = 0
    for i in range(len(time)):
        y_pred = a0+a1*time[i]+a2*np.sin(f*time[i])   # predict value for given x
        error_cost = error_cost +(y[i]-y_pred)**2
        partial_wrt_a0 = -2 *(y[i] - y_pred)                #partial derivative w.r.t a0
        partial_wrt_a1 = (-2*time[i])*(y[i]-y_pred)   #partial derivative w.r.t a1
        partial_wrt_a2 = (-2*np.sin(time[i]))*(y[i]-y_pred)
        cost_a0 = cost_a0 + partial_wrt_a0      #calculate cost for each number and add
        cost_a1 = cost_a1 + partial_wrt_a1      #calculate cost for each number and add
        cost_a2 = cost_a2 + partial_wrt_a2
    a0 = a0 - lr * cost_a0/n    #update a0
    a1 = a1 - lr * cost_a1/n    #update a1
    a2 = a2 - lr * cost_a2/n
    error.append(error_cost/n)      #Append the data in array


# In[153]:


print("a0 = ",a0)
print("a1 = ",a1)
print("a2 = ",a2)


# In[154]:


plt.figure(figsize=(10,5))
plt.plot(np.arange(1,len(error)+1),error,color='red',linewidth = 5)
plt.title("Iteration vr error")
plt.xlabel("iterations")
plt.ylabel("Error")
plt.show()


# In[155]:


pred = a0+a1*time+a2*np.sin(f*time)
print(pred)


# In[156]:


plt.scatter(time,y,color = 'red')
plt.plot(time,pred, color = 'green')
plt.xlabel("time")
plt.ylabel("Signal")
plt.show()


# In[157]:


error1 = y - pred
se = np.sum(error1 ** 2)
mse = se/n
print("mean squared error is", mse)

