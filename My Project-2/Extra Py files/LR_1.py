import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

experience= np.array([2.4,5.0,1.5,3.8,8.7,3.6,1.2,8.1,2.5,5,1.6,1.6,2.4,3.9,5.4])
salary = np.array([2.1,4.7,1.7,3.6,8.7,3.2,1.0,8.0,2.4,6,1.1,1.3,2.4,3.9,4.8])
n = np.size(experience)

plt.scatter(experience,salary, color = 'red')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

#initialize the parameters
a0 = 0                  #intercept
a1 = 0                  #Slope
lr = 0.001             #Learning rate
iterations = 1000       # Number of iterations
error = []              # Error array to calculate cost for each iterations.
for itr in range(iterations):
    error_cost = 0
    cost_a0 = 0
    cost_a1 = 0
    for i in range(len(experience)):
        y_pred = a0+a1*experience[i]   # predict value for given x
        error_cost = error_cost +(salary[i]-y_pred)**2
        partial_wrt_a0 = -2 *(salary[i] - y_pred)                #partial derivative w.r.t a0
        partial_wrt_a1 = (-2*experience[i])*(salary[i]-y_pred)   #partial derivative w.r.t a1
        cost_a0 = cost_a0 + partial_wrt_a0      #calculate cost for each number and add
        cost_a1 = cost_a1 + partial_wrt_a1      #calculate cost for each number and add
    a0 = a0 - lr * cost_a0    #update a0
    a1 = a1 - lr * cost_a1    #update a1
    error.append(error_cost/n)      #Append the data in array

print("a0 = ",a0)
print("a1 = ",a1)

plt.figure(figsize=(10,5))
plt.plot(np.arange(1,len(error)+1),error,color='red',linewidth = 5)
plt.title("Iteration vr error")
plt.xlabel("iterations")
plt.ylabel("Error")
plt.show()

pred = a0+a1*experience
print(pred)

plt.scatter(experience,salary,color = 'red')
plt.plot(experience,pred, color = 'green')
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()

error1 = salary - pred
se = np.sum(error1 ** 2)
mse = se/n
print("mean squared error is", mse)
