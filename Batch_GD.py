# -*- coding: utf-8 -*-
"""Created on Wed Jul  7 11:08:19 2021 @author: nourhan"""


from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pandas as pd
from numpy.linalg import norm

def plotting ():
    fig, ax = plt.subplots(2, 2,figsize=(15,15))
    fig.suptitle(title, fontsize="x-large")
    
    ax[0,0].scatter(x,y)
    ax[0,0].plot(x,y_pred,"red",label='implemented')
    ax[0,0].plot(x,y_pred_model,"black",label='sklearn')
    ax[0,0].set_xlabel("x")  
    ax[0,0].set_ylabel("y")     
    ax[0,0].legend()
    
    ax[0,1].scatter(list(range(len(cost_epoch))),cost_epoch)
    ax[0,1].set_xlabel("No. of iteration")  
    ax[0,1].set_ylabel("cost") 
    
    
    ax[1,0].scatter(theta0,cost_epoch)
    ax[1,0].set_xlabel("theta0")  
    ax[1,0].set_ylabel("cost") 
         
    
    ax[1,1].scatter(theta1,cost_epoch)
    ax[1,1].set_xlabel("theta1")  
    ax[1,1].set_ylabel("cost")    
    
#%%    
'''importing data'''
data_set=pd.read_csv("RegData.csv")

x=data_set.iloc[:,0].values.reshape(-1,1) 
y=data_set.iloc[:, 1].values.reshape(-1,1)   

# using sklearn.linear_model
y_pred_model=LinearRegression().fit(x,y).predict(x) 

def GD(x,y,lr=0.001,no_itr=10000):
    z0,z1=0,0
    cost_epoch=[]
    theta0=[]
    theta1=[]
    m=len(y)
    
    for i in range(no_itr):
        y_pred=z0+z1*x
        cost=(1/2*m)*sum((y_pred-y)**2)
        
        grad_z0=(1/m)*sum(y_pred-y)
        grad_z1=(1/m)*sum((y_pred-y)*x)
        grad=np.array([grad_z0,grad_z1])
        
        z0=z0-lr*grad_z0
        z1=z1-lr*grad_z1
        
        cost_epoch.append(cost)
        theta0.append(z0)
        theta1.append(z1)
        
        if i>1:
            if norm(grad,2)<0.001:
                break        
            if abs(cost_epoch[i]-cost_epoch[i-1])<0.001:
                break
    title=f"GD to single var. Linear Regression\n lr={lr},no_itr={i},no_epoch={i*m}"
    return z0,z1,title,cost_epoch,theta0,theta1

z0,z1,title,cost_epoch,theta0,theta1=GD(x, y)
y_pred=z0+z1*x

print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_pred))


plotting()


#%%
   
'''importing data'''
data_set=pd.read_csv("MultipleLR.csv")

x=data_set.iloc[:,:3].values
y=data_set.iloc[:, -1].values.reshape(-1,1)
ones=np.ones((x.shape[0],1))
x=np.concatenate((ones, x), axis=1)

del ones ,data_set


def MGD(x,y,lr,no_itr):
    cost_epoch=[]
    theta=np.zeros((x.shape[1])).reshape(-1,1)
    m=len(y)
    all_theta=[]

    for i in range(no_itr):
        y_pred=x@theta
        cost=(1/2*m)*sum((y_pred-y)**2)
        grad=(1/m)*((x.T)@(y_pred-y))
        
        theta=theta-lr*grad
        
        cost_epoch.append(cost)
        all_theta.append(theta)
        
        if i>1:
            if norm(grad,2)<0.001:
                break        
            if abs(cost_epoch[i]-cost_epoch[i-1])<0.001:
                break
        
    title=f"GD to multiple var. Linear Regression\n lr={lr},no_itr={i},no_epoch={i*m}"
    return cost_epoch,all_theta,y_pred,title


cost_epoch,all_theta,y_pred,title=MGD(x, y,0.0001,300)

theta0=[all_theta[i][0]for i in range(len(all_theta)) ]
theta1=[all_theta[i][1]for i in range(len(all_theta)) ]
  

print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_pred))


fig, ax = plt.subplots(2, 2)
fig.suptitle(title, fontsize="x-large")


ax[0,0].scatter(x[:,1],y)

ax[0,0].set_xlabel("x1")  
ax[0,0].set_ylabel("y")     
   

ax[0,1].scatter(list(range(len(cost_epoch))),cost_epoch)
ax[0,1].set_xlabel("No. of iteration")  
ax[0,1].set_ylabel("cost") 


ax[1,0].scatter(theta0,cost_epoch)
ax[1,0].set_xlabel("theta0")  
ax[1,0].set_ylabel("cost") 
     

ax[1,1].scatter(theta1,cost_epoch)
ax[1,1].set_xlabel("theta1")  
ax[1,1].set_ylabel("cost")  



    
    
    
    
    