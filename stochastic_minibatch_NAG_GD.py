# -*- coding: utf-8 -*-
"""Created on Wed Jul  7 14:43:03 2021 @author: nourhan"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from numpy.linalg import norm
plt.close('all')

x=np.linspace(0,20).reshape(-1,1)
y=(-2*x+1).reshape(-1,1)

lr=0.001

#%%  
def plotting ():
    fig, ax = plt.subplots(2, 2)
    fig.suptitle(title, fontsize="x-large")
    
    ax[0,0].scatter(x,y)
    ax[0,0].plot(x,y_pred,"red")
    ax[0,0].set_xlabel("x")  
    ax[0,0].set_ylabel("y")     
   
    
    ax[0,1].scatter(list(range(len(cost_epoch))),cost_epoch)
    ax[0,1].plot(list(range(len(cost_epoch))),cost_epoch)
    ax[0,1].set_xlabel("No. of iteration")  
    ax[0,1].set_ylabel("cost") 
    
    
    ax[1,0].scatter(theta0,cost_epoch)
    ax[1,0].plot(theta0,cost_epoch)
    ax[1,0].set_xlabel("theta0")  
    ax[1,0].set_ylabel("cost") 
         
    
    ax[1,1].scatter(theta1,cost_epoch)
    ax[1,1].plot(theta1,cost_epoch)
    ax[1,1].set_xlabel("theta1")  
    ax[1,1].set_ylabel("cost") 
    

#%%SGD
def SGD(x,y,lr,no_epoch):
    z0,z1=0,1
    m=len(y)
    cost_epoch=[]
    theta0=[]
    theta1=[]

    for j in range(no_epoch):
        for i in range(m):           
            y_pred=z0+z1*x[i]
            cost=(1/2)*((y_pred-y[i])**2)
            grad_z0=y_pred-y[i]
            grad_z1=(y_pred-y[i])*x[i]
            
            z0=z0-lr*grad_z0
            z1=z1-lr*grad_z1
            
            cost_epoch.append(cost)
            theta0.append(z0)
            theta1.append(z1)
            

    return z0,z1,cost_epoch,theta0,theta1

z0,z1,cost_epoch,theta0,theta1=SGD(x, y,lr,100)

y_pred=z0+z1*x

print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_pred))

title="stochastic Gradient descent"
plotting()


#%%

m=len(y)
cost_epoch=[]
theta0=[]
theta1=[]

no_epoch=100
no_batches=2

def miniBatch_GD(x,y,lr,no_epoch,no_batches):   
    z0,z1=0,0
    s=int(m/no_batches)
    for j in range(no_epoch):
        for i in range(no_batches):
            to=(s*i)+(s-1)
            y_pred=z0+z1*x[s*i:to]
            cost=(1/2*s)*sum((y_pred-y[s*i:to])**2)
            grad_z0=(1/s)*sum(y_pred-y[s*i:to])
            grad_z1=(1/s)*sum((y_pred-y[s*i:to])*x[s*i:to])
                       
            z0=z0-lr*grad_z0
            z1=z1-lr*grad_z1
    
            cost_epoch.append(cost)
            theta0.append(z0)
            theta1.append(z1)
    return z0,z1

z0,z1=miniBatch_GD(x, y,lr,no_epoch,no_batches)
y_pred=z0+z1*x

print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_pred))

title="miniBatch_GD to Linear Regression" 

plotting()


#%%momentum
def momentum_GD(x,y,lr,no_itr):
    z0,z1=0,0
    m=len(y)
    cost_epoch=[]
    theta0=[]
    theta1=[]
    v_th0,v_th1=0,0
    gamma=0.4
    for i in range(no_itr):
        y_pred=z0+z1*x
        cost=(1/2*m)*sum((y_pred-y)**2)
        grad_z0=(1/m)*sum(y_pred-y)
        grad_z1=(1/m)*sum((y_pred-y)*x)
        
        v_th0=(gamma*v_th0)+(lr*grad_z0)
        v_th1=(gamma*v_th1)+(lr*grad_z1)
        
        z0=z0-v_th0
        z1=z1-v_th1
        cost_epoch.append(cost)
        theta0.append(z0)
        theta1.append(z1)
    return z0,z1,cost_epoch,theta0,theta1

z0,z1,cost_epoch,theta0,theta1=momentum_GD(x, y,lr,100)
y_pred=z0+z1*x

print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_pred))

title="momentum gradient"
plotting()


#%%NAG
def NAG(x,y,lr,no_itr):
    z0,z1=0,0
    m=len(y)
    cost_epoch=[]
    theta0=[]
    theta1=[]
    v_th0,v_th1=0,0
    gamma=0.9
    
    for i in range(no_itr):
        y_pred=z0+z1*x
        cost=(1/2*m)*sum((y_pred-y)**2)
        
        z0_temp=z0-gamma*v_th0
        z1_temp=z1-gamma*v_th1
        
        h_temp=z0_temp+(z1_temp*x)
        
        grad_h0=(1/m)*sum(h_temp-y)
        grad_h1=(1/m)*sum((h_temp-y)*x)
        
        z0=z0_temp-(lr*grad_h0)
        z1=z1_temp-(lr*grad_h1)
        
        v_th0=gamma*v_th0+(lr*grad_h0)
        v_th1=gamma*v_th1+(lr*grad_h1)
        

        cost_epoch.append(cost)
        theta0.append(z0)
        theta1.append(z1)
        
        grad=np.array([grad_h0,grad_h1])
        if i>1:
            if norm(grad,2)<0.001:
                break        
            if abs(cost_epoch[i]-cost_epoch[i-1])<0.001:
                break
        
    title=f"NAG Applied to Linear Regression"
    return z0,z1,cost_epoch,theta0,theta1,title


z0,z1,cost_epoch,theta0,theta1,title=NAG(x, y,lr,100)
y_pred=z0+z1*x

print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_pred))


plotting()











    
    
    
    
    
    
    