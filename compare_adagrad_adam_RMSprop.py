# -*- coding: utf-8 -*-
"""Created on Fri Jul  9 17:28:12 2021 @author: nourhan"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from numpy.linalg import norm
plt.close('all')

#%%data

def data():
    x=np.linspace(0,20).reshape(-1,1)
    y=(-1*x+2).reshape(-1,1)
    ones=np.ones((x.shape[0],1))
    x=np.concatenate((ones, x), axis=1)
    del ones
    return x,y


#%%plotting
def plotting ():    
    fig, ax = plt.subplots(2, 2,figsize=(15,15))
    fig.suptitle(title, fontsize="x-large")
       
    ax[0,0].scatter(x[:,1],y)
    for j in range(0,len(y_predictions),25):
        ax[0,0].plot(x[:,1],y_predictions[j])
        
    ax[0,0].plot(x[:,1],y_predictions[-1],"red",label="best fit",linewidth=3.0)    
    ax[0,0].set_xlabel("x")  
    ax[0,0].set_ylabel("y")     
    ax[0,0].legend()

    ax[0,1].scatter(list(range(len(cost_epoch))),cost_epoch)
    ax[0,1].text(100,80000,f"gradient={float(all_gradient[-1])}",fontsize="large")
    ax[0,1].set_xlabel("No. of iteration")  
    ax[0,1].set_ylabel("cost") 
       
    ax[1,0].scatter(theta0,cost_epoch)
    ax[1,0].text(0.4,80000,f"theta_0={float(all_theta[-1][0])}",fontsize="large")
    ax[1,0].set_xlabel("theta0")  
    ax[1,0].set_ylabel("cost") 
             
    ax[1,1].scatter(theta1,cost_epoch)
    ax[1,1].text(-0.8,80000,f"theta_1={float(all_theta[-1][1])}",fontsize="large")
    ax[1,1].set_xlabel("theta1")  
    ax[1,1].set_ylabel("cost")    
    
    
#%% adagrad (adaptive gradient)

x,y=data()
def adaGrad(x,y,lr=0.1,no_itr=10000):
    theta=np.zeros((x.shape[1])).reshape(-1,1)
    cost_epoch=[]
    all_theta=[]
    y_predictions=[]
    all_gradient=[]
    m=len(y)
    gradient=0
    epsilon=1e-9
    for i in range(no_itr):
        y_pred=x@theta
        cost=(1/2*m)*sum((y_pred-y)**2)
        
        grad=(1/m)*((x.T)@(y_pred-y))
        gradient=gradient+(grad**2)
   
        theta=theta-((lr*grad)/(np.sqrt(gradient)+epsilon))
        
        cost_epoch.append(np.round(cost,2))
        all_theta.append(np.round(theta,2))
        all_gradient.append(np.round(norm(grad,2),2))    
        y_predictions.append(y_pred)
                
        if i>1:
            if norm(grad,2)<=0.001:
                break        
            elif abs(cost_epoch[i]-cost_epoch[i-1])<=0.001:
                break

    title=f"AdaGrad Applied to Linear Regression\n lr={lr},max_itr={no_itr},no_itr={i+1}"
    return cost_epoch,all_theta,all_gradient,y_predictions,title

cost_epoch,all_theta,all_gradient,y_predictions,title=adaGrad(x, y)

theta0=[all_theta[i][0]for i in range(len(all_theta)) ]
theta1=[all_theta[i][1]for i in range(len(all_theta)) ]
  
print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_predictions[-1]))

plotting()

# points=[all_theta[-1*i] for i in range(1,6) ]
# np.round(np.var(points_0))

#%% RMS Prop

x,y=data()
def RMSprop(x,y,lr=0.05,no_itr=10000):
    theta=np.zeros((x.shape[1])).reshape(-1,1)
    cost_epoch=[]
    all_theta=[]
    y_predictions=[]
    B=0.9
    all_gradient=[]
    m=len(y)
    gradient=0
    epsilon=1e-9
    for i in range(no_itr):
        y_pred=x@theta
        cost=(1/2*m)*sum((y_pred-y)**2)
        
        grad=(1/m)*((x.T)@(y_pred-y))
        gradient=(B*gradient)+((1-B)*(grad**2))
   
        theta=theta-((lr*grad)/(np.sqrt(gradient)+epsilon))
        
        cost_epoch.append(np.round(cost,2))
        all_theta.append(np.round(theta,2))
        all_gradient.append(np.round(norm(grad,2),2))       
        y_predictions.append(y_pred)
            
        if i>5:
            if norm(grad,2)<0.001:
                break        
            elif abs(cost_epoch[i]-cost_epoch[i-1])<0.001:
                break
            # elif all_theta[i][0]==all_theta[i-5][0] and all_theta[i][1]==all_theta[i-5][1]:
            #     break
        
        # if i>20:
        #     points_0=[all_theta[-1*j][0]for j in range(1,21) ]
        #     points_1=[all_theta[-1*j][1]for j in range(1,21) ]
        #     if np.round(np.var(points_0),3)==0.001 and np.round(np.var(points_1),3)==0.001:
        #         break        
        
    title=f"RMS Prop Applied to Linear Regression\n lr={lr},max_itr={no_itr},no_itr={i+1}"
    return cost_epoch,all_theta,all_gradient,y_predictions,title

cost_epoch,all_theta,all_gradient,y_predictions,title=RMSprop(x, y)

theta0=[all_theta[i][0] for i in range(len(all_theta)) ]
theta1=[all_theta[i][1] for i in range(len(all_theta)) ]
  
print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_predictions[-1]))

plotting()


#%% Adam
x,y=data()
def Adam(x,y,lr=0.05,no_itr=10000):
    theta=np.zeros((x.shape[1])).reshape(-1,1)
    cost_epoch=[]
    all_theta=[]
    y_predictions=[]
    B=0.9
    gamma=0.9
    vt=0
    all_gradient=[]
    m=len(y)
    G=0
    epsilon=1e-9
    for i in range(no_itr):
        y_pred=x@theta
        cost=(1/2*m)*sum((y_pred-y)**2)        
        grad=(1/m)*((x.T)@(y_pred-y))
        
       
        G=(B*G)+((1-B)*(grad**2))
        vt=(gamma*vt)+((1-gamma)*grad)
        
        #bias 
        # if i>1:
        #     vt=vt/(1-(gamma**(i)))
        #     G=G/(1-(B**(i)))
        
        theta=theta-((lr*vt)/(np.sqrt(G)+epsilon))
        
        cost_epoch.append(np.round(cost,2))
        all_theta.append(np.round(theta,2))
        all_gradient.append(np.round(norm(grad,2),2))    
        y_predictions.append(y_pred)
        
        #stop criteria
        if i>5:
            if norm(grad,2)<0.001:
                break        
            elif abs(cost_epoch[i]-cost_epoch[i-1])<0.001:
                break

        # if i>20:
        #     points_0=[all_theta[-1*j][0]for j in range(1,21) ]
        #     points_1=[all_theta[-1*j][1]for j in range(1,21) ]
        #     if np.round(np.var(points_0),3)==0.001 and np.round(np.var(points_1),3)==0.001:
        #         break
            
    title=f"Adam Applied to Linear Regression\n lr={lr},max_itr={no_itr},no_itr={i+1}"
    return cost_epoch,all_theta,all_gradient,y_predictions,title

cost_epoch,all_theta,all_gradient,y_predictions,title=Adam(x, y)

theta0=[all_theta[i][0]for i in range(len(all_theta)) ]
theta1=[all_theta[i][1]for i in range(len(all_theta)) ]
  
print("Evaluation of prediction performance")
print("r2 score=",r2_score(y, y_predictions[-1]))

plotting()



#%%
# def compare(alpha):
#     ada_list=adaGrad(x, y,alpha)
#     RMS_list=RMSprop(x, y,alpha)
#     Adam_list=Adam(x, y,alpha)
    
#     fig1, a = plt.subplots(1, 3,figsize=(15,15))
#     fig1.suptitle(f"learning rate={alpha}", fontsize="x-large")
    
    
#     a[0].scatter(list(range(len(ada_list[0]))),ada_list[0])
#     a[0].text(0.8, 0.8,f"gradient={float(ada_list[2][-1])}",ha='right', va='top',fontsize="large",transform=a[0].transAxes)
#     a[0].text(0.8, 0.9,f"iteration={len(ada_list[0])}",ha='right', va='top',fontsize="large",transform=a[0].transAxes)
#     a[0].set_xlabel("No. of iteration")  
#     a[0].set_ylabel("cost") 
#     a[0].set_title("AdaGrad")
    
    
#     a[1].scatter(list(range(len(RMS_list[0]))),RMS_list[0])
#     a[1].text(0.8, 0.8,f"gradient={float(RMS_list[2][-1])}",ha='right', va='top',fontsize="large",transform=a[1].transAxes)
#     a[1].text(0.8, 0.9,f"iteration={len(RMS_list[0])}",ha='right', va='top',fontsize="large",transform=a[1].transAxes)
#     a[1].set_xlabel("No. of iteration")  
#     a[1].set_ylabel("cost") 
#     a[1].set_title("RMS Prop")
    
#     a[2].scatter(list(range(len(Adam_list[0]))),Adam_list[0])
#     a[2].text(0.8, 0.8,f"gradient={float(Adam_list[2][-1])}",ha='right', va='top',fontsize="large",transform=a[2].transAxes)
#     a[2].text(0.8, 0.9,f"iteration={len(Adam_list[0])}",ha='right', va='top',fontsize="large",transform=a[2].transAxes)
#     a[2].set_xlabel("No. of iteration")  
#     a[2].set_ylabel("cost") 
#     a[2].set_title("Adam")


# compare(0.01)


#%%
def compare_cost_itr(alpha):
    Adagrad=adaGrad(x, y,alpha)
    RMS_Prop=RMSprop(x, y,alpha)
    adam=Adam(x, y,alpha)
    algorthims=[Adagrad,RMS_Prop,adam]
    
    fig1, a = plt.subplots(1, 3,figsize=(15,6))
    fig1.suptitle(f"learning rate={alpha}", fontsize="x-large")
    name=["Adagrad","RMS_Prop","Adam"]
    for i in range(len(algorthims)):
        a[i].scatter(list(range(len(algorthims[i][0]))),algorthims[i][0])
        a[i].text(0.8, 0.8,f"gradient={float(algorthims[i][2][-1])}",ha='right', va='top',fontsize="large",transform=a[i].transAxes)
        a[i].text(0.8, 0.9,f"iteration={len(algorthims[i][0])}",ha='right', va='top',fontsize="large",transform=a[i].transAxes)
        a[i].set_xlabel("No. of iteration")  
        a[i].set_ylabel("cost") 
        a[i].set_title(name[i])
    

compare_cost_itr(0.01)

#%%
def compare_x_y(alpha):
    x,y=data()
    Adagrad=adaGrad(x, y,alpha)
    RMS_Prop=RMSprop(x, y,alpha)
    adam=Adam(x, y,alpha)
    algorthims=[Adagrad,RMS_Prop,adam]
    
    fig1, a = plt.subplots(1, 3,figsize=(15,6))
    fig1.suptitle(f"learning rate={alpha}", fontsize="x-large")
    name=["Adagrad","RMS_Prop","Adam"]
    for i in range(len(algorthims)):
        a[i].scatter(x[:,1],y)
        for j in range(0,len(algorthims[i][3]),25):
            a[i].plot(x[:,1],algorthims[i][3][j])
        a[i].text(0.4, 0.4,f"r2 score={np.round(r2_score(y,algorthims[i][3][-1]),2)}",ha='right', va='top',fontsize="large",transform=a[i].transAxes)    
        a[i].plot(x[:,1],algorthims[i][3][-1],"red",label="best fit",linewidth=3.0)    
        a[i].set_xlabel("x")  
        a[i].set_ylabel("y")
        a[i].set_title(name[i])     
        a[i].legend()       
        
compare_x_y(0.01)        
        
        
        
    
        
        
        
        