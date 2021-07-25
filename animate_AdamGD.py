# -*- coding: utf-8 -*-
"""Created on Thu Jul 22 18:48:08 2021 @author: nourhan"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from celluloid import Camera
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

def plotting_animate():
    fig, ax = plt.subplots(2, 2,figsize=(15,15))
    fig.suptitle(title, fontsize="x-large")
    camera=Camera(fig) 
    
    ax[0,0].set_xlabel("x")  
    ax[0,0].set_ylabel("y") 
    ax[0,1].set_xlabel("No. of iteration")  
    ax[0,1].set_ylabel("cost") 
    ax[1,0].set_xlabel("theta0")  
    ax[1,0].set_ylabel("cost") 
    ax[1,1].set_xlabel("theta1")  
    ax[1,1].set_ylabel("cost") 
    
     
    for j in range(0,len(y_predictions)):
        ax[0,0].scatter(x[:,1],y,color = 'blue')
        a=ax[0,0].plot(x[:,1],y_predictions[j], lw = 2.5)
        ax[0,0].vlines(x[:,1], ymin=y, ymax=y_predictions[j],
                 linestyle="dashed",color='r',alpha=0.3) # alpha:degree color
        ax[0,0].legend(a,[f"itr {j}"])
     
        
        c=ax[0,1].plot(list(range(len(cost_epoch))),cost_epoch,linestyle="dashed",color="b")
        ax[0,1].text(0.9, 0.6,f"cost={float(cost_epoch[-1])}, gradient={float(all_gradient[-1])}",
                     ha='right', va='top',fontsize="large",transform= ax[0,1].transAxes)
        ax[0,1].scatter(j,cost_epoch[j],color='r')
        ax[0,1].legend(c,[f"itr {j}\ncost={cost_epoch[j]}"])
        
        
        tho=ax[1,0].plot(theta0,cost_epoch,linestyle="dashed",color="b")
        ax[1,0].text(0.9, 0.6,f"theta_0={float(all_theta[-1][0])}",
             ha='right', va='top',fontsize="large",transform= ax[1,0].transAxes)
        ax[1,0].scatter(theta0[j],cost_epoch[j],color='r')
        ax[1,0].legend(tho,[f" itr {j}\n theta0={theta0[j]}\n cost={cost_epoch[j]}"])
        
        
        th1=ax[1,1].plot(theta1,cost_epoch,linestyle="dashed",color="b")
        ax[1,1].text(0.9, 0.6,f"theta_1={float(all_theta[-1][1])}",
             ha='right', va='top',fontsize="large",transform= ax[1,1].transAxes)
        ax[1,1].scatter(theta1[j],cost_epoch[j],color='r')
        ax[1,1].legend(th1,[f" itr {j}\n theta1={theta1[j]}\n cost={cost_epoch[j]}"])

        
        camera.snap()

    animation = camera.animate(interval = 100) 
    animation.save('adam_optimizer.gif')
    return animation



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

p=plotting_animate()




