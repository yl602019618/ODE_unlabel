import torch
import torch.nn as nn
import numpy as np

import ot
def rk4(f, y0, t,device):
    '''
    f(t,y):y 1 dim -> 1,dim
    y0: 1,dim
    t: batcht
    output: batcht,dim
    '''
    
    dt = torch.diff(t,n=1,dim = 0)
    device = t.device
    step = dt.shape[0]
    y = y0
    y_all = torch.zeros(step+1,y0.shape[1]).to(device)
    y_all[0:1,:] = y
    for i in range(0, step):
        k1 = dt[i:i+1]*f(t[i:i+1],y) # y batch 1 ,dim ,t 1,dim -->batch 1 ,dim
        k2 =  dt[i:i+1]*f( t[i:i+1] + 0.5*dt[i:i+1],y + 0.5*k1)
        k3 =  dt[i:i+1]*f(t[i:i+1] + 0.5*dt[i:i+1],y + 0.5*k2)
        k4 = dt[i:i+1]*f(t[i:i+1],y + k3)
        y = y+(k1 + 2*k2 + 2*k3 + k4)/6
        
        y_all[i+1:i+2,:] = y
    return y_all



def rk4_batch(f, y0, t,device):
    '''
    f(t,y):y n_cluster dim -> n_cluster,dim
    y0: list of 1 dim tensor, after cat: n_cluster,dim
    t: n_cluster,batch
    output: batcht,n_cluster,dim
    '''
    
    dt = torch.diff(t,n=1,dim = 1).float() # n_cluster,batch-1
    
    device = t.device
    step = dt.shape[1]
    y0 = torch.cat(y0,dim = 0).to(device) #n_cluster,dim
    y = y0
    y_all = torch.zeros(step+1,y0.shape[0],y0.shape[1]).to(device) # batch,n_cluster,dim
    y_all[0:1,:] = y
    for i in range(0, step):
        k1 = dt[:,i:i+1]*f(t[:,i:i+1],y) # y n_cluster,dim ,t n_cluster,1 -->n_cluster,dim
        
        k2 =  dt[:,i:i+1]*f( t[:,i:i+1] + 0.5*dt[:,i:i+1],y + 0.5*k1)
        k3 =  dt[:,i:i+1]*f(t[:,i:i+1] + 0.5*dt[:,i:i+1],y + 0.5*k2)
        k4 = dt[:,i:i+1]*f(t[:,i:i+1],y + k3)
        y = y+(k1 + 2*k2 + 2*k3 + k4)/6
        
        y_all[i+1:i+2,:] = y
    return y_all
def forward_ode(batchsize,device,T,y0,model):
        '''
        T list of time length[] 1D
        y_0 1,dim
        out :batch,batchsize,dim
        '''
        t,_ = torch.sort(torch.rand(batchsize-1),dim = 0)  
        t  = torch.cat((torch.zeros(1).to(device) ,t.to(device)),dim = 0).unsqueeze(0)*(torch.tensor(T).unsqueeze(1).to(device))# n_cluster,self.batchsize
      
        y = rk4_batch(model, y0, t,device=device)
        return y,t

def forward_ode_val(t,device,y0,model):
    '''
    T list of time length[] 1D
    y_0 1,dim
    out :batch,batchsize,dim
    '''

    y = rk4(model, y0, t,device=device)
    return y

def compute_ode_loss(x,x_gen,gen,model,lamb):
    '''
    x:n_cluster,time,dim
    x_gen:n_cluster,time,dim
    '''
    loss = 0
    
    for i in range(x.shape[0]):
        loss+=ot.sliced_wasserstein_distance(x[i,:,:], x_gen[i,:,:], n_projections=50, seed=gen)
    
    l1 = 0
    for param in model.parameters():
        l1 +=  torch.sum(torch.abs(param))
    l1 +=  20*torch.sum(torch.abs(model.theta3))
    l1 +=  20*torch.sum(torch.abs(model.theta2))
    #norm = torch.norm(self.ode_model.theta ,p=1)
    #print(norm_pq)
    #print(x_gen[0],x[0])
    
    #loss = self.kl_loss(F.log_softmax(x, dim=1),F.log_softmax(y, dim=1))
    return loss,loss + lamb*l1 

def grad_clip_2(model,threshold):
    model.theta0.data = model.theta0.data*(torch.abs(model.theta0.data)>=threshold)
    model.theta1.data = model.theta1.data*(torch.abs(model.theta1.data)>=threshold)
    model.theta2.data = model.theta2.data*(torch.abs(model.theta2.data)>=threshold)
    model.theta3.data = model.theta3.data*(torch.abs(model.theta3.data)>=threshold)
    #model.theta_sin.data = model.theta_sin.data*(torch.abs(model.theta_sin.data)>=threshold)
    #model.theta_cos.data = model.theta_cos.data*(torch.abs(model.theta_cos.data)>=threshold)
    model.theta_exp.data = model.theta_exp.data*(torch.abs(model.theta_exp.data)>=threshold)
    model.theta1.grad *= (torch.abs(model.theta1.data)>=threshold)
    model.theta2.grad *= (torch.abs(model.theta2.data)>=threshold)
    model.theta0.grad *= (torch.abs(model.theta0.data)>=threshold)
    model.theta3.grad *= (torch.abs(model.theta3.data)>=threshold)
    #model.theta_sin.grad *= (torch.abs(model.theta_sin.data)>=threshold)
    #model.theta_cos.grad *= (torch.abs(model.theta_cos.data)>=threshold)
    model.theta_exp.grad *= (torch.abs(model.theta_exp.data)>=threshold)
    
def forward_ode_val_batch(t,device,y0,model,t_start,T,index_time):
    ''''
    t:t_val,2000
    T:time_length: n_cluster
    '''
    
    t_start = torch.cat((t_start,T),dim = 0)
    batch = t.shape[0]
    n_cluster = len(y0)
    dim = y0[0].shape[1]
    y = torch.zeros(batch,dim).to(device)
    for i in range(n_cluster):
        index = (t<=t_start[i+1]+0.0001)*  (t>=t_start[i])
        y[index,:] = rk4(model, y0[index_time[i]].to(device), t[index],device=device)
    return y