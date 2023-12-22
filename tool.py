import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from pyDOE import lhs
import time
import math
import ot
from torchdiffeq import odeint
from utils import linear_damped_SHO
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import os
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
from plot_tool import plot_cluster,plot_validate
from forward_tool import forward_ode_val,forward_ode_val_batch



def data_gen_real(full_bs,T,device,x_init,key,model):
    '''
    x_init:numpy dim
    return:
    data_x:tensor full_bs,dim
    data_t:tensor full_bs
    '''
    if model == 'Linear2D':
        model = linear_damped_SHO
    time_sample,_ = torch.sort(torch.rand(full_bs-1)*T)

    diff_t = time_sample[1:]-time_sample[:-1]
    ind = diff_t <= 0
    time_sample= (time_sample[:-1])[~ind]
    t_train  = torch.cat((torch.tensor([0]).to(device),time_sample.to(device)),dim = 0).detach().cpu().numpy() # self.full_bs
    t_train_span = (t_train[0], t_train[-1])        
    #dh = torch.from_numpy(np.diff(time_sample.cpu().numpy())).to(self.device)
    x_train = solve_ivp(model, t_train_span, 
                x_init, t_eval=t_train, **key).y.T
    
    data_x = torch.tensor(x_train).float().to(device)
    full_bs = data_x.shape[0]
    data_t = torch.tensor(t_train) 
    return data_x,data_t

def validate(val_sample,T,device,x_init,key,model):
    '''
    x_init:numpy: dim
    '''
    if model == 'Linear2D':
        model = linear_damped_SHO  
    time_val = torch.linspace(0,T,val_sample).to(device)
    t_train_span = (time_val[0], time_val[-1])        
    x_train = solve_ivp(model, t_train_span, 
                x_init, t_eval=time_val.detach().cpu().numpy(), **key).y.T
    x_val = torch.tensor(x_train).to(device) 
    return time_val,x_val



def cluster(data,n_cluster,T,data_t):
    '''
    data: numpy array: batch, dim
    data_t: numpy array: batch 
    return
    data_all list , num,dim
    data_init_all list 1,dim
    time_all list num
    t_start list 1

    '''
    knn_graph = kneighbors_graph(data, 30, include_self=False)
    model = AgglomerativeClustering(
                linkage="ward", connectivity=knn_graph, n_clusters=n_cluster
            )
    #print('cluster start')
    logger.info(f'cluster start')
    model.fit(data)
    #print('cluster end')
    logger.info(f'cluster end')
    label = model.labels_
    time_length = np.zeros(n_cluster)
    for i in range(n_cluster):
        time_length[i] = np.sum(label==i)/data.shape[0]*T
    data_all = []
    data_init_all = []
    time_all = []
    t_start = []
    for i in range(n_cluster):
        x = torch.tensor(data[model.labels_==i,:])
        data_all.append(x)
        data_init_all.append(x[0:1,:])
        time_all.append(data_t[model.labels_==i])
        t_start.append(data_t[model.labels_==i][0:1])
    
    if data.shape[1] == 2 : 
        plot_cluster(data = data,label = model.labels_,dim = data.shape[1])
    return data_all,data_init_all,time_all,t_start,time_length

def assemble_A(data,degree):
    '''
    data batch,dim

    A: 3 order polynomial , sin,cos,exp
    '''
    dim = data.shape[1]
    if degree == 0:
        basis_num = 1
    if degree == 1:
        basis_num = 1+dim
    if degree == 2:
        basis_num = int(np.math.factorial(dim+degree)
                            /(np.math.factorial(dim)*np.math.factorial(degree)))
    if degree == 3:
        basis_num = int(np.math.factorial(dim+degree)
                            /(np.math.factorial(dim)*np.math.factorial(degree)))
    
    basis_num = basis_num +dim
    batch = data.shape[0]
    ep = 2*basis_num
    A = np.zeros((batch,basis_num))
    ind = 0
    if degree >=0:
        A[:,ind] = 1
        ind +=1
    if degree >=1:
        for i in range(dim):
            A[:,ind] = data[:,i]
            ind+=1
    if degree >=2:
        for i in range(dim):
            for j in range(i,dim):
                A[:,ind] = data[:,i]*data[:,j]
                ind += 1
    if degree >=3:
        for i in range(dim):
            for j in range(i,dim):
                for k in range(j,dim):
                    A[:,ind] = data[:,i]*data[:,j]*data[:,k]
                    ind += 1
    for i in range(dim):
        A[:,ind] = ep*np.exp(np.clip(data[:,i],-7,7))
        ind+=1
        
    return A



def STRidge( X0, y, lam, maxit, tol, normalize=0, print_results = False):
    """
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """
    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0

    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y))[0]
    else: w = np.linalg.lstsq(X,y)[0]
    
    num_relevant = d
    biginds = np.where(abs(w) > tol)[0]

    # Threshold and continue
    for j in range(maxit):
        # Figure out which items to cut out
        smallinds = np.where(abs(w) < tol)[0]
        print("STRidge_j: ", j)
        print("smallinds", smallinds)
        new_biginds = [i for i in range(d) if i not in smallinds]

        # If nothing changes then stop
        if num_relevant == len(new_biginds):
            print("here1")
            break
        else: num_relevant = len(new_biginds)

        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0:
                print("here2")
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else:
                print("here3")
                break
        biginds = new_biginds

        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]

    if normalize != 0: return np.multiply(Mreg,w)
    else: return w



def least_square(batch,model,T,device,degree):
    torch.save(model.state_dict(),'results/ckpt_dict.pth')
    for para in model.parameters():	# 在训练前输出一下网络参数，与训练后进行对比
        para.requires_grad = False
    t,_ = torch.sort(torch.rand(batch - 1)*T/2)        
    t  = torch.cat((torch.tensor([0]).to(device) ,t.to(device) ),dim = 0).unsqueeze(-1)# self.full_bs
    t.requires_grad= True
    y = model(t)
    dim = y.shape[-1]
    # print(y.shape)
    #y_t= torch.autograd.functional.jacobian(self.ode_model, t, create_graph=False)
    if dim ==3:
        y_t1 = torch.autograd.grad(y[:,0:1], t, grad_outputs=torch.ones_like(y[:,0:1]),retain_graph=True,create_graph=True)[0]
        y_t2 = torch.autograd.grad(y[:,1:2], t, grad_outputs=torch.ones_like(y[:,1:2]),retain_graph=True,create_graph=True)[0]
        y_t3 = torch.autograd.grad(y[:,2:], t, grad_outputs=torch.ones_like(y[:,2:]),retain_graph=True,create_graph=True)[0]
    elif dim ==2:
        y_t1 = torch.autograd.grad(y[:,0:1], t, grad_outputs=torch.ones_like(y[:,0:1]),retain_graph=True,create_graph=True)[0]
        y_t2 = torch.autograd.grad(y[:,1:2], t, grad_outputs=torch.ones_like(y[:,1:2]),retain_graph=True,create_graph=True)[0]
    elif dim ==4:
        y_t1 = torch.autograd.grad(y[:,0:1], t, grad_outputs=torch.ones_like(y[:,0:1]),retain_graph=True,create_graph=True)[0]
        y_t2 = torch.autograd.grad(y[:,1:2], t, grad_outputs=torch.ones_like(y[:,1:2]),retain_graph=True,create_graph=True)[0]
        y_t3 = torch.autograd.grad(y[:,2:3], t, grad_outputs=torch.ones_like(y[:,2:3]),retain_graph=True,create_graph=True)[0]
        y_t4 = torch.autograd.grad(y[:,3:], t, grad_outputs=torch.ones_like(y[:,3:]),retain_graph=True,create_graph=True)[0]


    A = assemble_A(y.detach().cpu().numpy(), degree = degree) # batch, base number      
        
    if dim == 3 :   
        zeta1 = STRidge(X0 =A,y = y_t1.detach().cpu().numpy(),lam = 0.,maxit = 100,tol = 0.5) #0.35
        zeta2 = STRidge(X0 =A,y = y_t2.detach().cpu().numpy(),lam = 0.,maxit = 100,tol = 0.5) #0.4
        zeta3 = STRidge(X0 =A,y = y_t3.detach().cpu().numpy(),lam = 0,maxit = 100,tol = 0.2) #0.05
        
        logger.info(f'zeta1 is{zeta1}')
        logger.info(f'zeta2 is{zeta2}')
        logger.info(f'zeta3 is{zeta3}')
        return torch.tensor(zeta1),torch.tensor(zeta2),torch.tensor(zeta3)
        
    elif dim == 2 :   
        zeta1 = STRidge(X0 =A,y = y_t1.detach().cpu().numpy(),lam = 0.0,maxit = 100,tol = 0.2)
        zeta2 = STRidge(X0 =A,y = y_t2.detach().cpu().numpy(),lam = 0.0,maxit = 100,tol = 0.2)
    
        logger.info(f'zeta1 is{zeta1}')
        logger.info(f'zeta2 is{zeta2}')
        #print('zeta3 is ',zeta3)
        return torch.tensor(zeta1),torch.tensor(zeta2)
    elif dim == 4 :   
        zeta1 = STRidge(X0 =A,y = y_t1.detach().cpu().numpy(),lam = 0.,maxit = 100,tol = 0.5) #0.35
        zeta2 = STRidge(X0 =A,y = y_t2.detach().cpu().numpy(),lam = 0.,maxit = 100,tol = 0.5) #0.4
        zeta3 = STRidge(X0 =A,y = y_t3.detach().cpu().numpy(),lam = 0,maxit = 100,tol = 0.2) #0.05
        zeta4 = STRidge(X0 =A,y = y_t4.detach().cpu().numpy(),lam = 0,maxit = 100,tol = 0.2) #0.05
        
        logger.info(f'zeta1 is{zeta1}')
        logger.info(f'zeta2 is{zeta2}')
        logger.info(f'zeta3 is{zeta3}')
        logger.info(f'zeta3 is{zeta4}')
        return torch.tensor(zeta1),torch.tensor(zeta2),torch.tensor(zeta3),torch.tensor(zeta4)









def compute_loss(x,x_gen,init,gen):
    '''
    x:time,dim
    x_gen:time,dim
    '''
    loss = ot.sliced_wasserstein_distance(x, x_gen, n_projections=100, seed=gen)
    #norm = torch.norm(self.ode_model.theta ,p=1)
    #print(norm_pq)
    #print(x_gen[0],x[0])
    norm = torch.norm(x_gen[0]-init)
    #norm_t =  1e-2*torch.norm(x_t)
    
    loss  = loss + 0.5*norm #+ norm_t  #+ self.theta_norm*norm
    #loss = self.kl_loss(F.log_softmax(x, dim=1),F.log_softmax(y, dim=1))
    return loss 

def compute_loss_combine(x,x_gen,init,gen,batchsize,T,T_init,device,NN_model,ode_model,dim,lr):
    '''
    x:time,dim
    x_gen:time,dim
    '''
    loss_swd = ot.sliced_wasserstein_distance(x, x_gen, n_projections=100, seed=gen)
    norm = torch.norm(x_gen[0]-init)
    loss_pinn = compute_pinn_loss(batchsize,T,T_init,device,NN_model,ode_model,dim) +0.02*(torch.norm(ode_model.theta1,p=1)+torch.norm(ode_model.theta2,p=1))
    loss  = loss_swd + 0.5*norm + loss_pinn*lr 
    
    return loss,loss_pinn

def compute_loss_record(x,x_gen,gen,batchsize,T,T_init,device,NN_model,ode_model,dim):
    '''
    x:time,dim
    x_gen:time,dim
    '''
    loss_swd = ot.sliced_wasserstein_distance(x, x_gen, n_projections=100, seed=gen)
    
    loss_pinn = compute_pinn_loss(batchsize,T,T_init,device,NN_model,ode_model,dim) 
    
    return loss_swd,loss_pinn





def compute_pinn_loss(batchsize,T,T_init,device,NN_model,ode_model,dim):
    t,_ = torch.sort(T_init+torch.rand(batchsize - 1)*T)        
    t  = torch.cat((T_init.to(device),t.to(device) ),dim = 0).unsqueeze(-1)# self.full_bs
    t.requires_grad= True
    y = NN_model(t)
    # print(y.shape)
    #y_t= torch.autograd.functional.jacobian(self.ode_model, t, create_graph=False)
    if dim == 2:
        y_t1 = torch.autograd.grad(y[:,0:1], t, grad_outputs=torch.ones_like(y[:,0:1]),retain_graph=True,create_graph=True)[0]
        y_t2 = torch.autograd.grad(y[:,1:], t, grad_outputs=torch.ones_like(y[:,1:]),retain_graph=True,create_graph=True)[0]
        y_t = torch.cat((y_t1,y_t2),dim = 1)
    elif dim ==3:
        y_t1 = torch.autograd.grad(y[:,0:1], t, grad_outputs=torch.ones_like(y[:,0:1]),retain_graph=True,create_graph=True)[0]
        y_t2 = torch.autograd.grad(y[:,1:2], t, grad_outputs=torch.ones_like(y[:,1:2]),retain_graph=True,create_graph=True)[0]
        y_t3 = torch.autograd.grad(y[:,2:3], t, grad_outputs=torch.ones_like(y[:,2:3]),retain_graph=True,create_graph=True)[0]
        y_t = torch.cat((y_t1,y_t2),dim = 1)
        y_t = torch.cat((y_t,y_t3),dim = 1)
    RHS = ode_model(t,y)
    #print(RHS.shape,y_t.shape)
    #assert RHS.shape==y_t.shape
    loss_fn = nn.MSELoss()
    loss = loss_fn(y_t,RHS)
    return loss


def compute_val_loss(epoch,model,time_val,x_val):
    '''
    time:time 
    true_x : time,dim
    x_val: time,dim
    '''

    with torch.no_grad():
        #self.val_sample = 1000
        
        true_x=  model(time_val.unsqueeze(-1)) 
        x_pred = true_x
        
        plot_validate(epoch=epoch,x_val=x_val,x_pred=x_pred,dim = x_val.shape[-1])
        error_all = torch.abs(x_pred-x_val)
        val_sample = x_val.shape[0]
        #print(error_all)
        error = torch.sum(error_all)/val_sample
    #error = torch.abs(self.H_pred-self.H_val)
    return error,error_all,x_pred


def compute_time_error(model,time_val,data_x,data_t,mode,data_init_all=None,t_start=None,T1=None,index_time=None):
    '''
    time:time 
    true_x : time,dim
    x_val: time,dim
    '''
    
    with torch.no_grad():
        #self.val_sample = 1000
        if mode == 'NN':
            x_pred=  model(time_val.unsqueeze(-1)).detach().cpu().numpy() 
        else:
            device = time_val.device
            #x_pred = forward_ode_val(time_val,device,data_x[0:1,:],model).detach().cpu().numpy()
            x_pred = forward_ode_val_batch(time_val,device,data_init_all,model,t_start.to(device),T1,index_time).detach().cpu().numpy()
        batch_dense = x_pred.shape[0]
        batch_data = data_x.shape[0]
        data_x,data_t = data_x.detach().cpu().numpy(),data_t.detach().cpu().numpy()
        pred_t = np.zeros_like(data_t)
        for i in range(batch_data):
            diff =np.sum(np.abs(data_x[i:i+1,:]-x_pred)**2,axis = 1)
            index = np.argmin(diff,axis = 0)
            pred_t[i] = time_val[index]
        mean_error  = np.abs(pred_t - data_t)
    return pred_t,mean_error






def compute_ode_error(x_pred,x_val,epoch):
    '''
    time:time 
    true_x : time,dim
    x_val: time,dim
    '''
    with torch.no_grad():
        #self.val_sample = 1000
        plot_validate(epoch=epoch+3000,x_val=x_val,x_pred=x_pred,dim = x_val.shape[-1])
        error_all = torch.abs(x_pred-x_val)
        val_sample = x_val.shape[0]
        #print(error_all)
        error = torch.sum(error_all)/val_sample
    #error = torch.abs(self.H_pred-self.H_val)
    return error,error_all,x_pred

def record_model_param(model):
    param_1 = model.theta1.data
    with torch.no_grad():
        error = 0
        error +=  torch.sum(torch.abs(model.theta0.data))
        error +=  torch.sum(torch.abs(model.theta2.data))
        error +=  torch.sum(torch.abs(model.theta3.data))
        error +=  torch.sum(torch.abs(model.theta_exp.data))
        

    return param_1, error

