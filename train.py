import torch
import torch.nn as nn
import numpy as np
from pyDOE import lhs
import time
import ot
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"
import os
torch.autograd.set_detect_anomaly(True)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import rainbow
# ignore user warnings
import warnings
import seaborn as sns 
import matplotlib.animation as animation
import matplotlib.pylab as pl
import matplotlib as mpl 
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
warnings.filterwarnings("ignore", category=UserWarning)
# Integrator keywords for solve_ivp
from model import MLP,Linear2D_system
from tool import data_gen_real,cluster,validate,compute_loss,compute_val_loss,record_model_param
from tool import least_square,compute_pinn_loss,compute_loss_combine,compute_loss_record,compute_ode_error
from plot_tool import postplot,plot_ode
from forward_tool import forward_ode,rk4,compute_ode_loss,grad_clip_2,forward_ode_val,forward_ode_val_batch
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12
import logging
logger = logging.getLogger(__name__)


class ODE_Identifier(torch.nn.Module):
    def __init__(self,cfgs,x_init,key = integrator_keywords):
        '''
        full_bs : num of generating solution of Hamiltonian system
        batchsize: sample size of data 
        T: terminal time of evolution
        theta_p: dim,1
        length: constant
        q_init: initial Q  dim,1 
        p_init: initial P dim,1 
        dim : dimension
        '''
        super(ODE_Identifier, self).__init__()
        self.device = device
        self.full_bs = cfgs.full_bs
        self.batchsize = cfgs.batchsize
        self.T = cfgs.T
        self.x_init =torch.tensor(x_init).to(self.device).float()  # [2]
        self.dim = x_init.shape[0]
        self.epoch = cfgs.epoch
        self.ode_model = MLP(nh=500,nin = 1,nout=self.dim).to(self.device)
        self.gen = torch.Generator(device= self.device)
        self.gen.manual_seed(7)
        self.theta_norm = 0
        self.lr = cfgs.lr
        self.optimizer = optim.AdamW([{'params': self.ode_model.parameters(),'lr':self.lr}])
        self.key = key
        self.n_cluster = cfgs.n_cluster
        self.task = cfgs.task
        self.cfgs = cfgs
        if self.task == 'Linear2D':
            self.model_param = Linear2D_system(device=device)
        else: 
            raise RuntimeError("task name invalid")
        self.optimizer_param = optim.AdamW([{'params': self.model_param.parameters(),'lr':self.lr*5}])
        self.val_sample = cfgs.val_sample
        self.epoch = cfgs.epoch
        self.pinn_epoch = cfgs.pinn_epoch
        
        
    def forward(self,t_start,T):
        '''
        y_0 1*dim
        out :time,dim
        '''
        t,_ = torch.sort(t_start+torch.rand(self.batchsize - 1)*T)
        
        t  = torch.cat((t_start.to(self.device) ,t.to(self.device) ),dim = 0).unsqueeze(-1)# self.full_bs
        #y0 = self.x_init.unsqueeze(0)
        #t.requires_grad= True
        y = self.ode_model(t)
        return y
    
    def train(self, epoch ):
        self.data_x,self.data_t = \
            data_gen_real(self.full_bs,
                                   self.T,device,
                                   self.x_init.detach().cpu().numpy(),
                                   self.key,model = self.task)
        self.full_bs = self.data_x.shape[0]
        self.time_val,self.x_val =\
            validate(val_sample=self.val_sample,
                              T=self.T,device=self.device,
                              x_init=self.x_init.detach().cpu().numpy(),
                              key=self.key,model=self.task)
        self.data_all,self.data_init_all,self.time_all,self.t_start,self.time_length=\
            cluster(data=self.data_x.detach().cpu().numpy(),
                    n_cluster= self.n_cluster,
                    T=self.T,
                    data_t=self.data_t)
        loss_traj = [] # record swd loss
        error_traj = [] # record mean error 
        loss_pinn_traj = [] # record pinn loss
        param1_list = []
        error_param_list = []
        lr_base = self.cfgs.lr_base
        init_pinn = False
        init_ADO = False
        start_combine = False
        #theta_traj = []
        for i in range(epoch):
            for k in range(self.n_cluster):
                seed =  torch.randperm(self.data_all[k].shape[0])[:self.batchsize-1].to(self.device)
                x = torch.cat((self.data_init_all[k],self.data_all[k][seed,:]),dim = 0).to(self.device)
                x_gen = self.forward(t_start=self.t_start[k],T = self.time_length[k])
                if self.cfgs.PINN == False and self.cfgs.ADO == False:
                    loss = compute_loss(x,x_gen,self.data_init_all[k].to(self.device),self.gen)
                if self.cfgs.PINN == True or self.cfgs.ADO==True:
                    if start_combine == True:
                        #loss = compute_loss_combine(x=x,x_gen=x_gen,init= self.data_init_all[k].to(self.device),gen = self.gen,batchsize=self.cfgs.pinn_update_batch,T=self.time_length[k],T_init = self.t_start[k],device =device,NN_model=self.ode_model,ode_model=self.model_param,dim=self.dim,lr=lr)
                        loss,loss_pinn = compute_loss_combine(x=x,x_gen=x_gen,
                                                    init= self.data_init_all[k].to(self.device),
                                                    gen = self.gen,
                                                    batchsize=self.cfgs.pinn_update_batch,
                                                    T=self.time_length[k],
                                                    T_init =self.t_start[k],
                                                    device =device,
                                                    NN_model=self.ode_model,
                                                    ode_model=self.model_param,
                                                    dim=self.dim,
                                                    lr=lr)                
                    else:
                        loss = compute_loss(x,x_gen,self.data_init_all[k].to(self.device),self.gen)
                
                loss.backward()                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.optimizer_param.zero_grad()
                #print('epcch=',i)
            if self.cfgs.PINN ==True:
            
                if loss.item()<=self.cfgs.threshold and init_pinn ==False and i>= 500 :
                    logger.info(f"Init PINN Loss{loss.item()}")
                    start_combine = True
                    self.train_lsq()
                    self.train_pinn(self.cfgs.pinn_epoch,batch=self.batchsize)
                    logger.info(f"Init PINN done")
                    init_pinn = True
                    lr = lr_base*(i)/epoch

            
                if start_combine and i%200 == 0 :
                    self.train_pinn(self.cfgs.update_epoch,self.batchsize)
                    lr = lr_base*(i)/epoch
                    print('learning rate',lr)
                    logger.info(f"pinn loss{loss_pinn.item()}")
            if self.cfgs.ADO == True:
                if loss.item()<=self.cfgs.threshold and init_ADO ==False and i>= 2001:
                    torch.save(self.ode_model.state_dict(),'results/ckpt_ADO.pth')
                    logger.info(f"Init ADO Loss{loss.item()}")
                    start_combine = True
                    self.train_lsq()
                    logger.info(f"Init ADO done")
                    init_ADO = True
                    lr = lr_base*(i)/epoch
                    _,error_all_phase1,x_pred_phase_1 = compute_val_loss(epoch=i,model=self.ode_model,time_val=self.time_val,x_val=self.x_val)
                    
                if start_combine and i%200 == 0 :
                    self.train_lsq()
                    lr = lr_base*(i)/epoch
                    print('learning rate',lr)
                    logger.info(f"pinn loss{loss_pinn.item()}")

            if i % 20 == 0:
                #print('epcch=',i)
                logger.info(f"epcch={i}")
                # compute total swd loss
                
                seed =  torch.randperm(self.data_x.shape[0])[:self.batchsize-1].to(self.device)
                x = torch.cat((self.x_init.unsqueeze(0),self.data_x[seed,:]),dim = 0).to(self.device)
                x_gen = self.forward(t_start=torch.tensor([0.0]),T = self.T)
                loss_swd,loss_pinn = compute_loss_record(x=x,x_gen=x_gen,
                                                gen = self.gen,
                                                batchsize=self.cfgs.pinn_update_batch,
                                                T=self.T,
                                                T_init =torch.tensor([0.0]),
                                                device =device,
                                                NN_model=self.ode_model,
                                                ode_model=self.model_param,
                                                dim=self.dim)
                param1,error_param = record_model_param(self.model_param)
                self.optimizer.zero_grad()
                self.optimizer_param.zero_grad()
                

                logger.info(f"SWD loss{loss_swd.item()}")
                loss_traj.append(loss_swd.item())
                # compute pinn loss
                if start_combine == True:
                    logger.info(f"PINN loss{loss_pinn.item()}")
                    loss_pinn_traj.append(loss_pinn.item())
                    param1_list.append(param1.detach().cpu().numpy())
                    error_param_list.append(error_param.detach().cpu().numpy())
                #compute mean error
                error,_,_ = compute_val_loss(epoch=i,model=self.ode_model,time_val=self.time_val,x_val=self.x_val)
                #print('error',error)
                logger.info(f"error{error}")
                error_traj.append(error.detach().cpu().numpy())
                   
        savemat('ckpt/param_pinn.mat',{'param1':param1_list,'error_param':error_param_list})
        savemat('ckpt/loss_traj.mat', {'error':error_traj,'loss_swd':loss_traj,'loss_pinn_traj':loss_pinn_traj})

        
         
        if self.cfgs.lsq == True:
            self.train_lsq()
            for para in self.model_param.parameters():	# 在训练前输出一下网络参数，与训练后进行对比
                para.requires_grad = True

        else:             
            self.train_pinn(self.pinn_epoch,self.batchsize)
        postplot(loss_traj,error_traj,epoch)
        torch.save(self.ode_model.state_dict(),'results/ckpt_ADO_done.pth')
        self.train_ode_batch(epoch = self.cfgs.ode_epoch)
        torch.save(self.model_param.state_dict(),'results/ckpt_shoot.pth')
        _,error_all_phase2,x_pred_phase_2 = compute_val_loss(epoch=i,model=self.ode_model,time_val=self.time_val,x_val=self.x_val)  
        savemat('ckpt/error.mat', {'error_all_phase1':error_all_phase1.detach().cpu().numpy(),'error_all_phase2':error_all_phase2.detach().cpu().numpy(),'x_gen_phase1':x_pred_phase_1.detach().cpu().numpy(),'x_gen_phase2':x_pred_phase_2.detach().cpu().numpy(),'x_val':self.x_val.detach().cpu().numpy()}) 


    
    
    def train_lsq(self):
        #torch.save(self.ode_model.state_dict(),'./ckpt_dict.pth')# 在训练前输出一下网络参数，与训练后进行对比
        if self.dim == 2:
            num_poly = [2,3,4]
        if self.dim == 3:
            num_poly = [3,6,12]

        for para in self.ode_model.parameters():	
            para.requires_grad = False
        
        zeta_list = least_square(batch = self.cfgs.lsq_num,
                        model=self.ode_model,
                        T=self.T,
                        device=self.device,
                        degree=self.cfgs.lsq_degree)
        
        for i in range(self.dim):
            zeta = zeta_list[i].squeeze()
            self.model_param.theta0.data[:,i] = zeta[0]
            self.model_param.theta1.data[:,i] = zeta[1:1+num_poly[0]]
            self.model_param.theta2.data[:,i] = zeta[1+num_poly[0]:1+num_poly[0]+num_poly[1] ]
            self.model_param.theta3.data[:,i] = zeta[1+num_poly[0]+num_poly[1]:1+num_poly[0]+num_poly[1]+num_poly[2]]
            #self.model_param.theta_sin.data[:,i] = zeta[1+num_poly[0]+num_poly[1]+num_poly[2]:1+num_poly[0]+num_poly[1]+num_poly[2]+self.dim]
            #self.model_param.theta_cos.data[:,i] = zeta[1+num_poly[0]+num_poly[1]+num_poly[2]+self.dim:1+num_poly[0]+num_poly[1]+num_poly[2]+2*self.dim]
            self.model_param.theta_exp.data[:,i] = zeta[1+num_poly[0]+num_poly[1]+num_poly[2]:1+num_poly[0]+num_poly[1]+num_poly[2]+self.dim]
            
            
        for para in self.ode_model.parameters():	# 在训练前输出一下网络参数，与训练后进行对比
            para.requires_grad = True
        
        self.optimizer_param.zero_grad()
        logger.info(f"0th order param{self.model_param.theta0.data}")
        logger.info(f"1st order param{self.model_param.theta1.data}")
        logger.info(f"2ed order param{self.model_param.theta2.data}")
        logger.info(f"3ed order param{self.model_param.theta3.data}")
        #logger.info(f"sin param{self.model_param.theta_sin.data}")
        #logger.info(f"cos param{self.model_param.theta_cos.data}")
        logger.info(f"exp param{self.model_param.theta_exp.data}")



    def train_pinn(self,epoch,batch):
        torch.save(self.ode_model.state_dict(),'./ckpt_dict.pth')
        for para in self.ode_model.parameters():	# 在训练前输出一下网络参数，与训练后进行对比
            para.requires_grad = False
        for i in range(epoch):
            loss = compute_pinn_loss(batchsize=batch,
                                     T=self.T,
                                     T_init=torch.tensor([0.0]),
                                     device=self.device,
                                     NN_model=self.ode_model,
                                     ode_model=self.model_param,
                                     dim=self.dim)
            loss.backward()    
            self.optimizer_param.step()
            self.optimizer_param.zero_grad()
            self.optimizer.zero_grad()
            if i % 100 == 0:
                #print('epcch=',i)
                logger.info(f"epcch={i}")
                #print('loss',loss.item())
                logger.info(f"loss{loss.item()}")
                #print(self.model_param.theta1.data)
                if self.task == 'Linear2D' or 'Linear3D':
                    logger.info(f"1st order param{self.model_param.theta1.data}")
                    #print(self.model_param.theta2.data)
                    logger.info(f"2ed order param{self.model_param.theta2.data}")
                if self.task == 'Cubic2D':
                    logger.info(f"3rd order param{self.model_param.theta.data}")
        for para in self.ode_model.parameters():	# 在训练前输出一下网络参数，与训练后进行对比
            para.requires_grad = True
                    
    def train_ode_batch(self,epoch):
        loss_traj = []
        error_traj = []
        param1_list = []
        error_param_list = []
        self.t_start = torch.cat(self.t_start,dim = 0)
        t_start,index_time = torch.sort(self.t_start)
        T1 = torch.tensor([0]).to(self.device)
        T1[0] = self.T

        with torch.no_grad():
            y = forward_ode_val_batch(self.time_val,device,self.data_init_all,self.model_param,t_start.to(self.device),T1,index_time)
        fig = plt.figure(figsize = (6,6))
        ax = fig.add_subplot(111)
        ax.grid(False)
        ax.plot(y[:,0].detach().cpu(),y[:,1].detach().cpu(),"r-",label = 'Truth',linewidth = 2)
        plt.savefig('test.png')

        for i in range(epoch):
            x_list = []
            for k in range(self.n_cluster):
                seed =  torch.randperm(self.data_all[k].shape[0])[:self.batchsize-1].to(self.device)
                x = torch.cat((self.data_init_all[k],self.data_all[k][seed,:]),dim = 0).to(self.device) # batchsize,dim
                x_list.append(x.unsqueeze(0))
            x = torch.cat(x_list,dim=0).to(self.device) #n_cluster,batchsize,dim
            x_gen,_ = forward_ode(batchsize=self.batchsize,device=self.device,T=self.time_length,y0=self.data_init_all,model=self.model_param) 
            x_gen = x_gen.permute(1,0,2)
            loss_swd,loss = compute_ode_loss(x,x_gen,self.gen,self.model_param,self.cfgs.laml1)
            loss.backward()
            if i>=self.cfgs.grad_threshold_epoch:
                grad_clip_2(self.model_param,self.cfgs.grad_threshold)    
            self.optimizer_param.step()
            self.optimizer_param.zero_grad()
            if i % 10 == 0:
                logger.info(f"epcch={i}")
                logger.info(f"loss{loss.item()}")
                with torch.no_grad():
                    x_all = forward_ode_val_batch(self.time_val,device,self.data_init_all,self.model_param,t_start.to(self.device),T1,index_time)
                    error,error_all,_ = compute_ode_error(x_pred=x_all,x_val=self.x_val,epoch=i)
                    logger.info(f"error {error}")
                    param1,error_param = record_model_param(self.model_param)

                loss_traj.append(loss_swd.item())
                error_traj.append(error.detach().cpu().numpy())
                param1_list.append(param1.detach().cpu().numpy())
                error_param_list.append(error_param.detach().cpu().numpy())
                if self.task == 'Linear2D':
                    logger.info(f"0th order param{self.model_param.theta0.data}")
                    logger.info(f"1st order param{self.model_param.theta1.data}")
                    logger.info(f"2ed order param{self.model_param.theta2.data}")
                    logger.info(f"3ed order param{self.model_param.theta3.data}")
                    #logger.info(f"sin param{self.model_param.theta_sin.data}")
                    #logger.info(f"cos param{self.model_param.theta_cos.data}")
                    logger.info(f"exp param{self.model_param.theta_exp.data}")
                
            if i %40 ==0 :
                plot_ode(epoch = i,x_val = x.reshape(-1,self.dim),x_pred=x_gen.reshape(-1,self.dim),dim= self.dim)
        savemat('ckpt/shooting.mat', {'error':error_traj,'loss':loss_traj,'x_gen':x_all.detach().cpu().numpy(),'error_all':error_all.detach().cpu().numpy()}) 
        savemat('ckpt/param_ode.mat',{'param1':param1_list,'error_param':error_param_list})
