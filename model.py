import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import os
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from scipy.integrate import solve_ivp



class Linear2D_system(nn.Module):    
    def __init__(self,device,Scale = 4):
        super(Linear2D_system, self).__init__()
        self.device = device
        #self.theta =  nn.Parameter(torch.rand(2,2).to(self.device)*Scale-1/2*Scale).to(self.device)
        self.theta0= nn.Parameter((torch.rand(1,2).to(device))).to(self.device)
        self.theta1 =  nn.Parameter(torch.rand(2,2).to(device).to(self.device))
        self.theta2 =  nn.Parameter((torch.rand(3,2).to(device)*0)).to(self.device)
        self.theta3 = nn.Parameter((torch.rand(4,2).to(device)*0)).to(self.device)
        self.theta_exp =  nn.Parameter((torch.rand(2,2).to(device)*0)).to(self.device)

    def forward(self,t,y):
        out = y@self.theta1+self.theta0
        out += (y[...,0]*y[...,0]).unsqueeze(-1)*self.theta2[0:1,:]+(y[...,0]*y[...,1]).unsqueeze(-1)*self.theta2[1:2,:]+(y[...,1]*y[...,1]).unsqueeze(-1)*self.theta2[2:,:]
        out += (y[...,0]**3).unsqueeze(-1)*self.theta3[0:1,:]+(y[...,0]**2*y[...,1]).unsqueeze(-1)*self.theta3[1:2,:]+(y[...,0]*y[...,1]**2).unsqueeze(-1)*self.theta3[2:3,:]+(y[...,1]**3).unsqueeze(-1)*self.theta3[3:,:]
        #out += torch.sin(y)@self.theta_sin
        #out += torch.sin(y)@self.theta_cos
        out += torch.exp(torch.clamp(y,max = 4))@self.theta_exp
        return out

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.SiLU(),
            nn.Linear(nh, nh),
            nn.SiLU(),
            nn.Linear(nh, nh),
            nn.SiLU(),
            nn.Linear(nh, nh),
            nn.SiLU(),
            nn.Linear(nh, nout),
        )
    def forward(self, x):
        return self.net(x)
