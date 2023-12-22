import torch
import torch.nn as nn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
from matplotlib.cm import rainbow
import seaborn as sns 
import matplotlib.animation as animation
import matplotlib.pylab as pl
import matplotlib as mpl 

def plot_cluster(data,label,dim):
    '''
    data 2d: np.array batch,dim
    data 3d: data_all a list  of np.array batch,dim
    '''
    if dim == 2:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.nipy_spectral)
        plt.savefig('results/cluster_result.png')
        plt.close()
    if dim == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for data1 in data:
            ax.scatter3D(data1[:, 0], data1[:, 1], data1[:, 2])
            #ax.plot(x_val[:, 0].cpu().numpy(), x_val[:, 1].cpu().numpy(), x_val[:, 2].cpu().numpy(), "k--", label="model", **plot_kws)
        ax.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")
        ax.legend()
        plt.savefig('results/cluster.png')
        plt.close()
    if dim == 4:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for data1 in data:
            ax.scatter3D(data1[:, 0], data1[:, 1], data1[:, 2])
            #ax.plot(x_val[:, 0].cpu().numpy(), x_val[:, 1].cpu().numpy(), x_val[:, 2].cpu().numpy(), "k--", label="model", **plot_kws)
        ax.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")
        ax.legend()
        plt.savefig('results/cluster1.png')
        plt.close()
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for data1 in data:
            ax.scatter3D(data1[:, 1], data1[:, 2], data1[:, 3])
            #ax.plot(x_val[:, 0].cpu().numpy(), x_val[:, 1].cpu().numpy(), x_val[:, 2].cpu().numpy(), "k--", label="model", **plot_kws)
        ax.set(xlabel="$x_1$", ylabel="$x_2$", zlabel="$x_2$")
        ax.legend()
        plt.savefig('results/cluster2.png')
        plt.close()

def plot_validate(epoch,x_val,x_pred,dim):
    if dim == 2:
        plt.figure()
        plt.scatter(x_pred.detach().cpu().numpy()[:,0],x_pred.detach().cpu().numpy()[:,1])
        plt.scatter(x_val.detach().cpu().numpy()[:,0],x_val.detach().cpu().numpy()[:,1])
        plt.savefig('results/val_scatter'+str(epoch)+'.png')
        plt.close()
    if dim == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(x_pred[:, 0].detach().cpu().numpy(), x_pred[:, 1].detach().cpu().numpy(), x_pred[:, 2].detach().cpu().numpy(), label="$x$",linewidth=2)
        ax.plot(x_val[:, 0].detach().cpu().numpy(), x_val[:, 1].detach().cpu().numpy(), x_val[:, 2].detach().cpu().numpy() ,label="$x_k$",linewidth=2)
        ax.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")
        ax.legend()
        plt.savefig('results/val_scatter'+str(epoch)+'.png')
        plt.close()

def plot_ode(epoch,x_val,x_pred,dim):
    if dim == 2:
        plt.figure()
        plt.scatter(x_pred.detach().cpu().numpy()[:,0],x_pred.detach().cpu().numpy()[:,1])
        plt.scatter(x_val.detach().cpu().numpy()[:,0],x_val.detach().cpu().numpy()[:,1])
        plt.savefig('results/val_scatter'+str(epoch)+'.png')
        plt.close()
    if dim == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x_pred[:, 0].detach().cpu().numpy(), x_pred[:, 1].detach().cpu().numpy(), x_pred[:, 2].detach().cpu().numpy(), label="$x$",linewidth=2)
        ax.scatter(x_val[:, 0].detach().cpu().numpy(), x_val[:, 1].detach().cpu().numpy(), x_val[:, 2].detach().cpu().numpy() ,label="$x_k$",linewidth=2)
        ax.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$")
        ax.legend()
        plt.savefig('results/ode_scatter'+str(epoch)+'.png')
        plt.close()

def postplot(loss_traj,error_traj,epoch):
    epoch_loss = len(loss_traj)
    epoch_error = len(error_traj)
    fig1 = plt.subplot(1,2,1)
    fig1.set_ylabel('loss')
    fig1.set_xlabel('epoch')
    fig2 = plt.subplot(1,2,2)
    fig2.set_ylabel('error')
    fig2.set_xlabel('epoch')
    fig1.plot(np.linspace(0,epoch,epoch_loss),loss_traj, c = 'k', alpha = 0.9)
    fig2.plot(np.linspace(0,epoch,epoch_error),error_traj, c = 'k', alpha = 0.9)            
    plt.savefig('results/postplot.png')
