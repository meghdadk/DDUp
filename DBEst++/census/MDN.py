import Dataset
from sqlParser import Parser
import itertools as it
import math
import random
import sys
from concurrent import futures
from copy import deepcopy
from os import remove
from os.path import abspath
import category_encoders as ce
import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandasql as ps
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.multiprocessing import Pool
import torch.nn.functional as F


import scipy
from scipy import integrate,stats
from collections import Counter
from sklearn.preprocessing import OrdinalEncoder
from numpy.random import choice
import time
from itertools import cycle
import itertools
import matplotlib
import csv
import os
matplotlib.use('Agg')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dill._dill._reverse_typemap['ClassType'] = type


class MDN(nn.Module):
    """A mixture density network layer
    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.
    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions
    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians, device):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians)
        )

        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

        self.pi = self.pi.to(device)
        self.mu = self.mu.to(device)
        self.sigma = self.sigma.to(device)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


class DenMDN:
    """This class implements the regression using mixture density network for group by queries."""

    def __init__(self, dataset, b_normalize_data=True):

        self.model = None
        self.b_normalize_data = b_normalize_data
        self.batch_size = None
        self.dataset = dataset
        
    def fit(self,x_points, y_points, encoded=True,lr = 0.001,n_workers=0): 

        n_epoch = 30#self.config.config["n_epoch"]
        n_gaussians = 80#self.config.config["n_gaussians_reg"]
        n_hidden_layer = 2#self.config.config["n_hidden_layer"]
        n_mdn_layer_node = 64#self.config.config["n_mdn_layer_node_reg"]
        self.device = device
        self.batch_size = 64
        self.b_normalize_data = True
        self.b_store_training_data = True

        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))
         

        # move variables to cuda
        #tensor_xs = tensor_xs.to(device)
        #tensor_ys = tensor_ys.to(device)
        print (x_points.shape, y_points.shape)
        print (tensor_xs.shape,tensor_ys.shape)

        my_dataset = torch.utils.data.TensorDataset(
            tensor_xs, tensor_ys
        )  
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1

    
        # initialize the model
        if n_hidden_layer == 1:
            self.model = nn.Sequential(
                nn.Linear(input_dim, n_mdn_layer_node),
                nn.Tanh(),
                nn.Dropout(0.2),
                MDN(n_mdn_layer_node, 1, n_gaussians, device),
            )
        elif n_hidden_layer == 2:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.2),
                MDN(n_mdn_layer_node, 1, n_gaussians, device),
            )
        elif n_hidden_layer == 5:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.4),
                MDN(64, 1, n_gaussians, device),
            )
        else:
            raise ValueError(
                "The hidden layer should be 1 or 2, but you provided "
                + str(n_hidden_layer)
            )
        """
        means = list(np.linspace(0,100,10))

        for idx, m in enumerate(means):
            self.model[3].mu.weight[idx].data = torch.Tensor([m]*30)
            self.model[3].sigma.weight[idx].data = torch.Tensor([1]*30)
            self.model[3].pi[0].weight[idx].data = torch.Tensor([0.1]*30)
        """


        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        decay_rate = 0.96
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=decay_rate
        )
        loss = 0
        self.model = self.model.to(device)
        for epoch in range(n_epoch):
            if epoch % 1 == 0:
                print("< Epoch {}, loss {}, lr {:0.5f}".format(epoch, loss, optimizer.param_groups[0]['lr'])) 
            # train the model
            for minibatch, labels in my_dataloader: 
                minibatch = minibatch.to(device)
                labels = labels.to(device)
                self.model.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = mdn_loss(pi, sigma, mu, labels, device)
                loss.backward()
                optimizer.step()
            my_lr_scheduler.step()
        #torch.save(self.model.state_dict(), 'model.pt')
        self.model.eval()
        print("Finish regression training.")
        return self

    def update(self,x_points_update, y_points_update, x_points_transfer, y_points_transfer, encoded=True,lr = 0.0001, n_epoch=10, alpha=0.16, n_workers=0, num_gauss=40, hid_neurons=2): 

        n_gaussians = 30#self.config.config["n_gaussians_reg"]
        n_hidden_layer = 2#self.config.config["n_hidden_layer"]
        n_mdn_layer_node = 64#self.config.config["n_mdn_layer_node_reg"]
        self.device = device = 'cpu'#runtime_config["device"]
        self.batch_size = 32
        self.b_normalize_data = True
        self.b_store_training_data = True


        tensor_xs_transfer = torch.from_numpy(x_points_transfer.astype(np.float32)) 
        y_points_transfer = np.asarray(y_points_transfer).reshape(-1,1)
        tensor_ys_transfer = torch.from_numpy(y_points_transfer.astype(np.float32)) 
        
        tensor_xs_update = torch.from_numpy(x_points_update.astype(np.float32)) 
        y_points_update = np.asarray(y_points_update).reshape(-1,1)
        tensor_ys_update = torch.from_numpy(y_points_update.astype(np.float32))         


        print (tensor_xs_transfer.shape,tensor_ys_transfer.shape)
        print (tensor_xs_update.shape,tensor_ys_update.shape)


        my_dataset_transfer = torch.utils.data.TensorDataset(
            tensor_xs_transfer, tensor_ys_transfer
        )  
        my_dataloader_transfer = torch.utils.data.DataLoader(
            my_dataset_transfer,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
        )


        my_dataset_update = torch.utils.data.TensorDataset(
            tensor_xs_update, tensor_ys_update
        )  
        my_dataloader_update = torch.utils.data.DataLoader(
            my_dataset_update,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1


        pre_model = self.model

        """
        # initialize the model
        if n_hidden_layer == 1:
            self.model = nn.Sequential(
                nn.Linear(input_dim, n_mdn_layer_node),
                nn.Tanh(),
                nn.Dropout(0.2),
                MDN(n_mdn_layer_node, 1, n_gaussians, device),
            )       
        elif n_hidden_layer == 2:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64+hid_neurons),
                nn.Tanh(),
                nn.Dropout(0.2),
                MDN(n_mdn_layer_node+hid_neurons, 1, n_gaussians, device),
            )
        elif n_hidden_layer == 5:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.4),
                MDN(64, 1, n_gaussians, device),
            )
        else:
            raise ValueError(
                "The hidden layer should be 1 or 2, but you provided "
                + str(n_hidden_layer)
            )

        

        self.model[5].pi[0].weight = pre_model[5].pi[0].weight
        self.model[5].sigma.weight = pre_model[5].sigma.weight
        self.model[5].mu.weight = pre_model[5].mu.weight
        self.model[5].pi[0].weight.requires_grad = False
        self.model[5].sigma.weight.requires_grad = False
        self.model[5].mu.weight.requires_grad = False
        self.model[5].pi[0].bias.requires_grad = False
        self.model[5].sigma.bias.requires_grad = False
        self.model[5].mu.bias.requires_grad = False
        self.model = self.model.to(device)
        pre_model = pre_model.to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        decay_rate = 0.96
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=decay_rate
        )

        loss = 0
        self.model = self.model.to(device)
        pre_model = pre_model.to(device)
        self.model.train()
        pre_model.eval()
        for epoch in range(n_epoch):
            if epoch % 1 == 0:
                print("< Epoch {}, loss {}, lr {:0.5f}".format(epoch, loss, optimizer.param_groups[0]['lr']))
            for (minibatch_update, labels_update), (minibatch_transfer, labels_transfer) in zip(my_dataloader_update, cycle(my_dataloader_transfer)):

                minibatch_transfer.to(device)
                labels_transfer.to(device)

                minibatch_update.to(device)
                labels_update.to(device)

                self.model.zero_grad()
                pre_model.zero_grad()

                pi1, sigma1, mu1 = pre_model(minibatch_transfer)
                pi2, sigma2, mu2 = self.model(minibatch_transfer)
                pi3, sigma3, mu3 = self.model(minibatch_update)

                """
                targets = torch.flatten(mu1).reshape(-1,1)
                pi1 = torch.repeat_interleave(pi1,n_gaussians,dim=0)
                sigma1 = torch.repeat_interleave(sigma1,n_gaussians,dim=0)
                mu1 = torch.repeat_interleave(mu1,n_gaussians,dim=0)
                pi2 = torch.repeat_interleave(pi2,n_gaussians,dim=0)
                sigma2 = torch.repeat_interleave(sigma2,n_gaussians,dim=0)
                mu2 = torch.repeat_interleave(mu2,n_gaussians,dim=0)
                """


                loss1 = MSE_CE_loss(pi1,sigma1,mu1,pi2,sigma2,mu2,labels_transfer,device, T=1)
                loss2 = mdn_loss(pi2,sigma2,mu2,labels_transfer,device)
                loss3 = mdn_loss(pi3,sigma3,mu3,labels_update,device)
                loss =  alpha*(loss1) + (1-alpha)*((5/6)*loss2+(1/6)*loss3)

                loss.backward()
                optimizer.step()
            my_lr_scheduler.step()
        #torch.save(self.model.state_dict(), 'model.pt')
        self.model.eval()
        print("Finish regression training.")
        return self

    def distill(self,x_points_transfer, y_points_transfer, encoded=True,lr = 0.001 ,n_workers=0,n_epoch=30): 

        n_gaussians = 40#self.config.config["n_gaussians_reg"]
        n_hidden_layer = 2#self.config.config["n_hidden_layer"]
        n_mdn_layer_node = 64#self.config.config["n_mdn_layer_node_reg"]
        b_grid_search = False#self.config.config["b_grid_search"]
        self.device = device = 'cpu'#runtime_config["device"]
        self.batch_size = 64
        self.b_normalize_data = True
        self.b_store_training_data = True


        tensor_xs_transfer = torch.from_numpy(x_points_transfer.astype(np.float32)) 
        y_points_transfer = np.asarray(y_points_transfer).reshape(-1,1)
        tensor_ys_transfer = torch.from_numpy(y_points_transfer.astype(np.float32))         

        # move variables to cuda
        tensor_xs_transfer = tensor_xs_transfer.to(device)
        tensor_ys_transfer = tensor_ys_transfer.to(device)

        print (tensor_xs_transfer.shape,tensor_ys_transfer.shape)


        my_dataset_transfer = torch.utils.data.TensorDataset(
            tensor_xs_transfer, tensor_ys_transfer
        )  
        my_dataloader_transfer = torch.utils.data.DataLoader(
            my_dataset_transfer,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1


        pre_model = self.model
        """
        # initialize the model
        if n_hidden_layer == 1:
            self.model = nn.Sequential(
                nn.Linear(input_dim, n_mdn_layer_node),
                nn.Tanh(),
                nn.Dropout(0.2),
                MDN(n_mdn_layer_node, 1, n_gaussians, device),
            )       
        elif n_hidden_layer == 2:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.2),
                MDN(n_mdn_layer_node, 1, n_gaussians, device),
            )
        elif n_hidden_layer == 5:
            self.model = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.Tanh(),
                nn.Dropout(0.4),
                MDN(64, 1, n_gaussians, device),
            )
        else:
            raise ValueError(
                "The hidden layer should be 1 or 2, but you provided "
                + str(n_hidden_layer)
            )



        self.model[5].pi[0].weight = pre_model[5].pi[0].weight
        self.model[5].sigma.weight = pre_model[5].sigma.weight
        self.model[5].mu.weight = pre_model[5].mu.weight
        self.model[5].pi[0].weight.requires_grad = False
        self.model[5].sigma.weight.requires_grad = False
        self.model[5].mu.weight.requires_grad = False
        self.model[5].pi[0].bias.requires_grad = False
        self.model[5].sigma.bias.requires_grad = False
        self.model[5].mu.bias.requires_grad = False
        self.model = self.model.to(device)
        pre_model = pre_model.to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        decay_rate = 0.96
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=decay_rate
        )

        loss = 0

        self.model.train()
        pre_model.eval()
        for epoch in range(n_epoch):
            if epoch % 1 == 0:
                print ("epoch {}\tloss = {}".format(epoch, loss))

            for minibatch_transfer, labels_transfer in my_dataloader_transfer:

                minibatch_transfer.to(device)
                labels_transfer.to(device)

                self.model.zero_grad()
                pre_model.zero_grad()

                pi1, sigma1, mu1 = pre_model(minibatch_transfer)
                pi2, sigma2, mu2 = self.model(minibatch_transfer)

                """
                targets = torch.flatten(mu1).reshape(-1,1)
                pi1 = torch.repeat_interleave(pi1,n_gaussians,dim=0)
                sigma1 = torch.repeat_interleave(sigma1,n_gaussians,dim=0)
                mu1 = torch.repeat_interleave(mu1,n_gaussians,dim=0)
                pi2 = torch.repeat_interleave(pi2,n_gaussians,dim=0)
                sigma2 = torch.repeat_interleave(sigma2,n_gaussians,dim=0)
                mu2 = torch.repeat_interleave(mu2,n_gaussians,dim=0)
                """


                loss1 = MSE_CE_loss(pi1,sigma1,mu1,pi2,sigma2,mu2,labels_transfer,'cpu', T=1)
                loss2 = mdn_loss(pi2,sigma2,mu2,labels_transfer,'cpu')
                loss =  loss1 #+ (1/9)*loss2

                loss.backward()
                optimizer.step()
            my_lr_scheduler.step()
        #torch.save(self.model.state_dict(), 'model.pt')
        self.model.eval()
        print("Finish regression training.")
        return self

    def finetune(self,x_points, y_points, encoded=True,lr = 0.00001,n_workers=0):

        n_epoch = 10

        self.batch_size = 64
        self.b_normalize_data = True
        self.b_store_training_data = True

        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
        tensor_ys = torch.from_numpy(y_points.astype(np.float32))
         

        # move variables to cuda
        tensor_xs = tensor_xs.to(device)
        tensor_ys = tensor_ys.to(device)
        print (x_points.shape, y_points.shape)
        print (tensor_xs.shape,tensor_ys.shape)

        my_dataset = torch.utils.data.TensorDataset(
            tensor_xs, tensor_ys
        )  
        my_dataloader = torch.utils.data.DataLoader(
            my_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=n_workers,
        )

        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1

    

        self.model = self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        decay_rate = 0.96
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=decay_rate
        )
        loss = 0
        for epoch in range(n_epoch):
            if epoch % 1 == 0:
                print("< Epoch {}, loss {}".format(epoch, loss)) 
            # train the model
            for minibatch, labels in my_dataloader: 
                minibatch.to(device)
                labels.to(device)
                self.model.zero_grad()
                pi, sigma, mu = self.model(minibatch)
                loss = mdn_loss(pi, sigma, mu, labels, device)
                loss.backward()
                optimizer.step()
            my_lr_scheduler.step()

        self.model.eval()
        print("Finish regression training.")
        return self

    def predict(self,x_points,y_points, encoded=True,n_workers=0):
    
        tensor_xs = torch.from_numpy(x_points.astype(np.float32)) 
        y_points = np.asarray(y_points).reshape(-1,1)
         

        # move variables to cuda
        tensor_xs = tensor_xs.to(device)


        if encoded:
            input_dim = len((list(self.dataset.encoders.values())[0].categories_[0]))
        else:
            input_dim = 1
        self.model = self.model.to(device)
        softmax = nn.Softmax(dim=1).to(device)
        pis, sigmas, mus = self.model(tensor_xs)
        pis = softmax(pis)
        pis = pis.cpu()
        sigmas = sigmas.cpu()
        mus = mus.cpu()

        #pis = torch.cat(y_points.shape[0] * [pis])
        #mus = torch.cat(y_points.shape[0] * [mus])
        #sigmas = torch.cat(y_points.shape[0] * [sigmas])
        
        

        mus = mus.squeeze().detach().numpy().reshape(1,-1)
        pis = pis.squeeze().detach().numpy().reshape(1,-1)
        sigmas = sigmas.squeeze().detach().numpy().reshape(1,-1)
        #print (pis.shape, mus.shape, sigmas.shape, y_points.shape)

        probs = np.array(
            [
                np.multiply(stats.norm(mus, sigmas).pdf(y), pis)
                .sum(axis=1)
                .tolist()
                for y in y_points
            ]
        ).transpose()
        #noises = probs < np.max(probs)/7
        #probs[noises] = 0
        return probs


def gaussian_probability(sigma, mu, data):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.
    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """

    data = data.unsqueeze(1).expand_as(sigma)
    ret = (
        1.0
        / math.sqrt(2 * math.pi)
        * torch.exp(-0.5 * ((data - mu) / sigma) ** 2)
        / sigma
    )
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target, device):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    softmax = nn.Softmax(dim=1).to(device)
    pi = softmax(pi)
    
    prob = pi * gaussian_probability(sigma, mu, target)

    nll = -torch.log(torch.sum(prob, dim=1))
    #cross_entropy = -torch.log(torch.add(torch.sum(prob,dim=1), 1e-10))
    #cross_entropy = cross_entropy.to(device)
    return torch.mean(nll)


def MSE_KD_loss(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn, target, device, T = 1):

    softmax = nn.Softmax(dim=1)
    pi_do = softmax(pi_do/T)
    pi_dn = softmax(pi_dn/1)
    prob = torch.sum(pi_do * gaussian_probability(sigma_do, mu_do, target),dim=1)
    prob_d = torch.sum(pi_dn * gaussian_probability(sigma_dn, mu_dn, target),dim=1)
    #prob_d = torch.log(torch.add(pi_dn * gaussian_probability(sigma_dn, mu_dn, target), 1e-7))
    #cross_entropy = -torch.sum(prob*prob_d,dim=1).to(device)
    #cross_entropy = -prob * torch.log(prob_d)
    #cross_entropy = cross_entropy.to(device)
    
    
    cross_entropy = torch.square(prob_d - prob).to(device)
    #cross_entropy = torch.log(torch.max(prob, prob_d)/torch.min(prob, prob_d))
    return torch.mean(cross_entropy)


def MSE_CE_loss(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn, target, device, T = 1): 

    KD = nn.KLDivLoss()(F.log_softmax(pi_dn/T, dim=1),F.softmax(pi_do/T, dim=1)).to(device)

    mse_loss = nn.MSELoss(reduction="mean").to(device)
    mse_mus = mse_loss(mu_dn, mu_do)
    mse_sigmas = mse_loss(sigma_dn, sigma_do)


    kd_loss = KD + mse_mus #+ mse_sigmas
    return torch.mean(kd_loss)


def KS_KD_loss(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn):
    T = 1
    softmax = nn.Softmax(dim=1)
    pi_do = softmax(pi_do/T)
    pi_dn = softmax(pi_dn/T)

    samples1 = MoGsampling(pi_do, sigma_do, mu_do,1000,'cpu')
    samples2 = MoGsampling(pi_dn, sigma_dn, mu_dn,1000,'cpu')   

    stat = []
    for i, row in enumerate(samples1):
        s , p = stats.ks_2samp(row.detach().numpy(), samples2[i].detach().numpy())
        stat.append(s)

    loss = torch.log(Variable(torch.mean(torch.as_tensor(stat)), requires_grad=True)).to('cpu')
    return loss
    

def MoG_kl_divergance(MoG1, MoG2, device):
   
    pis1,sigmas1,mus1 = MoG1[0],MoG1[1],MoG1[2]
    pis2,sigmas2,mus2 = MoG2[0],MoG2[1],MoG2[2]    

    T = 1
    softmax = nn.Softmax(dim=1)
    pis1 = softmax(pis1/T)
    pis2 = softmax(pis2/T)

    samples1 = MoGsampling(pis1, sigmas1, mus1,10000,'cpu')
    samples2 = MoGsampling(pis2, sigmas2, mus2,10000,'cpu')

    samples1 = [i for i in samples1[0]]
    samples2 = [i for i in samples2[0]]

    kl = scipy.stats.entropy(samples1,samples2,base=2)
    return kl


def entropy_with_sampling(pi_do, sigma_do, mu_do, pi_dn, sigma_dn, mu_dn, device):

    T = 1
    softmax = nn.Softmax(dim=1)
    pi_do = softmax(pi_do/T)
    pi_dn = softmax(pi_dn/T)
    
    samples = MoGsampling(pi_do, sigma_do, mu_do, 100, device) 
    #samples_dn = MoGsampling(pi_dn, sigma_dn, mu_dn, 100, device)

    entropies = torch.zeros(samples.shape[0],1)
    for i in range(samples.shape[1]):
        probs_do = torch.sum(pi_do * gaussian_probability(sigma_do, mu_do, samples[:,i].unsqueeze(1)), dim=1) 
        probs_dn = torch.sum(pi_dn * gaussian_probability(sigma_dn, mu_dn, samples[:,i].unsqueeze(1)), dim=1) 

        entropy = -probs_do * torch.log(probs_dn/probs_do)
        #entropy = torch.log(torch.max(probs_dn, probs_do)/torch.min(probs_dn, probs_do))
        #entropy = torch.abs(probs_dn - probs_do)
        entropies = torch.cat([entropies, entropy.unsqueeze(1)],1)

    cross_entropy = torch.sum(entropies, dim=1).to(device)
    return torch.mean(cross_entropy) 


def MoGsampling(pi, sigma, mu, n_samples, device):

    pis = pi

    indices = pis.multinomial(num_samples=n_samples, replacement=True)

    sigma = sigma.reshape(sigma.shape[0], sigma.shape[1])
    mu = mu.reshape(mu.shape[0], mu.shape[1])
    sigmas = torch.gather(sigma, 1, indices)
    mus = torch.gather(mu, 1, indices)
    samples = torch.normal(mus, sigmas)
   
    return samples


def sample(pi, sigma, mu):
    """Draw samples from a MoG."""
    categorical = Categorical(pi)
    pis = list(categorical.sample().data)
    sample = Variable(sigma.data.new(sigma.size(0), sigma.size(2)).normal_())
    for i, idx in enumerate(pis):
        sample[i] = sample[i].mul(sigma[i, idx]).add(mu[i, idx])
    return sample


def gaussion_predict(weights: list, mus: list, sigmas: list, xs: list, n_jobs=1):
    if n_jobs == 1:
        result = np.array(
            [
                np.multiply(stats.norm(mus, sigmas).pdf(x), weights)
                .sum(axis=1)
                .tolist()
                for x in xs
            ]
        ).transpose()
    else:
        with Pool(processes=n_jobs) as pool:
            instances = []
            results = []
            for x in xs:
                i = pool.apply_async(gaussion_predict, (weights, mus, sigmas, [x], 1))
                instances.append(i)
            for i in instances:
                result = i.get()
                # print("partial result", result)
                results.append(result)
            result = np.concatenate(results, axis=1)

            # with futures.ThreadPoolExecutor() as executor:
            #     for x in xs:
            #         future = executor.submit(
            #             gaussion_predict, weights, mus, sigmas, [x], 1)
            #         results.append(future.result())
            # result = np.concatenate(results, axis=1)
    return result

    """given a list of points, calculate the gaussian mixture probability

    Args:
        weights (list): weights
        mus (list): the centroids of gaussions.
        vars (list): the variances.
        x (list): the targeting points.
        b_plot (bool, optional): whether return the value for plotting. Defaults to False.
        n_division (int, optional): number of division, if b_plot=True. Defaults to 100.

    Returns:
        float: the pdf of a gaussian mixture.
    """
    if not b_plot:
        result = [
            stats.norm(mu_i, vars_i).pdf(x) * weights_i
            for mu_i, vars_i, weights_i in zip(mus, vars, weights)
        ]
        result = sum(result)
        # result = 0
        # for index in range(len(weights)):
        #     result += stats.norm(mus[index], vars[index]
        #                          ).pdf(x) * weights[index]
        # print(result)
        return result
    else:
        xs = np.linspace(-1, 1, n_division)
        # ys = [gm(weights, mus, vars, xi, b_plot=False) for xi in xs]
        ys = gm(weights, mus, vars, xs, b_plot=False)
        return xs, ys


def train(mode,name):
   
    if os.path.exists(name):
        print ("Model already exists!")
        return

    print ("Start preparing data ...")
    t1 = time.time()
    d = Dataset.Data('./data/',train_file=None, test_file=None, update_path=None, transfer_set=None, x_attributes = ["country"], y_attributes = ["age"], sep=',')    
    #d.train_update_split(table='census_for_MDN.csv', haveheader=True, permute=True, convert_to_timestamp=False, percentage=0.2,num_update_chunks=1)  
    d.train_file = "data/train_set.csv"
    #d.test_file = "data/test_set.csv"
    d.transfer_set = "data/transfer_set.csv"
    d.update_path = "data/update_batches"
    x_values, y_values = d.read_data(d.train_file, haveheader=True, to_dict=True)

    d.create_encoders(x_values)
    d.create_frequency_tables(x_values)
    d.get_normalizing_stats(data=y_values, min_max=None)#{'ss_sold_date_sk':(2450816.0, 2452642.0)}) 
    y_normalized = d.normalize(y_values)
    x_encoded = {}
    for key in x_values.keys():
        x_encoded[key] = d.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()

    MDN = DenMDN(dataset=d)
    t2 = time.time()
    print ("Data is prepared: \n\t1-Encoders are created, \n\t2-data has been normalized, \n\t3-frequency tables are created ")
    print ("preparation time: {} seconds".format(t2-t1))
    t1 = time.time()
    MDN.fit(x_points = x_encoded[d.x_attributes[0]] ,y_points=y_normalized[d.y_attributes[0]])
    t2 = time.time()
    print ("Training finished in {} seconds".format(t2-t1))
    with open(name,'wb') as dum:
        dill.dump(MDN,dum)


def distilling(basemodel):
    if not os.path.exists(basemodel):
        print ("Model not found!")
        return

    model = None
    with open(basemodel, 'rb') as d:
        model = dill.load(d)


    x_values, y_values = model.dataset.read_data(model.dataset.train_file, haveheader=True, to_dict=True)
    train_set = pd.DataFrame({**x_values, **y_values})
    transfer_data = Dataset.stratified_sample(train_set, model.dataset.x_attributes, size=5000, seed=123, keep_index= False) 
    transfer_data.to_csv(model.dataset.transfer_set, sep=model.dataset.delimiter, index=None)


    pre_model = basemodel
    for num in range(1):
        if num>0:
            with open(pre_model, 'rb') as d:
                model = dill.load(d)

        x_att = model.dataset.x_attributes[0]
        y_att = model.dataset.y_attributes[0]

        
        #x_values_transfer = {x_att:list(model.dataset.FTs[x_att].keys())}
        #y_values_transfer = {y_att:[0]*len(list(model.dataset.FTs[x_att].keys()))}
        x_values_transfer, y_values_transfer = model.dataset.read_data(model.dataset.transfer_set, haveheader=True, to_dict=True)

        x_encoded_transfer = {}
        for key in x_values_transfer.keys():
            x_encoded_transfer[key] = model.dataset.encoders[key].transform(np.asarray(x_values_transfer[key]).reshape(-1,1)).toarray()
        y_normalized_transfer = model.dataset.normalize(y_values_transfer)


        
        t1 = time.time()
        model.distill(x_points_transfer=x_encoded_transfer[x_att], y_points_transfer=y_normalized_transfer[y_att])
        t2 = time.time()
        print ("distilling finished in {} seconds".format(t2-t1))
        
        model_name = 'distilled'+str(num+1).zfill(2)+'.dill'
        with open(model_name,'wb') as d:
            dill.dump(model,d)
        pre_model = model_name


def distill_musample(basemodel):
    if not os.path.exists(basemodel):
        print ("Model not found!")
        return

    model = None
    with open(basemodel, 'rb') as d:
        model = dill.load(d)


    pre_model = basemodel
    for num in range(10):
        if num>0:
            with open(pre_model, 'rb') as d:
                model = dill.load(d)

        x_att = model.dataset.x_attributes[0]
        y_att = model.dataset.y_attributes[0]


        x_values = list(model.dataset.FTs[x_att].keys())
        y_values = [0]*len(x_values)


        x_encoded_transfer = model.dataset.encoders[x_att].transform(np.asarray(x_values).reshape(-1,1)).toarray()


        print ("x_encoded", x_encoded_transfer.shape)
        print ("y_values", len(y_values))
        
        t1 = time.time()
        model.distill(x_points_transfer=x_encoded_transfer, y_points_transfer=y_values)
        t2 = time.time()
        print ("distilling finished in {} seconds".format(t2-t1))
        
        model_name = 'distilled'+str(num+1).zfill(2)+'.dill'
        with open(model_name,'wb') as d:
            dill.dump(model,d)
        pre_model = model_name


def update_with_fusion(basemodel):
    if not os.path.exists(basemodel):
        print ("Model not found!")
        return

    model = None
    with open(basemodel, 'rb') as d:
        model = dill.load(d)

    x_values, y_values = model.dataset.read_data(model.dataset.train_file, haveheader=True, to_dict=True)
    train_set = pd.DataFrame({**x_values, **y_values})
    transfer_data = Dataset.stratified_sample(train_set, model.dataset.x_attributes, size=10000, seed=123, keep_index= True) 
    transfer_data.to_csv(model.dataset.transfer_set, sep=model.dataset.delimiter, index=None)

    print ("Start preparing data ...")
    t1 = time.time()
    update_batches = []
    for u in os.listdir(model.dataset.update_path):
        if u.endswith('.csv') and not ('test' in u):
            update_batches.append(u) 
    update_batches.sort()
    print ("Updates order ==>")
    print (update_batches)

    pre_model = basemodel
    num_gaussians = 40
    hid_neurons = 0
    for num,batch in enumerate(update_batches):
        if num>0:
            with open(pre_model, 'rb') as d:
                model = dill.load(d)

        x_att = model.dataset.x_attributes[0]
        y_att = model.dataset.y_attributes[0]
        #x_values = list(itertools.chain.from_iterable(itertools.repeat(x, 60) for x in model.dataset.FTs[x_att].keys()))
        #x_values_transfer = {x_att:x_values}

        x_values_transfer, y_values_transfer = model.dataset.read_data(model.dataset.transfer_set, haveheader=True, to_dict=True)
        update_file = os.path.join(model.dataset.update_path,batch)
        model.dataset.train_file=update_file
        x_values_update, y_values_update = model.dataset.read_data(update_file, haveheader=True, to_dict=True)

  
        df_tr_tmp = pd.DataFrame.from_dict({**x_values_transfer, **y_values_transfer})
        df_up_tmp = pd.DataFrame.from_dict({**x_values_update, **y_values_update})
        df_up_tmp = Dataset.stratified_sample(df_up_tmp, list(x_values_update.keys()), size=500, seed=123, keep_index= False)#df_up_tmp.sample(frac=0.1)
        df_tr_new = pd.concat([df_tr_tmp,df_up_tmp])
        if num%15 == 0:
            df_tr_new = df_tr_new.sample(n=5000)
        df_tr_new.to_csv(model.dataset.transfer_set,sep=model.dataset.delimiter)

 
        x_encoded_transfer = {}
        for key in x_values_transfer.keys():
            x_encoded_transfer[key] = model.dataset.encoders[key].transform(np.asarray(x_values_transfer[key]).reshape(-1,1)).toarray()
        y_normalized_transfer = model.dataset.normalize(y_values_transfer)
        

        x_encoded_update = {}
        for key in x_values_update.keys():
            x_encoded_update[key] = model.dataset.encoders[key].transform(np.asarray(x_values_update[key]).reshape(-1,1)).toarray()
            freq = Counter(x_values_update[key])
            model.dataset.FTs[key] = dict(freq+Counter(model.dataset.FTs[key]))
        y_normalized_update = model.dataset.normalize(y_values_update)


        """
        tensor_xs = torch.from_numpy(x_encoded_transfer[x_att].astype(np.float32)) 
        tensor_xs = tensor_xs.to(device)
        softmax = nn.Softmax(dim=1)
        pis, sigmas, mus = model.model(tensor_xs)
        pis = softmax(pis)
        pis = pis.cpu()
        sigmas = sigmas.cpu()
        mus = mus.cpu()
        samples = MoGsampling(pis, sigmas, mus,1,'cpu')
        samples = [i[0] for i in samples.tolist()]
        """
        
        print ("Data is prepared: \n\t1-Encoders are created, \n\t2-data has been normalized, \n\t3-frequency tables are created ")
        
        t1 = time.time()
        num_gassians = num_gaussians 
        hid_neurons = hid_neurons
        model.update(x_points_update=x_encoded_update[x_att], y_points_update=y_normalized_update[y_att],
                     x_points_transfer=x_encoded_transfer[x_att], y_points_transfer=y_normalized_transfer[y_att],
                     num_gauss=num_gassians)
        t2 = time.time()
        print ("Updating finished in {} seconds".format(t2-t1))
        
        pre_model = batch.split('.')[0]+'.dill'
        with open(pre_model,'wb') as d:
            dill.dump(model,d)


def update_with_finetune(basemodel):
    if not os.path.exists(basemodel):
        print ("Model not found!")
        return

    model = None
    with open(basemodel, 'rb') as d:
        model = dill.load(d)

    print ("Start preparing data ...")

    update_batches = []
    for u in os.listdir(model.dataset.update_path):
        if u.endswith('.csv') and not ('test' in u):
            update_batches.append(u) 
    update_batches.sort()
    print ("Updates order ==>")
    print (update_batches)

    pre_model = basemodel

    for num,batch in enumerate(update_batches):
        if num>0:
            with open(pre_model, 'rb') as d:
                model = dill.load(d)

        x_att = model.dataset.x_attributes[0]
        y_att = model.dataset.y_attributes[0]


        update_file = os.path.join(model.dataset.update_path,batch)
        model.dataset.train_file=update_file
        x_values_update, y_values_update = model.dataset.read_data(update_file, haveheader=True, to_dict=True)


        x_encoded_update = {}
        for key in x_values_update.keys():
            x_encoded_update[key] = model.dataset.encoders[key].transform(np.asarray(x_values_update[key]).reshape(-1,1)).toarray()
            freq = Counter(x_values_update[key])
            model.dataset.FTs[key] = dict(freq+Counter(model.dataset.FTs[key]))
        y_normalized_update = model.dataset.normalize(y_values_update)


        
        t1 = time.time()

        model.finetune(x_points=x_encoded_update[x_att], y_points=y_normalized_update[y_att])
        t2 = time.time()
        print ("Updating finished in {} seconds".format(t2-t1))

       
        pre_model = batch.split('.')[0]+'.dill'
        with open('finetune'+str(num+1).zfill(2)+'.dill','wb') as d:
            dill.dump(model,d)


def update_FTs(basemodel):
    if not os.path.exists(basemodel):
        print ("Model not found!")
        return

    model = None
    with open(basemodel, 'rb') as d:
        model = dill.load(d)

    print ("Start preparing data ...")

    update_batches = []
    for u in os.listdir(model.dataset.update_path):
        if u.endswith('.csv') and not ('test' in u):
            update_batches.append(u) 
    update_batches.sort()
    print ("Updates order ==>")
    print (update_batches)

    pre_model = basemodel

    for num,batch in enumerate(update_batches):
        if num>0:
            with open(pre_model, 'rb') as d:
                model = dill.load(d)

        x_att = model.dataset.x_attributes[0]
        y_att = model.dataset.y_attributes[0]


        update_file = os.path.join(model.dataset.update_path,batch)
        model.dataset.train_file=update_file
        x_values_update, y_values_update = model.dataset.read_data(update_file, haveheader=True, to_dict=True)


        x_encoded_update = {}
        for key in x_values_update.keys():
            freq = Counter(x_values_update[key])
            model.dataset.FTs[key] = dict(freq+Counter(model.dataset.FTs[key]))

        print (pre_model) 
        pre_model = batch.split('.')[0]+'.dill'
        with open('stale'+str(num+1).zfill(2)+'.dill','wb') as d:
            dill.dump(model,d)


def retrain_all(basemodel):
    #files = ['update01.csv','update02.csv','update03.csv','update04.csv','update05.csv']
    files = ['update01.csv'] 
    model = None
    with open(basemodel, 'rb') as d:
        model = dill.load(d)

    x_values, y_values = model.dataset.read_data(model.dataset.train_file, haveheader=True, to_dict=True)
 
    y_normalized = model.dataset.normalize(y_values)
    x_encoded = {}
    for key in x_values.keys():
        x_encoded[key] = model.dataset.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()


    for i, file in enumerate(files):
        t1 = time.time()
        x_tmp, y_tmp = model.dataset.read_data('data/update_batches/'+file, haveheader=True, to_dict=True)
        update_file = os.path.join(model.dataset.update_path,file)
        model.dataset.train_file=update_file
        for key in x_tmp.keys():
            x_values[key].extend(x_tmp[key])
            freq = Counter(x_tmp[key])
            model.dataset.FTs[key] = dict(freq+Counter(model.dataset.FTs[key]))

        for key in y_tmp.keys():
            y_values[key].extend(y_tmp[key])



        y_normalized = model.dataset.normalize(y_values)
        x_encoded = {}
        for key in x_values.keys():
            x_encoded[key] = model.dataset.encoders[key].transform(np.asarray(x_values[key]).reshape(-1,1)).toarray()


        model.fit(x_points = x_encoded[model.dataset.x_attributes[0]] ,y_points=y_normalized[model.dataset.y_attributes[0]])
        
        t2 = time.time()
        print ("Training finished in {} seconds".format(t2-t1))
        
        with open('retrain'+str(i+1).zfill(2)+'.dill','wb') as dum:
            dill.dump(model,dum)



if __name__=="__main__":
    
    #train(mode="train", name='base.dill')
    #distilling(basemodel='DenMDN_census.dill')
    #distill_musample(basemodel='DenMDN_census.dill')
    #update_with_fusion(basemodel="base.dill")
    #update_with_finetune(basemodel="base.dill")
    #update_FTs(basemodel="base.dill")
    retrain_all(basemodel="base.dill")
