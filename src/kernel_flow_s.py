# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 07:40:37 2022

@author: Silver
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src import kernel_zoo
import tqdm
import copy
import os,random

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	# torch.cuda.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True

seed_torch()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def torch2D_Hausdorff_distance(x,y): # Input be like (Batch,width,height)
    x = x.float()
    y = y.float()
    distance_matrix = torch.cdist(x,y,p=2) # p=2 means Euclidean Distance
    
    value1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    value2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]
    
    value = torch.cat((value1, value2), dim=1)
    
    return value.max(1)[0]

def sample_selection(N, size):
    indices = np.arange(N)
    sample_indices = np.sort(np.random.choice(indices, size, replace= False))
    #sample_indices = indices[:size]
    return sample_indices

# The pi or selection matrix
def pi_matrix(sample_indices, dimension):
    pi = torch.zeros(dimension)
    
    for i in range(dimension[0]):
        pi[i][sample_indices[i]] = 1
    
    return pi

def batch_creation(N, batch_size, sample_proportion = 0.5):
    if batch_size == False:
        batch_indices = np.arange(N)
    elif 0 < batch_size <= 1:
        batch_size = int(N * batch_size)
        batch_indices = sample_selection(N, batch_size)
    else:
        batch_indices = sample_selection(N, batch_size)
        
    # Sample from the mini-batch
    sample_size = math.ceil(len(batch_indices)*sample_proportion)
    sample_indices = sample_selection(len(batch_indices), sample_size)
    
    return sample_indices, batch_indices

class KernelFlows(torch.nn.Module):
    
    def __init__(self, kernel_keyword, nparameters, nelements, dim, pred_len, metric="rho_ratio", batch_size=100, regu_lambda=None, lasso_gamma=None, device=None):
        super().__init__()
        self.kernel_keyword = kernel_keyword
        
        self.regu_lambda = regu_lambda
        self.lasso_gamma = lasso_gamma

        self.active_func = F.relu if self.lasso_gamma > 0 else F.sigmoid

        self.device = device
        if self.kernel_keyword == 'river':
            self.kernel = kernel_zoo.kernel_river_multi
        elif self.kernel_keyword == 'anl3':
            self.kernel = kernel_zoo.kernel_anl3_multi
        elif self.kernel_keyword == 'guass':
            self.kernel = kernel_zoo.kernel_gaussian_multi
            
        self.kernel_elements = torch.nn.Parameter(torch.empty((pred_len, nelements)).uniform_(0.1, 2), requires_grad=True)
        self.kernel_params = torch.nn.Parameter(torch.empty((pred_len, nparameters)).uniform_(0.1, 5), requires_grad=True)

        self.dim = dim
        self.pred_len = pred_len
        self.nelements = nelements
        self.nparameters = nparameters
        self.batch_size = batch_size

        if metric == "rho_ratio":
            self.rho_fun = self.rho_ratio
        elif metric == "rho_general":
            self.rho_fun = self.rho_general
        elif metric == "rho_general_l1":
            self.rho_fun = self.rho_general_l1
        elif metric == "rho_ratio_l1":
            self.rho_fun = self.rho_ratio_l1
        elif metric == "rho_hausdorff":
            self.rho_fun = self.rho_hausdorff
        elif metric == "rho_hausdorff_l1":
            self.rho_fun = self.rho_hausdorff_l1
        else:
            raise("Metric not supported")
        self.metric = metric

    def get_parameters(self):
        return (params for params in [self.kernel_elements, self.kernel_params])
    
    def set_train(self, train):
        self.train = train

    def set_training_data(self,X,Y):
        if not self.device is None:
            X = X.to(self.device)
            Y = Y.to(self.device)
        self.X_train = X
        self.Y_train = Y

    def prepare_semi_group(self,n_z, max_delay, device):
        
        random_idx = np.random.choice(self.X_train.shape[0],n_z, replace = False)
        random_delays_pre = torch.Tensor(np.random.randint(max_delay,size=n_z)+1, device = device)
        random_delays_post = torch.Tensor(np.random.randint(max_delay,size=n_z)+1, device = device)

        self.X_train_phi2 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]),-1)
        
        self.z_tensor = torch.nn.Parameter(torch.randn(n_z,self.dim),requires_grad = True)

        self.X_train_phi3 = torch.cat((self.z_tensor,random_delays_post[...,None]),-1) -  torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)

        self.X_train = torch.cat((self.X_train,self.X_train_phi2,self.X_train_phi3))
        self.Y_train  = torch.cat((self.Y_train,self.z_tensor,torch.zeros_like(self.z_tensor, device = device)))

    def prepare_semi_group_new(self,n_z, delays_train):
        
        random_idx = np.random.choice(self.X_train.shape[0]-1,n_z, replace = False)
        random_delays_pre = delays_train[random_idx]
        random_delays_post = delays_train[random_idx+1]

        self.X_train_phi2 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]),-1)
        
        self.z_tensor = self.X_train[random_idx+1,:-1] 
        
        self.X_train_phi3 = torch.cat((self.z_tensor,random_delays_post[...,None]),-1) -  torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)
        #self.X_train_phi3 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)

        self.X_train = torch.cat((self.X_train,self.X_train_phi2,self.X_train_phi3))
        self.Y_train  = torch.cat((self.Y_train,self.z_tensor,torch.zeros_like(self.z_tensor)))

    def prepare_semi_group_simple(self,n_z, delays_train):
        
        random_idx = np.random.choice(self.X_train.shape[0]-2,n_z, replace = False)
        random_delays_pre = delays_train[random_idx]
        random_delays_post = delays_train[random_idx+1]

        self.X_train_phi2 = torch.cat((self.X_train[random_idx,:-1],random_delays_pre[...,None]+random_delays_post[...,None]),-1)
        
        self.z_tensor = self.X_train[random_idx+2,:-1] 

        self.X_train = torch.cat((self.X_train,self.X_train_phi2))
        self.Y_train  = torch.cat((self.Y_train,self.z_tensor))

    def rho_ratio(self, matrix_data, Y_data, sample_indices,  regu_lambda=0.000001, **kwargs):

        kernel_var = torch.cat([self.kernel_elements, self.kernel_params],dim=1)

        kernel_matrix = self.kernel(matrix_data, matrix_data, kernel_var, self.active_func).to(self.device)
        # kernel_matrix = kernel_matrix.reshape(-1,kernel_matrix.shape[0],kernel_matrix.shape[1])
        
        pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0])).to(self.device) 
        
        sample_matrix = torch.matmul(pi, torch.matmul(kernel_matrix, torch.transpose(pi,0,1)))

        inverse_data = torch.linalg.pinv(kernel_matrix.to(self.device) + \
                       self.regu_lambda * torch.eye(kernel_matrix.shape[1]).to(self.device))
        inverse_sample = torch.linalg.pinv(sample_matrix.to(self.device) + \
                         self.regu_lambda * torch.eye(sample_matrix.shape[1]).to(self.device))

        Y_sample = Y_data[sample_indices]

        Y_data = Y_data.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)
        Y_sample = Y_sample.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)

        top = torch.tensordot(Y_sample, torch.matmul(inverse_sample, Y_sample), dims=([1,2],[1,2]))
        bottom = torch.tensordot(Y_data, torch.matmul(inverse_data, Y_data), dims=([1,2],[1,2]))
        
        return (1 - top/bottom).sum()

    def rho_ratio_l1(self, matrix_data, Y_data, sample_indices,  regu_lambda = 0.05, lasso_gamma = 0.000001, is_detach=False, **kwargs):
        
        kernel_var = torch.cat([self.kernel_elements, self.kernel_params],dim=1)

        kernel_matrix = self.kernel(matrix_data, matrix_data, kernel_var,self.active_func).to(self.device)
        
        pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0])).to(self.device) 
        
        sample_matrix = torch.matmul(pi, torch.matmul(kernel_matrix, torch.transpose(pi,0,1)))

        inverse_data = torch.linalg.pinv(kernel_matrix.to(self.device) + \
                       self.regu_lambda * torch.eye(kernel_matrix.shape[1]).to(self.device))
        inverse_sample = torch.linalg.pinv(sample_matrix.to(self.device) + \
                         self.regu_lambda * torch.eye(sample_matrix.shape[1]).to(self.device))

        Y_sample = Y_data[sample_indices]

        Y_data = Y_data.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)
        Y_sample = Y_sample.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)

        top = torch.tensordot(Y_sample, torch.matmul(inverse_sample, Y_sample), dims=([1,2],[1,2]))
        bottom = torch.tensordot(Y_data, torch.matmul(inverse_data, Y_data), dims=([1,2],[1,2]))
        
        k_ele = (1-torch.exp(self.kernel_elements)**(-1)) if self.lasso_gamma > 0  else self.kernel_elements # self.kernel_elements  # 
        l1_comp = (self.active_func(k_ele)).sum()
            
        return ((1 - top/(bottom+1e-08))**2).sum() + l1_comp*lasso_gamma

    def rho_hausdorff(self, matrix_data, Y_data, sample_indices,  regu_lambda = 0.000001, lasso_gamma = 0.000001, is_detach=False, **kwargs):
        
        kernel_var = torch.cat([self.kernel_elements, self.kernel_params],dim=1)

        kernel_matrix = self.kernel(matrix_data, matrix_data, kernel_var, self.active_func).to(self.device)
        # kernel_matrix = kernel_matrix.reshape(-1,kernel_matrix.shape[0],kernel_matrix.shape[1])
        
        pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0])).to(self.device) 
        
        sample_matrix = torch.matmul(pi, torch.matmul(kernel_matrix, torch.transpose(pi,0,1)))

        inverse_data = torch.linalg.pinv(kernel_matrix.to(self.device) + \
                       self.regu_lambda * torch.eye(kernel_matrix.shape[1]).to(self.device))
        inverse_sample = torch.linalg.pinv(sample_matrix.to(self.device) + \
                         self.regu_lambda * torch.eye(sample_matrix.shape[1]).to(self.device))

        Y_sample = Y_data[sample_indices]
        matrix_data_sample = matrix_data[sample_indices]
        index = torch.LongTensor(random.sample(range(matrix_data.shape[0]), int(matrix_data.shape[0]*0.7)))
        X_2_hat = torch.index_select(matrix_data, 0, index)

        Y_data = Y_data.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)
        Y_sample = Y_sample.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)

        X_kernel_prod = self.kernel(X_2_hat, matrix_data, kernel_var, self.active_func).to(self.device) # torch.Size([1, 61, 100])  torch.Size([1, 61, 50])
        X_sample_kernel_prod = self.kernel(X_2_hat, matrix_data_sample, kernel_var, self.active_func).to(self.device)

        bottom = torch.squeeze(torch.tensordot(X_sample_kernel_prod, torch.matmul(inverse_sample, Y_sample), dims=([-1,],[1,])), dim=2)
        top = torch.squeeze(torch.tensordot(X_kernel_prod, torch.matmul(inverse_data, Y_data), dims=([-1,],[1,])), dim=2)

        rho = torch2D_Hausdorff_distance(bottom,top).mean() 
        return rho

    def rho_hausdorff_l1(self, matrix_data, Y_data, sample_indices,  regu_lambda = 0.000001, lasso_gamma = 0.000001, is_detach=False, **kwargs):
        
        kernel_var = torch.cat([self.kernel_elements, self.kernel_params],dim=1)

        kernel_matrix = self.kernel(matrix_data, matrix_data, kernel_var, self.active_func).to(self.device)
        # kernel_matrix = kernel_matrix.reshape(-1,kernel_matrix.shape[0],kernel_matrix.shape[1])
        
        pi = pi_matrix(sample_indices, (sample_indices.shape[0], matrix_data.shape[0])).to(self.device) 
        
        sample_matrix = torch.matmul(pi, torch.matmul(kernel_matrix, torch.transpose(pi,0,1)))

        inverse_data = torch.linalg.pinv(kernel_matrix.to(self.device) + \
                       self.regu_lambda * torch.eye(kernel_matrix.shape[1]).to(self.device))
        inverse_sample = torch.linalg.pinv(sample_matrix.to(self.device) + \
                         self.regu_lambda * torch.eye(sample_matrix.shape[1]).to(self.device))

        Y_sample = Y_data[sample_indices]
        matrix_data_sample = matrix_data[sample_indices]
        index = torch.LongTensor(random.sample(range(matrix_data.shape[0]), int(matrix_data.shape[0]*0.7)))
        X_2_hat = torch.index_select(matrix_data, 0, index)

        Y_data = Y_data.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)
        Y_sample = Y_sample.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)

        X_kernel_prod = self.kernel(X_2_hat, matrix_data, kernel_var, self.active_func).to(self.device) # torch.Size([1, 61, 100])  torch.Size([1, 61, 50])
        X_sample_kernel_prod = self.kernel(X_2_hat, matrix_data_sample, kernel_var, self.active_func).to(self.device)

        bottom = torch.squeeze(torch.tensordot(X_sample_kernel_prod, torch.matmul(inverse_sample, Y_sample), dims=([-1,],[1,])), dim=2)
        top = torch.squeeze(torch.tensordot(X_kernel_prod, torch.matmul(inverse_data, Y_data), dims=([-1,],[1,])), dim=2)

        rho = torch2D_Hausdorff_distance(bottom,top).mean()
        
        k_ele = (1-torch.exp(self.kernel_elements)**(-1)) if self.lasso_gamma > 0  else self.kernel_elements  #self.kernel_elements  # 
        l1_comp = (self.active_func(k_ele)).sum()
            
        return rho + l1_comp*lasso_gamma
        #(1 - top/bottom).sum() + torch.sum((self.kernel_elements**2))*lasso_gamma
    
    def rho_general(self, matrix_data, Y_data,  regu_lambda = 0.000001, **kwargs):
        
        kernel_matrix = self.kernel(matrix_data, matrix_data, self.kernel_params, self.active_func)
        kernel_matrix = kernel_matrix / torch.trace(kernel_matrix)
        inverse_matrix = torch.linalg.pinv(kernel_matrix + self.regu_lambda * torch.eye(kernel_matrix.shape[0]))
        rho = torch.tensordot(Y_data, torch.matmul(inverse_matrix, Y_data))
        
        return rho

    def rho_general_l1(self, matrix_data, Y_data,  regu_lambda = 0.000001, lasso_gamma = 0.000001, ns = None, **kwargs):
        
        kernel_matrix = self.kernel(matrix_data, matrix_data, self.kernel_params, self.active_func)
        kernel_matrix = kernel_matrix / torch.trace(kernel_matrix)
        inverse_matrix = torch.linalg.pinv(kernel_matrix + self.regu_lambda * torch.eye(kernel_matrix.shape[0]))
        rho = torch.tensordot(Y_data, torch.matmul(inverse_matrix, Y_data)) + torch.sum(((self.kernel_params[[1,3,6,9,11,12,17,20]])**2)**(1/4))*lasso_gamma
        
        return rho

    def init_forward(self, adaptive_size = "Dynamic", proportion = 0.5, batch_size_train=1):
        if adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
        rho = 0
        for i in range(batch_size_train):  
            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(self.X_train.shape[0], batch_size= self.batch_size, sample_proportion = sample_size)
            X_data = self.X_train[batch_indices]
            Y_data = self.Y_train[batch_indices]

            #optimizer and backward
            rho_now = self.rho_fun( X_data, Y_data, \
                sample_indices = sample_indices, regu_lambda = self.regu_lambda, lasso_gamma = self.lasso_gamma)
                
            rho= rho + rho_now
           
        return rho/batch_size_train

    def forward(self, adaptive_size = "Dynamic", proportion = 0.5, batch_size_train=1, is_detach=False):            
        
        if adaptive_size == "Dynamic":
            sample_size = proportion
        elif adaptive_size == "Linear":
            sample_size_array = sample_size_linear(iterations, adaptive_range) 
        else:
            print("Sample size not recognized")
            
        rho = 0
        for i in range(batch_size_train):  
            # Create a batch and a sample
            sample_indices, batch_indices = batch_creation(self.X_train.shape[0], batch_size= self.batch_size, sample_proportion = sample_size)
            X_data = self.X_train[batch_indices]
            Y_data = self.Y_train[batch_indices]
        
            #optimizer and backward
            rho_now = self.rho_fun( X_data, Y_data, \
                sample_indices=sample_indices, regu_lambda=self.regu_lambda, lasso_gamma=self.lasso_gamma, is_detach=is_detach)
                
            rho= rho + rho_now
           
        return rho/batch_size_train

    def compute_kernel_and_inverse(self,regu_lambda = 0.0000001):
        torch.cuda.empty_cache()
        X_data = self.X_train
        #print('X_data:',X_data)
        kernel_var = torch.cat([self.kernel_elements, self.kernel_params],dim=1)
        self.kernel_matrix = self.kernel(X_data,X_data, kernel_var, self.active_func).to(self.device)
        # self.kernel_matrix = self.kernel_matrix.reshape(-1,self.kernel_matrix.shape[0],self.kernel_matrix.shape[1])

        self.kernel_matrix += self.regu_lambda * torch.eye(self.kernel_matrix.shape[1]).to(self.device)
        
        self.inverse_kernel = torch.linalg.pinv(self.kernel_matrix)

        Y_train_p = self.Y_train.reshape((-1,self.pred_len,self.dim)).permute(1,0,2)

        self.A_matrix = torch.matmul(self.inverse_kernel,Y_train_p)
        torch.cuda.empty_cache()


    def predict(self,x_test):
        torch.cuda.empty_cache()
        kernel_var = torch.cat([self.kernel_elements, self.kernel_params],dim=1)
        kernel_pred = self.kernel(x_test,self.X_train,kernel_var, self.active_func)
        # kernel_pred = kernel_pred.reshape(-1,kernel_pred.shape[0],kernel_pred.shape[1])
        prediction = torch.matmul(kernel_pred.to(self.device),self.A_matrix)
        return prediction

    def predict_ahead(self, x_test, horizon, delay, pre_steps = None, delta_t_mode = False, device = None):
        torch.cuda.empty_cache()
        """
        Perform n=horizon steps ahead prediction.

        If delta_t_mode is True, x_test is expected to have the the following structure (X(t-1),delta_t-1,X(t),delta_t))
        
        out_dim is the dimension of the y vector (and of the observations in x as well)

        delay : delay used in the x
        """
        if pre_steps is None:
            pre_steps = self.pred_len-1


        assert horizon >0 # minimum horizon is 1
        assert delay >0 

        if not self.device is None:
            device = self.device

        Y_p = torch.zeros((self.pred_len,x_test.shape[0],self.dim))
        X_test_ = torch.Tensor(x_test).to(device)

        if delta_t_mode:
            indices_delays = [((self.dim+1)*i,(self.dim+1)*i+1) for i in range(delay)] # We should not touch the delta t
        else:
            indices_delays = [(self.dim*i,self.dim*i+1) for i in range(delay)]
        
        # Make sure there is no contamination (deleting the previous values)
        for dim in range(horizon):
            n_delays = min(dim,delay)
            for n in range(1,n_delays+1):
                X_test_[dim::horizon][:,indices_delays[-n]] = 0

        # Predicting and propagating predictions to the next step.
        for dim in range(horizon):
    
            Y_p[:,dim::horizon] = self.predict(X_test_[dim::horizon])
            
            for dim_plus in range(dim+1,min(horizon,delay+dim+1)):
                
                l_x = X_test_[dim_plus::horizon].shape[0]
                idx = dim_plus-dim

                Y_p_trans = Y_p[:,dim::horizon]
                # print(Y_p_trans.shape)
                Y_p_trans = Y_p_trans.permute(1,0,2).reshape(Y_p[:,dim::horizon].shape[1],-1)
                # print(Y_p_trans[:3,:])

                X_test_[dim_plus::horizon][:,indices_delays[-1*idx][0]:(self.dim-1)+indices_delays[-1*idx][1]] = Y_p_trans[:l_x,:] # should be -1-2:-1 for irregular

        return Y_p
    

def train_kernel(X_train, Y_train, model,  lr = 0.1, verbose= False, th=5e-2):
    """
    dim is the dimension of a single observation
    """ 
    model.set_training_data(torch.Tensor(X_train),torch.Tensor(Y_train))

    rho_list = []
    w = None

    nan_num = 0
    

    if model.lasso_gamma <= 0:
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        for i in range(7000):
            optimizer.zero_grad()
            rho = model.forward()
            rho_list.append(rho)
            if rho>=0:
                rho.backward()
                optimizer.step()
                if verbose:
                    print(rho)
        return model,rho_list

    else:
        for num, p in enumerate(model.parameters()):
            if (num==0):
                p.requires_grad = True
        filtered_parameters_init = []
        for p in filter(lambda p: p.requires_grad, model.parameters()):
            filtered_parameters_init.append(p)
        optimizer = torch.optim.SGD(filtered_parameters_init, lr = lr)

        for epoch_i in range(14):

            mode_num = epoch_i%2
            epoch = epoch_i
            
            opt_layer = 0 if epoch%2 == 0 else 1
            iter_num = 500 if epoch%2 == 0 else 500
            for num, p in enumerate(model.parameters()):
                if (num != opt_layer)&(num==mode_num): # 
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            filtered_parameters = []
            for p in filter(lambda p: p.requires_grad, model.parameters()):
                filtered_parameters.append(p)
            optimizer = torch.optim.SGD(filtered_parameters, lr = lr)
        
            nan_num_0 = 0
            for i in range(iter_num):
                optimizer.zero_grad()
                rho = model.forward(batch_size_train=2)
                if (torch.isnan(rho).any()) | (torch.isinf(rho).any()):
                    nan_num_0 = nan_num_0 + 1
                    if nan_num_0>=10:
                        return model,rho_list
                rho_list.append(rho)
                if rho>=0:
                    rho.backward()
                    optimizer.step()
                    if verbose:
                        print(rho)
                
    return model, rho_list