# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:52:54 2023

@author: sunxiuwenO2
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy


class KernelRegister(dict):
    def __init__(self, *args, **kwargs):
        super(KernelRegister, self).__init__(*args, **kwargs)
        self._dict = {}
    
    def register(self, target):
        def add_register_item(key, value):
            if not callable(value):
                raise Exception(f"register object must be callable! But receice:{value} is not callable!")
            if key in self._dict:
                print(f"warning: \033[33m{value.__name__} has been registered before, so we will overriden it\033[0m")
            self[key] = value
            return value

        if callable(target):     
            return add_register_item(target.__name__, target)
        else:                 
            return lambda x : add_register_item(target, x)
    
    def __call__(self, target):
        return self.register(target)
    
    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]
    
    def __contains__(self, key):
        return key in self._dict
    
    def __str__(self):
        return str(self._dict)
    
    def keys(self):
        return self._dict.keys()
    
    def values(self):
        return self._dict.values()
    
    def items(self):
        return self._dict.items()

kernel_register = KernelRegister()


class KernelRiverElement():
    def __init__(self, kernel_list, kernel_register=kernel_register):

        self.kernel_list = kernel_list
        self.throughput = len(self.kernel_list)
        self.num_array = np.array([kernel_register[kernel_elem].param_num for \
                                    kernel_elem in kernel_list])
        self.cumsum_array = np.insert(self.num_array.cumsum(axis=0), 0, 0)
        self.fishshoal = self.num_array.sum()




def dispatch(cls, vct):
    '''

    '''
    if cls.has_pattern:
        vct_h = cls.vct_h
    elif isinstance(vct, tuple):
        vct_h, vct = vct
        vct_h = vct_h.transpose(-1,-2) [...,None,:,:]
    else:
        vct_h = vct.transpose(-1,-2) [...,None,:,:]
    return vct_h, vct

def norm_measure(vct_h, vct):
    diff = vct_h - vct[...,None]
    '''

    '''
    norm = torch.linalg.norm(diff, dim=-2)
    return norm

def innerprod_measure(vct_h, vct):
    vct_h = vct_h.squeeze(dim=-3)
    inner = torch.matmul(vct, vct_h).squeeze(0)
    return inner


# ==========================================
#                
# ==========================================


@kernel_register.register('gaussian')
class gaussian_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(gaussian_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum = nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2

        kernel_matrix = torch.exp(-quad/(2*self.ori_kernel_fish_posnum**2 + 1e-40))

        return kernel_matrix



@kernel_register.register('laplacianl1')
class laplacian_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(laplacian_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)

        kernel_matrix = torch.exp(-norm/(2*self.ori_kernel_fish_posnum + 1e-40))

        return kernel_matrix



@kernel_register.register('expsinexp')
class expsinexp_kernel_layer(nn.Module):
    param_num = 3
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(expsinexp_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish = nn.Parameter(torch.tensor(1.2), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.ones(2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2

        kernel_matrix = torch.exp(-torch.sin(quad*self.ori_kernel_fish)**2/\
                                  (self.ori_kernel_fish_posnum[0]+1e-40))*\
                                  torch.exp(-quad/(self.ori_kernel_fish_posnum[1]+1e-40))

        return kernel_matrix



@kernel_register.register('expsin')
class expsin_kernel_layer(nn.Module):
    param_num = 2
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(expsin_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish = nn.Parameter(torch.tensor(1.2), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2
        
        kernel_matrix = torch.exp(-torch.sin(quad*self.ori_kernel_fish)**2/\
                                  (self.ori_kernel_fish_posnum+1e-40))
        
        return kernel_matrix



@kernel_register.register('expsinexpl1')
class expsinexpl1_kernel_layer(nn.Module):
    param_num = 3
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(expsinexpl1_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish = nn.Parameter(torch.tensor(1.2), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.ones(2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)

        kernel_matrix = torch.exp(-torch.sin(norm*self.ori_kernel_fish)**2/\
                                  (self.ori_kernel_fish_posnum[0]+1e-40))*\
                                  torch.exp(-norm/(self.ori_kernel_fish_posnum[1]+1e-40))

        return kernel_matrix



@kernel_register.register('expsinl1')
class expsinl1_kernel_layer(nn.Module):
    param_num = 2
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(expsinl1_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish = nn.Parameter(torch.tensor(1.2), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        
        kernel_matrix = torch.exp(-torch.sin(norm*self.ori_kernel_fish)**2/\
                                  (self.ori_kernel_fish_posnum+1e-40))
        
        return kernel_matrix


@kernel_register.register('multiquad')
class multiquad_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(multiquad_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2
        
        kernel_matrix = (quad/(self.ori_kernel_fish_posnum + 1e-40) + 1)**(1/2)
        
        return kernel_matrix


@kernel_register.register('invmultiquad')
class invmultiquad_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(invmultiquad_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2
        
        kernel_matrix = (quad/(self.ori_kernel_fish_posnum + 1e-40) + 1)**(-1/2)
        
        return kernel_matrix



@kernel_register.register('invmultiquadl1')
class invmultiquadl1_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(invmultiquadl1_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)

        kernel_matrix = (norm/(self.ori_kernel_fish_posnum + 1e-40) + 1)**(-1/2)
        
        return kernel_matrix


@kernel_register.register('invmultipower')
class invmultipower_kernel_layer(nn.Module):
    param_num = 2
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(invmultipower_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
        self.ori_kernel_fish_posint = nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2

        kernel_matrix = (quad/(self.ori_kernel_fish_posnum + 1e-40) + 1)**\
            (-torch.ceil(torch.abs(self.ori_kernel_fish_posint)))
        
        return kernel_matrix


@kernel_register.register('invmultipowerl1')
class invmultipowerl1_kernel_layer(nn.Module):
    param_num = 2
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(invmultipowerl1_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
        self.ori_kernel_fish_posint = nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)

        kernel_matrix = (norm/(self.ori_kernel_fish_posnum + 1e-40) + 1)**\
            (-torch.ceil(torch.abs(self.ori_kernel_fish_posint)))
        
        return kernel_matrix


@kernel_register.register('cauchy')
class cauchy_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(cauchy_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2

        kernel_matrix = 1/(1+quad/(self.ori_kernel_fish_posnum) + 1e-40)
        
        return kernel_matrix


@kernel_register.register('cauchyl1')
class cauchyl1_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(cauchyl1_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)

        kernel_matrix = 1/(1+norm/(self.ori_kernel_fish_posnum + 1e-40))
        
        return kernel_matrix



@kernel_register.register('rationquadratic')
class rationquadratic_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(rationquadratic_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2

        kernel_matrix = 1-quad/(quad + self.ori_kernel_fish_posnum + 1e-40)
        
        return kernel_matrix


@kernel_register.register('unitcircle')
class unitcircle_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(unitcircle_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        quad = norm**2

        kernel_matrix = torch.maximum(torch.zeros(1,device=quad.device), \
                                      1-quad/(self.ori_kernel_fish_posnum + 1e-40))
        
        return kernel_matrix


@kernel_register.register('unitmanhattan')
class unitmanhattan_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(unitmanhattan_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)

        kernel_matrix = torch.maximum(torch.zeros(1,device=norm.device), \
                                      1-norm/(self.ori_kernel_fish_posnum + 1e-40))
        
        return kernel_matrix



@kernel_register.register('log')
class log_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(log_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posint = nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)
        
        kernel_matrix = torch.log(norm**torch.ceil(torch.abs(self.ori_kernel_fish_posint))+1)
        
        return kernel_matrix



@kernel_register.register('circular')
class circular_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(circular_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum =  nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        norm = norm_measure(vct_h, vct)

        circ_sigma = self.ori_kernel_fish_posnum + 1e-40
        candidate_K = (torch.arccos(-norm/circ_sigma) - \
                       norm/circ_sigma*torch.sqrt(1-(norm/(circ_sigma))**2))
        kernel_matrix = torch.where(norm<(circ_sigma+1e-40), candidate_K, 0)
        
        return kernel_matrix



# ==========================================
#                   
# ==========================================



@kernel_register.register('linear')
class linear_kernel_layer(nn.Module):
    param_num = 1
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(linear_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum = nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        inner = innerprod_measure(vct_h, vct)

        kernel_matrix = inner + self.ori_kernel_fish_posnum

        return kernel_matrix


@kernel_register.register('polynomial')
class polynomial_kernel_layer(nn.Module):
    param_num = 3
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(polynomial_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum = nn.Parameter(torch.ones(2), requires_grad=True)
        self.ori_kernel_fish_posint = nn.Parameter(torch.tensor(1.2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        inner = innerprod_measure(vct_h, vct)

        kernel_matrix = (inner*self.ori_kernel_fish_posnum[0] + self.ori_kernel_fish_posnum[1])**\
            torch.ceil(torch.abs(self.ori_kernel_fish_posint))

        return kernel_matrix


@kernel_register.register('sigmiod')
class sigmiod_kernel_layer(nn.Module):
    param_num = 2
    def __init__(self, input_dims=1, output_dims=1, has_pattern=False, **kwargs):
        super(sigmiod_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)
        self.ori_kernel_fish_posnum = nn.Parameter(torch.ones(2), requires_grad=True)
    
    def forward(self, vct):
        vct_h, vct = dispatch(self, vct)
        inner = innerprod_measure(vct_h, vct)

        kernel_matrix = torch.tanh(self.ori_kernel_fish_posnum[0]*inner + \
                                   self.ori_kernel_fish_posnum[1])

        return kernel_matrix


# The Volterra reservoir kernel
# torch.exp(-quad/(2*self.ori_kernel_fish_posnum**2 + 1e-40))

@kernel_register.register('volterra_reservoir')
class voltreservoir_kernel_layer(nn.Module):
    param_num = 2
    def __init__(self, input_dims=1, output_dims=1, point_dim=3, has_pattern=False, **kwargs):
        self.dc = copy.deepcopy
        super(voltreservoir_kernel_layer, self).__init__()
        self.has_pattern = has_pattern
        self.point_dim = point_dim
        if self.has_pattern:
            self.vct_h = nn.Parameter(torch.ones((input_dims, output_dims)), requires_grad=True)

        if 'M_Kinf' in kwargs:
            print(kwargs['M_Kinf'])
            self.M_Kinf = kwargs['M_Kinf']
        else:
            self.M_Kinf = torch.ones(1)

        tau = torch.sqrt(1/(self.M_Kinf**2)) * 0.99
        lambda_val = torch.sqrt(1-(tau**2)*(self.M_Kinf**2))*0.99

        self.ori_kernel_fish_posnum = nn.Parameter(torch.tensor([tau, lambda_val]), requires_grad=True)
        self.ori_kernel_fish_posnum_clamp_min = torch.zeros(2) + 1e-16
        self.ori_kernel_fish_posnum_clamp_max = torch.ones(2)
        
    
    def forward(self, vct):
        # gn = 1
        vct_h, vct = dispatch(self, vct)
        M_Kinf = self.dc(self.M_Kinf)

        vct_h_tuple = torch.split(vct_h, self.point_dim, dim=-2)
        vct_tuple = torch.split(vct, self.point_dim, dim=-1)

        # torch.exp(-norm_measure(vct_h_item, vct_item)/(8*M_Kinf**2+1e-40))
        # M_Kinf-norm_measure(vct_h_item, vct_item)
        # innerprod_measure(vct_h_item, vct_item)

        inner_prod = torch.stack([M_Kinf-norm_measure(vct_h_item, vct_item) for \
                                       vct_h_item, vct_item in zip(vct_h_tuple, vct_tuple)], dim=0)
        inner_reservoir = 1/(1-(self.ori_kernel_fish_posnum[0]**2*inner_prod))

        # K_Volt_sum = None
        K_Volt_sum = self.ori_kernel_fish_posnum[1]**(2*(1)) * torch.prod(inner_reservoir[0:1,...], dim=0)

        for k in range(inner_reservoir.shape[0]-1): #
            K_Volt_sum += self.ori_kernel_fish_posnum[1]**(2*(k+2)) * torch.prod(inner_reservoir[0:k+2,...], dim=0)

        kernel_matrix = 1 + K_Volt_sum

        self.ori_kernel_fish_posnum_clamp_max[0] = 1/M_Kinf * 0.999
        self.ori_kernel_fish_posnum_clamp_max[1] = (1-self.ori_kernel_fish_posnum[0]**2*\
                                             M_Kinf**2)**(1/2) * 0.999

        return kernel_matrix

