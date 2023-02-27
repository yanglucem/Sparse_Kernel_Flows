import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# torch.set_default_tensor_type(torch.FloatTensor)
#%% Kernel operations

# Returns the norm of the pairwise difference
def norm_matrix(matrix_1, matrix_2):
    norm_square_1 = torch.sum(torch.square(matrix_1), axis = 1)
    norm_square_1 = torch.reshape(norm_square_1, (-1,1))
    
    norm_square_2 = torch.sum(torch.square(matrix_2), axis = 1)
    norm_square_2 = torch.reshape(norm_square_2, (-1,1))
    
    d1=matrix_1.shape
    d2=matrix_2.shape
#    print(d1)
#    print(d2)
    if d1[1]!=d2[1]:
        matrix_1=torch.transpose(matrix_1)
    
    inner_matrix = torch.matmul(matrix_1, torch.transpose(matrix_2,0,1))
    
    norm_diff = -2 * inner_matrix + norm_square_1 + torch.transpose(norm_square_2,0,1)
#    print(norm_diff.shape)
    
    return norm_diff

def norm_matrix(x,y):
   
    diff = (x[:, None, :] - y[None, :, :])
    #print(diff)
    norm = torch.linalg.norm(diff, dim=-1)
   
    return norm

# Returns the pairwise inner product
def inner_matrix(matrix_1, matrix_2):
    d1=matrix_1.shape
    d2=matrix_2.shape
    # print(d1)
    # print(d2)
    if d1[1]!=d2[1]:
        matrix_1=torch.transpose(matrix_1,0,1)
    return torch.matmul(matrix_1, torch.transpose(matrix_2,0,1))

def minu_matrix(matrix_1, matrix_2, p):
    d1=matrix_1.shape
    d2=matrix_2.shape
    # print(d1)
    # print(d2)
    if d1[1]!=d2[1]:
        matrix_1=torch.transpose(matrix_1,0,1)
    return matrix_1**p - torch.transpose(matrix_2,0,1)**p

def kernel_anl3(matrix_1, matrix_2, parameters, active_func=None):

    i_elements = 0
    i_params = 15

    sum_elements = torch.sum(parameters[:i_params]**2)
    
    matrix_sqrt = norm_matrix(matrix_1, matrix_2)
    matrix = matrix_sqrt**2

    sigma = parameters[i_params+0]
    K = torch.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i_elements])**2/sum_elements
    i_params=i_params+1
    i_elements=i_elements+1
    
    c = (parameters[i_params])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i_elements])**2/sum_elements *(imatrix+c) ** 2
    i_params=i_params+1
    i_elements=i_elements+1
    
    beta = parameters[i_params]
    gamma = (parameters[i_params+1])**2
    K=K+ (parameters[i_elements])**2/sum_elements *(beta**2 + gamma*matrix)**(-1/2)
    i_params=i_params+2
    i_elements=i_elements+1
    
    alpha = parameters[i_params]
    beta = parameters[i_params+1]
    K=K+ (parameters[i_elements])**2/sum_elements *(beta**2 + matrix)**(-alpha)
    i_params=i_params+2
    i_elements=i_elements+1
    
    sigma_0 = parameters[i_params]
    K=K+ (parameters[i_elements])**2/sum_elements * 1/(1 + matrix/sigma_0**2)
    i_params=i_params+1
    i_elements=i_elements+1

    sigma_1 = parameters[i_params]
    K =  K+ (parameters[i_elements])**2/sum_elements *torch.maximum(torch.zeros(1,device = parameters.device), 1-matrix/(sigma_1))
    i_params=i_params+1
    i_elements=i_elements+1
    
    p = parameters[i_params]
    l = parameters[i_params+1]
    sigma_3 = parameters[i_params+2]
    K =K+ (parameters[i_elements])**2/sum_elements * torch.exp(-torch.sin(matrix*np.pi/p)**2/l**2)*torch.exp(-matrix/sigma_3**2)
    i_params=i_params+3
    i_elements=i_elements+1
    
    p = parameters[i_params]
    l = parameters[i_params+1]
    K = K+ (parameters[i_elements])**2/sum_elements *torch.exp(-torch.sin(matrix*np.pi/p)/l**2)
    i_params=i_params+2
    i_elements=i_elements+1

    # matrix_sqrt
    sigma_2 = parameters[i_params]
    K=K + torch.exp(-(matrix_sqrt)/ (2* sigma_2**2))*(parameters[i_elements])**2/sum_elements
    i_params=i_params+1
    i_elements=i_elements+1

    beta = parameters[i_params]
    gamma = (parameters[i_params+1])**2
    K=K+ (parameters[i_elements])**2/sum_elements *(beta**2 + gamma*matrix_sqrt)**(-1/2)
    i_params=i_params+2
    i_elements=i_elements+1
    
    alpha = parameters[i_params]
    beta = parameters[i_params+1]
    K=K+ (parameters[i_elements])**2/sum_elements *(beta**2 + matrix_sqrt)**(-alpha)
    i_params=i_params+2
    i_elements=i_elements+1
    
    sigma_0 = parameters[i_params]
    K=K+ (parameters[i_elements])**2/sum_elements * 1/(1 + matrix_sqrt/sigma_0**2)
    i_params=i_params+1
    i_elements=i_elements+1

    sigma_1 = parameters[i_params]
    K =  K+ (parameters[i_elements])**2/sum_elements *torch.maximum(torch.zeros(1,device = parameters.device), 1-matrix_sqrt/(sigma_1))
    i_params=i_params+1
    i_elements=i_elements+1
    
    p = parameters[i_params]
    l = parameters[i_params+1]
    sigma_3 = parameters[i_params+2]
    K =K+ (parameters[i_elements])**2/sum_elements * torch.exp(-torch.sin(matrix_sqrt*np.pi/p)**2/l**2)*torch.exp(-matrix_sqrt/sigma_3**2)
    i_params=i_params+3
    i_elements=i_elements+1
    
    p = parameters[i_params]
    l = parameters[i_params+1]
    K = K+ (parameters[i_elements])**2/sum_elements *torch.exp(-torch.sin(matrix_sqrt*np.pi/p)/l**2)
    i_params=i_params+2
    i_elements=i_elements+1
    
    return K

def kernel_gaussian(matrix_1, matrix_2, parameters, active_func=None):
    num_element = 1
    num_param = 1
    i_elements = 0
    i_params = num_element

    nn_active_func = nn.ReLU() if active_func is None else active_func
    max_elements = 1 # nn_active_func(parameters[:i_params].detach()).max()

    matrix_sqrt = norm_matrix(matrix_1, matrix_2)
    matrix_quad = matrix_sqrt**2

    ## Gaussian Kernel, (3, 1, 5)
    guass_elem = parameters[i_elements]
    guass_sigma = parameters[i_params]
    K = (nn_active_func(guass_elem))/max_elements * torch.exp(-matrix_quad/(2*guass_sigma**2 + 1e-40))
    i_elements=i_elements+1
    i_params=i_params+1

    return K


def kernel_river(matrix_1, matrix_2, parameters, active_func=None):

    num_element = 21
    num_param = 32
    i_elements = 0
    i_params = num_element

    nn_active_func = F.reLU if active_func is None else active_func

    max_elements = 1 # nn_active_func(parameters[:i_params].detach()).max()
    
    matrix_sqrt = norm_matrix(matrix_1, matrix_2)
    matrix_quad = matrix_sqrt**2
    inner_prod = inner_matrix(matrix_1, matrix_2)

    ## Linear Kernel, (1, 1, 1)
    linear_elem = parameters[i_elements]
    linear_c = parameters[i_params]
    ######      
    linear_elem = (1-torch.exp(linear_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else linear_elem
    K = (nn_active_func(linear_elem))/max_elements * (inner_prod + linear_c)
    i_elements = i_elements+1
    i_params = i_params+1

    ## Polynomial Kernel, (2, 3, 4)
    poly_elem = parameters[i_elements]
    poly_alpha = parameters[i_params]**2
    poly_c = parameters[i_params+1]**2
    poly_d = int(torch.abs(parameters[i_params+2]))
    ######      
    poly_elem = (1-torch.exp(poly_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else poly_elem
    K = K + (nn_active_func(poly_elem))/max_elements * (poly_alpha*inner_prod+poly_c)**poly_d
    i_elements = i_elements+1
    i_params = i_params+3

    ## Gaussian Kernel, (3, 1, 5)
    guass_elem = parameters[i_elements]
    guass_sigma = parameters[i_params]**2
    ######      
    guass_elem = (1-torch.exp(guass_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else guass_elem
    K = K + (nn_active_func(guass_elem))/max_elements * torch.exp(-matrix_quad/(2*guass_sigma + 1e-40))
    i_elements=i_elements+1
    i_params=i_params+1

    ## Laplacian Kernel, (4, 1, 6)
    laplac_elem = parameters[i_elements]
    laplac_sigma = parameters[i_params]**2
    ######      
    laplac_elem = (1-torch.exp(laplac_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else laplac_elem
    K = K + (nn_active_func(laplac_elem))/max_elements * torch.exp(-matrix_sqrt/(2*laplac_sigma + 1e-40))
    i_elements=i_elements+1
    i_params=i_params+1

    ## (5,3,9)
    anl3_8_elem = parameters[i_elements]
    anl3_8_gamma_1 = parameters[i_params]
    anl3_8_gamma_2 = parameters[i_params+1]**2
    anl3_8_gamma_3 = parameters[i_params+2]**2
    ######      
    anl3_8_elem = (1-torch.exp(anl3_8_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else anl3_8_elem
    K = K + (nn_active_func(anl3_8_elem))/max_elements * torch.exp(-torch.\
        sin(matrix_quad*np.pi/anl3_8_gamma_1)**2/(anl3_8_gamma_2 + 1e-40))*torch.exp(-matrix_quad/(anl3_8_gamma_3 + 1e-40))
    i_elements=i_elements+1
    i_params=i_params+3

    ## (6,2,11)
    anl3_9_elem = parameters[i_elements]
    anl3_9_gamma_1 = parameters[i_params]
    anl3_9_gamma_2 = parameters[i_params+1]**2 + 1e-40
    ######      
    anl3_9_elem = (1-torch.exp(anl3_9_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else anl3_9_elem
    K = K + (nn_active_func(anl3_9_elem))/max_elements * torch.exp(-torch.\
        sin(matrix_quad*np.pi/(anl3_9_gamma_1))/anl3_9_gamma_2)
    i_elements=i_elements+1
    i_params=i_params+2

    ## ,(7,3,14)
    coef_elem = parameters[i_elements]
    p = parameters[i_params]
    l = parameters[i_params+1]**2
    sigma_3 = parameters[i_params+2]**2 + 1e-40
    ######      
    coef_elem = (1-torch.exp(coef_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else coef_elem
    K =K+ (nn_active_func(coef_elem))/max_elements * torch.exp(-torch.\
        sin(matrix_sqrt*np.pi/p)**2/l)*torch.exp(-matrix_sqrt/sigma_3)
    i_elements=i_elements+1
    i_params=i_params+3
    
    ## ,(8,2,16)
    coef_elem = parameters[i_elements]
    p = parameters[i_params]
    l = parameters[i_params+1]**2 + 1e-40
    ######      
    coef_elem = (1-torch.exp(coef_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else coef_elem
    K = K+ (nn_active_func(coef_elem))/max_elements *torch.exp(-torch.\
        sin(matrix_sqrt*np.pi/p)/l)
    i_elements=i_elements+1
    i_params=i_params+2

    ## Multiquadric Kernel, (9,1,17)
    multiqua_elem = parameters[i_elements]
    multiqua_c = parameters[i_params]**2 + 1e-40
    ######      
    multiqua_elem = (1-torch.exp(multiqua_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else multiqua_elem
    K = K + (nn_active_func(multiqua_elem))/max_elements * (matrix_quad/multiqua_c + 1)**(1/2)
    i_elements=i_elements+1
    i_params=i_params+1

    ## Inverse Multiquadric Kernel, (10,1,19)
    invmultiqua_elem = parameters[i_elements]
    invmultiqua_c = parameters[i_params]**2 + 1e-40
    # invalpha = parameters[i_params+1]**2
    ######      
    invmultiqua_elem = (1-torch.exp(invmultiqua_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else invmultiqua_elem
    K = K + (nn_active_func(invmultiqua_elem))/max_elements * (matrix_quad/invmultiqua_c + 1)**(-1/2)
    i_elements=i_elements+1
    i_params=i_params+1

    ## ,(11,1,21)
    coef_elem = parameters[i_elements]
    beta = parameters[i_params]**2 + 1e-40
    # gamma = (parameters[i_params+1])**2
    ######      
    coef_elem = (1-torch.exp(coef_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else coef_elem
    K=K+ (nn_active_func(coef_elem))/max_elements * (1 + matrix_sqrt/beta)**(-1/2)
    i_elements=i_elements+1
    i_params=i_params+1
    # print('3', torch.any(torch.isnan(K)))
    
    ## (12, 2, 23)
    anl3_4_elem = parameters[i_elements]
    anl3_4_gamma_1 = parameters[i_params]**2 + 1e-40
    anl3_4_gamma_2 = int(torch.abs(parameters[i_params+1]))
    ######      
    anl3_4_elem = (1-torch.exp(anl3_4_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else anl3_4_elem
    K = K+ (nn_active_func(anl3_4_elem))/max_elements * (1 + matrix_quad/anl3_4_gamma_1)**(-anl3_4_gamma_2)
    i_elements=i_elements+1
    i_params=i_params+2

    ## ,(13,2,25)
    coef_elem = parameters[i_elements]
    alpha = int(torch.abs(parameters[i_params]))
    beta = parameters[i_params+1]**2 + 1e-40
    ######      
    coef_elem = (1-torch.exp(coef_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else coef_elem
    K=K+ (nn_active_func(parameters[i_elements]))/max_elements *(1 + matrix_sqrt/beta)**(-alpha)
    i_elements=i_elements+1
    i_params=i_params+2
  
    ## Cauchy Kernel, (14,1,26)
    cauchy_elem = parameters[i_elements]
    cauchy_sigma = parameters[i_params]**2 + 1e-40
    ######      
    cauchy_elem = (1-torch.exp(cauchy_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else cauchy_elem
    K = K + (nn_active_func(cauchy_elem))/max_elements * 1/(1 + matrix_quad/cauchy_sigma)
    i_elements=i_elements+1
    i_params=i_params+1

    ## ,(15,1,27)
    coef_elem = parameters[i_elements]
    sigma_0 = parameters[i_params]**2 + 1e-40
    ######      
    coef_elem = (1-torch.exp(coef_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else coef_elem
    K=K+ (nn_active_func(coef_elem))/max_elements * 1/(1 + matrix_sqrt/sigma_0)
    i_elements=i_elements+1
    i_params=i_params+1
    # print('4', torch.any(torch.isnan(K)))

    ## Rational Quadratic Kernel, (16,1,28)
    ratqua_elem = parameters[i_elements]
    ratqua_c = parameters[i_params]**2
    ######      
    ratqua_elem = (1-torch.exp(ratqua_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else ratqua_elem
    K = K + (nn_active_func(ratqua_elem))/max_elements * (1-matrix_quad/(matrix_quad+ratqua_c))
    i_elements=i_elements+1
    i_params=i_params+1

    ## (17,1,29)
    anl3_6_elem = parameters[i_elements]
    anl3_6_gamma_1 = parameters[i_params]**2 + 1e-40
    ######      
    anl3_6_elem = (1-torch.exp(anl3_6_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else anl3_6_elem
    K =  K + (nn_active_func(anl3_6_elem))/max_elements * \
        torch.maximum(torch.zeros(1,device = parameters.device), 1-matrix_quad/anl3_6_gamma_1)
    i_elements=i_elements+1
    i_params=i_params+1

    ## ,(18,1,30)
    coef_elem = parameters[i_elements]
    sigma_1 = parameters[i_params]**2 + 1e-40
    ######      
    coef_elem = (1-torch.exp(coef_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else coef_elem
    K =  K+ (nn_active_func(coef_elem))/max_elements * \
        torch.maximum(torch.zeros(1,device = parameters.device), 1-matrix_sqrt/sigma_1)
    i_elements=i_elements+1
    i_params=i_params+1
    
    ## Log Kernel, (19,1,31)
    log_elem = parameters[i_elements]
    log_d = int(torch.abs(parameters[i_params]))
    ######      
    log_elem = (1-torch.exp(log_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else log_elem
    K = K + (nn_active_func(log_elem))/max_elements * (torch.log(matrix_sqrt**log_d+1))
    i_elements=i_elements+1
    i_params=i_params+1

    ## Hyperbolic Tangent (Sigmoid) Kernel, (20,2,33)
    hytan_elem = parameters[i_elements]
    hytan_alpha = parameters[i_params]
    hytan_c = parameters[i_params+1]
    ######      
    hytan_elem = (1-torch.exp(hytan_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else hytan_elem
    K = K + (nn_active_func(hytan_elem))/max_elements * torch.tanh(hytan_alpha*inner_prod + hytan_c)
    i_elements=i_elements+1
    i_params=i_params+2

    ## Circular Kernel, (21,1,34)
    circ_elem = parameters[i_elements]
    circ_sigma = parameters[i_params]**2 + 1e-40
    candidate_K = (torch.arccos(-matrix_sqrt/circ_sigma) - \
        matrix_sqrt/circ_sigma*torch.sqrt(1-(matrix_sqrt/(circ_sigma))**2)) # matrix_quad
    circ_K = torch.where(matrix_sqrt<(circ_sigma+1e-30), candidate_K, 0)
    ######      
    circ_elem = (1-torch.exp(circ_elem)**(-1)) if isinstance(nn_active_func, nn.ReLU) else circ_elem
    K = K + (nn_active_func(circ_elem))/max_elements * circ_K
    i_params=i_params+1
    i_elements=i_elements+1

 

    # ## Spherical Kernel, (22,1,35)
    # spher_elem = parameters[i_elements]
    # spher_sigma = parameters[i_params]
    # candidate_K = 1 - 3/2*(matrix_sqrt/spher_sigma) + 1/2*(matrix_sqrt/spher_sigma)**3
    # spher_K = torch.where(matrix_sqrt<spher_sigma, candidate_K, 0)
    # K = K + spher_elem**2/sum_elements * spher_K
    # i_elements=i_elements+1
    # i_params=i_params+1

    # ## Power Kernel, (11,1,14)
    # power_elem = parameters[i_elements]
    # power_d = int(torch.abs(parameters[i_params]))
    # K = K + power_elem**2/sum_elements * (-matrix_sqrt**power_d)
    # i_elements=i_elements+1
    # i_params=i_params+1

    # ## (18, 2,21)
    # anl3_3_elem = parameters[i_elements]
    # anl3_3_gamma_1 = parameters[i_params]
    # anl3_3_gamma_2 = parameters[i_params+1]
    # K = K+ anl3_3_elem**2/sum_elements * (anl3_3_gamma_1**2 + anl3_3_gamma_2**2*matrix_quad)**(-1/2)
    # i_elements=i_elements+1
    # i_params=i_params+2

    # ## (20,1,24)
    # anl3_5_elem = parameters[i_elements]
    # anl3_5_gamma_1 = parameters[i_params]
    # K = K + anl3_5_elem**2/sum_elements * 1/(1 + matrix_quad/(anl3_5_gamma_1**2))
    # i_elements=i_elements+1
    # i_params=i_params+1

    # ## Chi-Square Kernel, (14,0,16)
    # chisqua_elem = parameters[i_elements]
    # matrix_2_chi = torch.unsqueeze(matrix_2,-1)
    # K = K + chisqua_elem**2/sum_elements * ((1/2-torch.sum((matrix_1.T-matrix_2_chi)**2/(matrix_1.T+matrix_2_chi), dim=-2)).T)
    # i_elements=i_elements+1

    # ## Histogram Intersection Kernel, (15,0,16)
    # histog_elem = parameters[i_elements]
    # K = K + histog_elem**2/sum_elements * torch.sum(torch.minimum(matrix_1.T,matrix_2_chi),dim=-2).T
    # i_elements=i_elements+1

    # ## Generalized Histogram Intersection, (16,2,18)
    # generhistog_elem = parameters[i_elements]
    # generhistog_alpha = parameters[i_params]
    # generhistog_beta = parameters[i_params+1]
    # K = K + generhistog_elem**2/sum_elements * torch.sum(torch.minimum(torch.abs(matrix_1.T)**generhistog_alpha, torch.abs(matrix_2_chi)**generhistog_beta), dim=-2).T
    # i_elements=i_elements+1
    # i_params=i_params+2

    # ## Generalized T-Student Kernel, (17,1,19)
    # genetstu_elem = parameters[i_elements]
    # genetstu_d = int(torch.abs(parameters[i_params]))
    # K = K + genetstu_elem**2/sum_elements * 1/(1+matrix_quad**genetstu_d)
    # i_elements=i_elements+1
    # i_params=i_params+1

    # ## Wave Kernel, (11,1,14)
    # wave_elem = parameters[i_elements]
    # wave_sigma = int(torch.abs(parameters[i_params]))
    # wava_i = matrix_sqrt/(wave_sigma+0.00001)
    # K = K + wave_elem**2/sum_elements * 1/wava_i*torch.abs(torch.sin(wava_i))
    # i_elements=i_elements+1
    # i_params=i_params+1

    ## ANOVA Kernel, (6,2)
    # anova_elem = parameters[i_elements]
    # anova_n = parameters[i_params]
    # anova_d = parameters[i_params+1]
    # anova_K = 0
    # for k in range(int(anova_n.cpu().clone().detach().numpy())**2):
    #     anova_i = minu_matrix(matrix_1, matrix_2, k)
    #     anova_K = anova_K + torch.exp(-(1/(1+torch.exp(-anova_i)))**2)**anova_d
    # K = K + anova_elem**2 * anova_K
    # i_params=i_params+1
    # i_elements=i_elements+2

    ## Bayesian Kernel, (24,x)
    ## Wavelet Kernel, (25,x)
    ## Spline Kernel, (16,x)
    ## B-Spline (Radial Basis Function) Kernel, (17,x)
    ## Bessel Kernel, (18,x)

    return K


def kernel_anl3_multi(matrix_1, matrix_2, parameters):
    K_res = None#torch.ones((parameters.shape[0],matrix_1.shape[0],matrix_2.shape[0]))
    for item in range(parameters.shape[0]):
        K_res = torch.unsqueeze(kernel_anl3(matrix_1,matrix_2,parameters[item,:], active_func=None), 0) if K_res==None else \
            torch.cat([K_res,torch.unsqueeze(kernel_anl3(matrix_1,matrix_2,parameters[item,:], active_func=None), 0)], dim=0)
    return K_res

def kernel_river_multi(matrix_1, matrix_2, parameters, active_func=None):
    K_res = None #torch.ones((parameters.shape[0],matrix_1.shape[0],matrix_2.shape[0]))
    for item in range(parameters.shape[0]):
        K_res = torch.unsqueeze(kernel_river(matrix_1,matrix_2,parameters[item,:],active_func), 0) if K_res==None else \
            torch.cat([K_res,torch.unsqueeze(kernel_river(matrix_1,matrix_2,parameters[item,:],active_func), 0)], dim=0)
    return K_res

def kernel_gaussian_multi(matrix_1, matrix_2, parameters, active_func=None):
    K_res = None #torch.ones((parameters.shape[0],matrix_1.shape[0],matrix_2.shape[0]))
    for item in range(parameters.shape[0]):
        K_res = torch.unsqueeze(kernel_gaussian(matrix_1,matrix_2,parameters[item,:], active_func=None), 0) if K_res==None else \
            torch.cat([K_res,torch.unsqueeze(kernel_gaussian(matrix_1,matrix_2,parameters[item,:], active_func=None), 0)], dim=0)
    return K_res



'''
def kernel_anl3(matrix_1, matrix_2, parameters):
    i=0
    
    matrix = norm_matrix(matrix_1, matrix_2)
    sigma = parameters[i+0]
    K =  torch.exp(-matrix/ (2* sigma**2))
    K=K*(parameters[i+1])**2
    i=i+2
    
    
    c = (parameters[i])**2
    imatrix = inner_matrix(matrix_1, matrix_2)
    K = K+ (parameters[i+1])**2 *(imatrix+c) ** 2
    i=i+2
    
    beta = parameters[i]
    gamma = (parameters[i+1])**2
    K=K+ (parameters[i+2])**2 *(beta**2 + gamma*matrix)**(-1/2)
    i=i+3
    
    alpha = parameters[i]
    beta = parameters[i+1]
    K=K+ (parameters[i+2])**2 *(beta**2 + matrix)**(-alpha)
    i=i+3
    
    sigma = parameters[i]
    K=K+ (parameters[i+1])**2 * 1/(1 + matrix/sigma**2)
    i=i+2
    
    alpha_0 = parameters[i]
    sigma_0 = parameters[i+1]
    alpha_1 = parameters[i+2]
    sigma_1 = parameters[i+3]
    K =  K+ (parameters[i+4])**2 *alpha_0*torch.maximum(torch.zeros(1,device = parameters.device), 1-matrix/(sigma_0))  + alpha_1 * torch.exp(-matrix/ (2* sigma_1**2))
    i=i+5
    
    p = parameters[i]
    l = parameters[i+1]
    sigma = parameters[i+2]
    K =K+ (parameters[i+3])**2 * torch.exp(-torch.sin(matrix*np.pi/p)**2/l**2)*torch.exp(-matrix/sigma**2)
    i=i+4
    
    p = parameters[i]
    l = parameters[i+1]
    K = K+ (parameters[i+2])**2 *torch.exp(-torch.sin(matrix*np.pi/p)/l**2)
    i=i+3
    
    return K
'''