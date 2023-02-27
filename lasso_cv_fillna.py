# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 00:12:10 2022

@author: Administrator
"""

import numpy as np
import pandas as pd
import torch
from src import kernel_flow_s, datagen
import tqdm
import matplotlib.pyplot as plt
import os, gc
# from dysts.base import make_trajectory_ensemble
from scipy.spatial.distance import directed_hausdorff
from sklearn.model_selection import KFold, train_test_split
import warnings
import pickle
import matplotlib
import argparse
import random
import seaborn as sns
sns.set_theme(style="whitegrid")

matplotlib.use('Agg')

warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.DoubleTensor)

def seed_torch(seed=42):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

exper_config = {
    "exp_delay" : [5, 3],
    "exp_noise_type":['Laplace_noise','Gaussian_noise'], 
    "exp_noise" : [0, 0.001, 0.01, 0.1], # , 
    "exp_lasso_gamma":[0.0002, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.05, 0.1, 0.3], # 
    "exp_kernel_type":['river',], 
 }

class CFG():
    delay = 5
    noise_ratio = 0.618
    lasso_gamma = 0
    noise_type = 'Laplace_noise'
    kernel_type = 'river'
    seed = 42
    regu_lambda = 0.05
    pred_len = 1
    learning_rate = 0.01
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    device = torch.device("cpu") 

# build catalogue
def mkdir_func(path): # specified folder path
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

def lasso_chaos(cv_data, config_dict):
    
    # Some constants
    nelements=21
    nparameters=32

    delay = config_dict["delay"]
    lasso_gamma = config_dict["lasso_gamma"]
    noisepercentage = config_dict["noise_ratio"]
    noise_type = config_dict["noise_type"]

    pred_len = config_dict["pred_len"]
    device = config_dict["device"]
    regu_lambda = config_dict["regu_lambda"]
    lr = config_dict["learning_rate"]
    kernel_set = config_dict["kernel_type"]
    dims = config_dict["dims"]
    name = config_dict["name"]
    k_fold = config_dict["k_fold"]
    num = config_dict["num"]

    method_name = 'Sparse KFs' if lasso_gamma>0 else 'Regular KFs'
    fold_name = 'lasso_cv_result' if lasso_gamma>0 else 'lasso_cv_result_0'

    X_train,Y_train,X_test,Y_test = cv_data
    gc.collect()
    
    if lasso_gamma <= 0:
        metric = "rho_ratio" 
    else:
        metric = "rho_ratio_l1" 

    model = kernel_flow_s.KernelFlows(kernel_set, nparameters=nparameters, nelements=nelements, regu_lambda=regu_lambda, dim=dims, \
        pred_len=pred_len, metric=metric, batch_size=100, lasso_gamma=lasso_gamma, device=device)
    model.to(device)
    model,rho_list = kernel_flow_s.train_kernel(X_train, Y_train, model, lr=lr) # , verbose=True
    rho_ser = [item.cpu().detach().numpy() for item in rho_list]
    rho_ser_np = np.array(rho_ser)
    rho_mean = rho_ser_np.mean()
    
    gc.collect()

    model.compute_kernel_and_inverse(regu_lambda=regu_lambda)
    gc.collect()

    data_res = []
    for i,item in enumerate(model.get_parameters()):
        if i == 0:
            columns = [f'elem_{x}' for x in range(item.shape[1])]
            data_e = pd.DataFrame(item.cpu().clone().detach().numpy(),columns=columns)
        else:
            columns = [f'param_{x}' for x in range(item.shape[1])]
            data_p = pd.DataFrame(item.cpu().clone().detach().numpy(),columns=columns)
            
    data_res = pd.concat([data_e,data_p], axis=1)

    horizon = 1
    Y_pred = model.predict_ahead(X_test, horizon=horizon, delay=delay, delta_t_mode=False, device=device)
    gc.collect()

    mse_pred = (Y_pred[0,:,:].cpu().detach()-Y_test[:,:dims]).pow(2).mean()
    r2 = 1-mse_pred/Y_test.var()
    smape_pred = ((2*torch.abs(Y_pred[0,:,:].cpu().detach()-Y_test[:,:dims]))/(torch.abs(Y_pred[0,:,:].cpu().detach())+torch.abs(torch.from_numpy(Y_test[:,:dims])))).mean()
    directed_hausdorff_value = directed_hausdorff(Y_test[:,:dims].T, Y_pred[0,:,:].cpu().detach().T)

    data_res[f'MSE_On_test_{horizon}'] = mse_pred.cpu().clone().detach().numpy()
    data_res[f'R2_{horizon}'] = r2.cpu().clone().detach().numpy()
    data_res[f'smape_{horizon}'] = smape_pred.cpu().clone().detach().numpy()
    data_res[f'directed_hausdorff_{horizon}'] = directed_hausdorff_value[0]
    data_res[f'directed_hausdorff_u_{horizon}'] = directed_hausdorff_value[1]
    data_res[f'directed_hausdorff_v_{horizon}'] = directed_hausdorff_value[2]
    
    print(f'done: {name}-{lasso_gamma}-{k_fold}')
    gc.collect()

    horizon = 2
    gc.collect()
    
    data_res[f'MSE_On_test_{horizon}'] = 11# mse_pred_2.cpu().clone().detach().numpy()
    data_res[f'R2_{horizon}'] = 11# r2_2.cpu().clone().detach().numpy()
    data_res[f'smape_{horizon}'] = 11# smape_pred_2.cpu().clone().detach().numpy()
    data_res[f'directed_hausdorff_{horizon}'] = 11# directed_hausdorff_value_2[0]
    data_res[f'directed_hausdorff_u_{horizon}'] = 11# directed_hausdorff_value_2[1]
    data_res[f'directed_hausdorff_v_{horizon}'] = 11# directed_hausdorff_value_2[2]
    
    data_res['Name'] = name
    data_res['lasso_gamma'] = lasso_gamma
    data_res['delay'] = delay
    data_res['noise'] = noise_type
    data_res['noise_percentage'] = noisepercentage
    data_res['k_fold'] = k_fold
    gc.collect()

    if os.path.exists(f'./{fold_name}/{num} {name}/chaosres_cv gamma-{lasso_gamma:.05f}.csv'):
        data_res.to_csv(f'./{fold_name}/{num} {name}/chaosres_cv gamma-{lasso_gamma:.05f}.csv', mode='a', header=False, index=False)
    else:
        data_res.to_csv(f'./{fold_name}/{num} {name}/chaosres_cv gamma-{lasso_gamma:.05f}.csv', index=False)

    watch_i = 0
    Y_pred_p = Y_pred[watch_i]
    
    plot_len = 300
    start_plot_point = 2000
    
    test_color = '#00A19D'
    pred_color = '#E05D5D'

    plt.figure(figsize=(18, 11.124))
    plt.suptitle(f'Predicted {name} system, {method_name}, delay:{delay}, gamma:{lasso_gamma}', fontsize=22) #  fold:{k_fold}
    plt.subplot(2,2,1)
    lw = 2
    
    plt.plot(Y_pred_p[0+start_plot_point:plot_len+start_plot_point,0].cpu().detach().numpy(),color=pred_color,linewidth=lw)
    plt.plot(Y_test[0+start_plot_point:plot_len+start_plot_point,0+watch_i*dims],color=test_color, linestyle='--',linewidth=lw-0.5)
    
    plt.legend(['prediction','true value'])
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('x', fontsize=18)

    plt.subplot(2,2,2)
    
    plt.plot(Y_pred_p[0+start_plot_point:plot_len+start_plot_point,1].cpu().detach().numpy(),color=pred_color,linewidth=lw)
    plt.plot(Y_test[0+start_plot_point:plot_len+start_plot_point,1+watch_i*dims],color=test_color, linestyle='--',linewidth=lw-0.5)
    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('y', fontsize=18)

    plt.subplot(2,2,3)
    
    plt.plot(Y_pred_p[0+start_plot_point:plot_len+start_plot_point,-1].cpu().detach().numpy(),color=pred_color,linewidth=lw)
    plt.plot(Y_test[0+start_plot_point:plot_len+start_plot_point,-1+watch_i*dims],color=test_color, linestyle='--',linewidth=lw-0.5)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel('z', fontsize=18)

    plt.subplot(2,2,4)
    
    
    y_34 = np.percentile(rho_ser,75)
    y_14 = np.percentile(rho_ser,25)
    y_top = y_34+10*(y_34-y_14)
    if lasso_gamma <= 0:
        y_bottom = -0.25
    else:
        y_top = 1 if np.isnan(y_top) else y_top
        y_bottom = max(-0.25, y_14-3.1*(y_34-y_14))
    try:
        plt.plot(rho_ser,marker='.',alpha=0.33, color='#283C63')
    except:
        pass
    plt.ylim(y_bottom, y_top)
    plt.ylabel('Rho', fontsize=18)
    plt.xlabel('train iter', fontsize=18)

    plt.savefig(f'./{fold_name}/{num} {name}/{name} gamma-{lasso_gamma:.05f} fold-{k_fold} traj.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10,10))
    pose1 = 1800
    pose2 = 3900
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Y_test[:pose1,0+watch_i*dims],Y_test[:pose1,1+watch_i*dims],Y_test[:pose1,-1+watch_i*dims],\
        color=test_color)
    ax.plot(Y_pred_p[pose1-1:pose2,0].cpu().detach(),Y_pred_p[pose1-1:pose2,1].cpu().detach(),Y_pred_p[pose1-1:pose2,-1].cpu().detach(),\
        color=pred_color)
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.set_zlabel('z', fontsize=18)
    ax.tick_params(axis='x',labelsize=16)
    ax.tick_params(axis='y',labelsize=16)
    ax.tick_params(axis='z',labelsize=16)
    ax.set_title(f'Predicted {name} system, {method_name}, delay:{delay}, gamma:{lasso_gamma}', fontsize=22) # fold:{k_fold} {noise_type}: {noisepercentage:.2%}, 
    ax.legend(['true value','prediction'])

    plt.savefig(f'./{fold_name}/{num} {name}/{name} gamma-{lasso_gamma:.05f} fold-{k_fold} phase.pdf', dpi=600, format='pdf', bbox_inches='tight')
    plt.close()
    gc.collect()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', help='number', required=True, type=int)
    args = parser.parse_args()
    end_number = args.number

    exp_kernel_type = 'river'
    exp_noise_type = 'Laplace_noise'
    exp_noise = 0
    exp_delay = 5

    skip_list = ['',]
    point_num = 7200 
    with open (f'./all_trajectories_{point_num}_new.pkl','rb') as f:
        all_trajectories = pickle.load(f)

    for num, name in enumerate(all_trajectories.keys()):
        # if not num in sele_list:
            # continue
        for exp_lasso_gamma in [0.0001, 0.001, 0.01, 0.1, 1, 10,]: 
            if (name in skip_list) | (num!=end_number): 
                continue
            else:
                Data = all_trajectories[name]

                observed_data= Data.copy()
                nlen,dims = Data.shape

                # generate noisy data
                noisepercentage = exp_noise
                np.random.seed(CFG.seed)

                if exp_noise_type == 'Laplace_noise':
                    noise = np.random.laplace(0, 1, (nlen, dims))
                elif exp_noise_type == 'Gaussian_noise':
                    noise = np.random.randn(nlen, dims)

                normalize = observed_data.std(axis=0)
                noise = noise/normalize
                observed_data = Data + noise * noisepercentage

                del Data
                gc.collect()

                normalize = observed_data.std(axis=0).T

                regu_lambda_now = 0.05

                kfold = KFold(n_splits=3, shuffle=True, random_state=CFG.seed)

                X_data, Y_data = datagen.prepare_data_fast_m(observed_data.T, normalize, delay=exp_delay, pred_len=1)
                fold_name = 'lasso_cv_result' if exp_lasso_gamma>0 else 'lasso_cv_result_0'

                for fold, (trn_ind, val_ind) in enumerate(kfold.split(X_data)):
                    last_value = None
                    k_fold = fold
                    X_train, Y_train = X_data[trn_ind,:],Y_data[trn_ind,:]
                    X_test, Y_test = X_data[val_ind,:],Y_data[val_ind,:]

                    condition_now = (not os.path.exists(f'./{fold_name}/{num} {name}/{name} gamma-{exp_lasso_gamma:.05f} fold-{k_fold} phase.pdf'))
                    if condition_now:
                        print(f'==================== {num} {name} ====================')
                        print(f'Process: {num} {name} gamma-{exp_lasso_gamma:.05f} fold-{k_fold}')
                        config_dict = {
                            "num":num,
                            "name":name,
                            "dims":dims,
                            "k_fold":k_fold,
                            "seed":CFG.seed,
                            "regu_lambda": regu_lambda_now,
                            "pred_len":1,
                            "learning_rate":0.012,
                            "device":torch.device("cpu"),
                            "kernel_type": exp_kernel_type,
                            "delay": exp_delay,
                            "noise_type": exp_noise_type,
                            "noise_ratio": exp_noise,
                            "lasso_gamma": exp_lasso_gamma,
                        }
                        cv_data = [X_train,Y_train,X_test,Y_test]
                        
                        try:
                            lasso_chaos(cv_data, config_dict)
                        except Exception as e:
                            print(e)
                    else:
                        continue


