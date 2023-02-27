import os
import pickle

# build catalogue
def mkdir_func(path): # path
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)

skip_list = ['',]
point_num = 3300    
with open (f'./all_trajectories_{point_num}_new.pkl','rb') as f:
    all_trajectories = pickle.load(f)

for num,name in enumerate(all_trajectories.keys()):
    if name in skip_list:
        print(f"Passed: {num} {name}")
        mkdir_func(f'./lasso_cv_result/{num} {name} passed/')
        mkdir_func(f'./lasso_cv_result_0/{num} {name} passed/')
    else:
        mkdir_func(f'./lasso_cv_result/{num} {name}/')
        mkdir_func(f'./lasso_cv_result_0/{num} {name}/')

