#!/usr/bin/env python3
# Marcos del Cueto
# Department of Chemistry and MIF, University of Liverpool
#################################################################################
import sys
import ast
import math
import functools
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib import ticker, gridspec, cm
from math import sqrt
from time import time
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from scipy.optimize import differential_evolution
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import KNeighborsRegressor, DistanceMetric
from sklearn.neighbors import NearestNeighbors
#import rdkit
#from rdkit import Chem, DataStructs
#from rdkit.Chem import rdMolDescriptors
#################################################################################
####### START CUSTOMIZABLE PARAMETERS #######
input_file_name = 'input_ML_d_a.txt'  # name of input file
# The rest of input options are inside the file 'input_file_name'
#######  END CUSTOMIZABLE PARAMETERS  #######
#################################################################################

#################################################################################
###### START MAIN ######
def main(alpha,gamma1,gamma2,gamma3,alpha_lim,gamma_lim1,gamma_lim2,gamma_lim3):
    # Read data
    df=pd.read_excel(input_file)
    # Preprocess data
    #df=preprocess_smiles(df) (not needed, we're reading directly FP)
    X=df[xcols].values
    y=df[ycols].values
    for i in range(Ndata):
        X_d=[]
        X_a=[]
        X1=X[i][elec_descrip][1:-1].split()
        X2=X[i][elec_descrip+1][1:-1].split()
        for j in range(FP_length):
            X_d.append(int(float(X1[j])))
            X_a.append(int(float(X2[j])))
        X[i][elec_descrip]=X_d
        X[i][elec_descrip+1]=X_a
    X=preprocess_fn(X)
    # Print some verbose info
    print('X:',flush=True)
    print(X,flush=True)
    print('Entries for each a/d pair:',len(X[0]))
    if print_log==True: f_out.write('X: \n')
    if print_log==True: f_out.write('%s \n' %(str(X)))
    if print_log==True: f_out.write('Entries for each a/d pair: %i \n' %(len(X[0])))
    if print_log==True: f_out.flush()
    # Get optimimum hyperparameters, or just use initial used
    if optimize_hyperparams==True:
        # Call kNN or KRR functions
        if ML=='kNN': 
            gammas = []
            if gamma1==0.0: # use just structural descriptors
                condition=1
                hyperparams=[gamma2,gamma3]
                bounds = [gamma_lim2] + [gamma_lim3]
                gammas =[gamma1]
                mini_args = (X, y, condition,gammas)
            elif gamma2==0.0 and gamma3==0.0: # use just electronic descriptors
                condition=2
                hyperparams=[gamma1]
                bounds = [ gamma_lim1 ]
                gammas =[gamma2,gamma3]
                mini_args = (X, y, condition,gammas)
            else: # use both electronic and structural descriptors
                condition=3
                hyperparams=[gamma1,gamma2,gamma3]
                bounds = [ gamma_lim1 ] + [gamma_lim2] + [gamma_lim3]
                gammas =[]
                mini_args = (X, y, condition,gammas)
            solver = differential_evolution(kNN,bounds,args=mini_args,popsize=15,tol=0.01,polish=False,workers=NCPU,updating='deferred')
        elif ML=='KRR':
            gammas = []
            if gamma1==0.0: # use just structural descriptors
                condition=1
                hyperparams=[alpha,gamma2,gamma3]
                bounds = [alpha_lim] + [gamma_lim2] + [gamma_lim3]
                gammas =[gamma1]
                mini_args = (X, y, condition,gammas)
            elif gamma2==0.0 and gamma3==0.0: # use just electronic descriptors
                condition=2
                hyperparams=[alpha,gamma1]
                bounds = [alpha_lim] + [ gamma_lim1 ]
                gammas =[gamma2,gamma3]
                mini_args = (X, y, condition,gammas)
            else: # use both electronic and structural descriptors
                condition=3
                hyperparams=[alpha,gamma1,gamma2,gamma3]
                bounds = [alpha_lim] + [ gamma_lim1 ] + [gamma_lim2] + [gamma_lim3]
                gammas = []
                mini_args = (X, y, condition,gammas)
            solver = differential_evolution(KRR,bounds,args=mini_args,popsize=15,tol=0.01,polish=False,workers=NCPU,updating='deferred')
        if ML=='SVR': 
            gammas = []
            if gamma1==0.0: # use just structural descriptors
                condition=1
                hyperparams=[gamma2,gamma3]
                bounds = [gamma_lim2] + [gamma_lim3]
                gammas =[gamma1]
                mini_args = (X, y, condition,gammas)
            elif gamma2==0.0 and gamma3==0.0: # use just electronic descriptors
                condition=2
                hyperparams=[gamma1]
                bounds = [ gamma_lim1 ]
                gammas =[gamma2,gamma3]
                mini_args = (X, y, condition,gammas)
            else: # use both electronic and structural descriptors
                condition=3
                hyperparams=[gamma1,gamma2,gamma3]
                bounds = [ gamma_lim1 ] + [gamma_lim2] + [gamma_lim3]
                gammas =[]
                mini_args = (X, y, condition,gammas)
            solver = differential_evolution(func_SVR,bounds,args=mini_args,popsize=15,tol=0.01,polish=False,workers=NCPU,updating='deferred')
        # Get best hyperparams
        best_hyperparams = solver.x
        best_rmse = solver.fun
        print('Best hyperparameters:', best_hyperparams,flush=True)
        print('Best rmse:', best_rmse,flush=True)
        if print_log==True: f_out.write('Best hyperparameters: %s \n' %(str(best_hyperparams)))
        if print_log==True: f_out.write('Best rmse: %s \n' %(str(best_rmse)))
        if print_log==True: f_out.flush()
        hyperparams=best_hyperparams.tolist()
    elif optimize_hyperparams==False:
        pass
        if ML=='kNN': 
            condition = 3 
            gammas = []
            hyperparams=[gamma1,gamma2,gamma3]
            kNN(hyperparams,X,y,condition,gammas)
        elif ML=='KRR': 
            condition = 3 
            gammas = []
            hyperparams=[alpha,gamma1,gamma2,gamma3]
            KRR(hyperparams,X,y,condition,gammas)
        elif ML=='SVR': 
            condition = 3 
            gammas = []
            hyperparams=[gamma1,gamma2,gamma3]
            func_SVR(hyperparams,X,y,condition,gammas)
###### END MAIN ######
#################################################################################

#################################################################################
###### START OTHER FUNCTIONS ######
### Function reading input parameters
def read_initial_values(inp):
    # open input file
    input_file_name = inp
    f_in = open('%s' %input_file_name,'r')
    f1 = f_in.readlines()
    # initialize arrays
    input_info = []
    var_name = []
    var_value = []
    # read info before comments. Ignore commented lines and blank lines
    for line in f1:
        if not line.startswith("#") and line.strip(): 
            input_info.append(line.split('#',1)[0].strip())
    # read names and values of variables
    for i in range(len(input_info)):
        var_name.append(input_info[i].split('=')[0].strip())
        var_value.append(input_info[i].split('=')[1].strip())
    # close input file
    f_in.close()
    # assign input variables    
    ML = ast.literal_eval(var_value[var_name.index('ML')])               # 'kNN' or 'KRR' or 'SVR'
    Neighbors = ast.literal_eval(var_value[var_name.index('Neighbors')]) # number of nearest-neighbors (only used for kNN)
    alpha  = ast.literal_eval(var_value[var_name.index('alpha')])       # kernel hyperparameter (only used for KRR)
    gamma1 = ast.literal_eval(var_value[var_name.index('gamma1')])     # hyperparameter with weight of d_el
    gamma2 = ast.literal_eval(var_value[var_name.index('gamma2')])     # hyperparameter with weight of d_fp_d
    gamma3 = ast.literal_eval(var_value[var_name.index('gamma3')])     # hyperparameter with weight of d_fp_a
    optimize_hyperparams = ast.literal_eval(var_value[var_name.index('optimize_hyperparams')])# whether hyperparameters are optimized (T) or just use initial values (F). If hyperparam=0.0, then that one is not optimized
    alpha_lim  = ast.literal_eval(var_value[var_name.index('alpha_lim')])     # range in which alpha hyperparam is optimized (only used for KRR)
    gamma_lim1 = ast.literal_eval(var_value[var_name.index('gamma_lim1')])    # range in which gamma1 is optimized
    gamma_lim2 = ast.literal_eval(var_value[var_name.index('gamma_lim2')])    # range in which gamma1 is optimized
    gamma_lim3 = ast.literal_eval(var_value[var_name.index('gamma_lim3')])    # range in which gamma1 is optimized
    input_file = ast.literal_eval(var_value[var_name.index('input_file')])    # name of input file with database
    elec_descrip = ast.literal_eval(var_value[var_name.index('elec_descrip')])# number of electronic descriptors: they must match the number in 'xcols', and be followed by the two structural descriptors
    xcols = ast.literal_eval(var_value[var_name.index('xcols')])              # specify which descriptors are used
    ycols = ast.literal_eval(var_value[var_name.index('ycols')])              # specify which is target property
    Ndata = ast.literal_eval(var_value[var_name.index('Ndata')])              # number of d/a pairs
    print_log = ast.literal_eval(var_value[var_name.index('print_log')])      # choose whether information is also written into a log file (Default: True)
    log_name = ast.literal_eval(var_value[var_name.index('log_name')])        # name of log file
    NCPU = ast.literal_eval(var_value[var_name.index('NCPU')])	              # select number of CPUs (-1 means all CPUs in a node)
    FP_length = ast.literal_eval(var_value[var_name.index('FP_length')])      # select number of CPUs (-1 means all CPUs in a node)
    weight_RMSE = ast.literal_eval(var_value[var_name.index('weight_RMSE')])  # select number of CPUs (-1 means all CPUs in a node)
    # open log file to write intermediate information
    if print_log==True:
        f_out = open('%s' %log_name,'w')
    else:
        f_out=None
    print('### START PRINT INPUT OPTIONS ###')
    print('ML = ', ML)
    print('Neighbors = ', Neighbors)
    print('alpha = ', alpha)
    print('gamma1 = ', gamma1)
    print('gamma2 = ', gamma2)
    print('gamma3 = ', gamma3)
    print('optimize_hyperparams = ', optimize_hyperparams)
    print('alpha_lim = ', alpha_lim)
    print('gamma_lim1 = ', gamma_lim1)
    print('gamma_lim2 = ', gamma_lim2)
    print('gamma_lim3 = ', gamma_lim3)
    print('input_file = ', input_file)
    print('elec_descrip = ', elec_descrip)
    print('xcols = ', xcols)
    print('ycols = ', ycols)
    print('Ndata = ', Ndata)
    print('print_log = ', print_log)
    print('log_name = ', log_name)
    print('NCPU = ', NCPU)
    #print('f_out = ', f_out)
    print('FP_length = ', FP_length)
    print('weight_RMSE = ', weight_RMSE)
    print('### END PRINT INPUT OPTIONS ###')
    if print_log==True: f_out.write('### START PRINT INPUT OPTIONS ###')
    if print_log==True: f_out.write('ML %s\n' % str(ML))
    if print_log==True: f_out.write('Neighbors %s\n' % str(Neighbors))
    if print_log==True: f_out.write('alpha %s\n' % str(alpha))
    if print_log==True: f_out.write('gamma1 %s\n' % str(gamma1))
    if print_log==True: f_out.write('gamma2 %s\n' % str(gamma2))
    if print_log==True: f_out.write('gamma3 %s\n' % str(gamma3))
    if print_log==True: f_out.write('optimize_hyperparams %s\n' % str(optimize_hyperparams))
    if print_log==True: f_out.write('alpha_lim %s\n' % str(alpha_lim))
    if print_log==True: f_out.write('gamma_lim1 %s\n' % str(gamma_lim1))
    if print_log==True: f_out.write('gamma_lim2 %s\n' % str(gamma_lim2))
    if print_log==True: f_out.write('gamma_lim3 %s\n' % str(gamma_lim3))
    if print_log==True: f_out.write('input_file %s\n' % str(input_file))
    if print_log==True: f_out.write('elec_descrip %s\n' % str(elec_descrip))
    if print_log==True: f_out.write('xcols %s\n' % str(xcols))
    if print_log==True: f_out.write('ycols %s\n' % str(ycols))
    if print_log==True: f_out.write('Ndata %s\n' % str(Ndata))
    if print_log==True: f_out.write('print_log %s\n' % str(print_log))
    if print_log==True: f_out.write('log_name %s\n' % str(log_name))
    if print_log==True: f_out.write('NCPU %s\n' % str(NCPU))
    #if print_log==True: f_out.write('f_out %s\n' % str(f_out))
    if print_log==True: f_out.write('FP_length %s\n' % str(FP_length))
    if print_log==True: f_out.write('weight_RMSE %s\n' % str(weight_RMSE))
    if print_log==True: f_out.write('### END PRINT INPUT OPTIONS ###')

    return (ML,Neighbors,alpha,gamma1,gamma2,gamma3,optimize_hyperparams,alpha_lim,gamma_lim1,gamma_lim2,gamma_lim3,input_file,elec_descrip,xcols,ycols,Ndata,print_log,log_name,NCPU,f_out,FP_length,weight_RMSE)

### Preprocess function to scale data ###
def preprocess_fn(X):
    '''
    Function to preprocess raw data for the KRR.

    Parameters
    ----------
    X: np.array.
        raw data array.

    Returns
    -------
    X: np.array.
        processed data array.
    '''
    X_el=[[] for j in range(Ndata)]
    X_fp_d=[]
    X_fp_a=[]
    for i in range(Ndata):
        for j in range(elec_descrip):
            X_el[i].append(X[i][j])
        X_fp_d.append(X[i][elec_descrip])
        X_fp_a.append(X[i][elec_descrip+1])
    xscaler = StandardScaler()
    X_el = xscaler.fit_transform(X_el)
    X = np.c_[ X_el,X_fp_d,X_fp_a]

    return X

### Function to calculate custom metric for electronic and structural properties ###
def custom_distance(X1,X2,gamma1,gamma2,gamma3):
    d_el=0.0
    d_fp=0.0
    # Calculate distance for electronic properties
    for i in range(0,elec_descrip):
        d_el = d_el + (X1[i]-X2[i])**2
    d_el = d_el**(1.0/2.0)
    # Calculate distances for FP
    ndesp1 = elec_descrip + FP_length
    ndesp2 = elec_descrip + FP_length + FP_length
    T_d = ( np.dot(np.transpose(X1[elec_descrip:ndesp1]),X2[elec_descrip:ndesp1]) ) / ( np.dot(np.transpose(X1[elec_descrip:ndesp1]),X1[elec_descrip:ndesp1]) + np.dot(np.transpose(X2[elec_descrip:ndesp1]),X2[elec_descrip:ndesp1]) - np.dot(np.transpose(X1[elec_descrip:ndesp1]),X2[elec_descrip:ndesp1]) )
    T_a = ( np.dot(np.transpose(X1[ndesp1:ndesp2]),X2[ndesp1:ndesp2]) ) / ( np.dot(np.transpose(X1[ndesp1:ndesp2]),X1[ndesp1:ndesp2]) + np.dot(np.transpose(X2[ndesp1:ndesp2]),X2[ndesp1:ndesp2]) - np.dot(np.transpose(X1[ndesp1:ndesp2]),X2[ndesp1:ndesp2]) )
    d_fp_d = 1 - T_d
    d_fp_a = 1 - T_a
    # Calculate final distance
    distance=gamma1*d_el+gamma2*d_fp_d+gamma3*d_fp_a
    return distance

### Function to calculate rmse and r with k-NN ###
def kNN(hyperparams,X,y,condition,gammas):
    # Assign hyperparameters
    if condition==1:
        gamma1 = gammas[0]
        gamma2, gamma3 = hyperparams
    elif condition==2:
        gamma1 = hyperparams
        gamma2 = gammas[0]
        gamma3 = gammas[1]
    elif condition==3:
        gamma1, gamma2, gamma3 = hyperparams
    # LOO
    y_predicted_knn=[]
    y_real_knn=[]
    loo=LeaveOneOut()
    counter=1
    # For each entry of LOO
    for train_index, test_index in loo.split(X, y):
        print('Step',counter," / ", Ndata,flush=True)
        counter=counter+1
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        knn = KNeighborsRegressor(n_neighbors=Neighbors, weights='distance', metric=custom_distance,metric_params={"gamma1":gamma1,"gamma2":gamma2,"gamma3":gamma3})
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        y_predicted_knn.append(y_pred_knn.tolist())
        y_real_knn.append(y_test.tolist())
    # Put results together in a list
    y_real_knn_list=[]
    y_predicted_knn_list=[]
    y_real_knn_list = [item for caca in y_real_knn for item in caca ]
    y_predicted_knn_list = [item for caca in y_predicted_knn for item in caca ]
    y_real_knn_list_list=[]
    y_predicted_knn_list_list=[]
    y_real_knn_list_list = [item for caca in y_real_knn_list for item in caca ]
    y_predicted_knn_list_list = [item for caca in y_predicted_knn_list for item in caca ]
    # Calculate rmse and r
    weights = np.ones_like(y_real_knn_list_list)
    r_knn, _ = pearsonr(y_real_knn_list_list, y_predicted_knn_list_list)
    rms_knn  = sqrt(mean_squared_error(y_real_knn_list_list, y_predicted_knn_list_list,sample_weight=weights))

    y_real_knn_list_list_array=np.array(y_real_knn_list_list)
    y_predicted_knn_list_list_array=np.array(y_predicted_knn_list_list)
    plot_scatter(y_real_knn_list_list_array, y_predicted_knn_list_list_array)
    # Print results
    print('New k-NN call:')
    print('gamma1:', gamma1, 'gamma2:', gamma2, 'gamma3:', gamma3, 'r k-NN:', r_knn.tolist(), 'rmse k-NN:',rms_knn,flush=True)
    if print_log==True: f_out.write('New k-NN call: \n')
    if print_log==True: f_out.write('gamma1: %f, gamma2: %f gamma3: %f, r k-NN: %f, rmse k-NN: %f \n' %(gamma1, gamma2, gamma3, r_knn.tolist(), rms_knn))
    if print_log==True: f_out.flush()
    return rms_knn

### Function to calculate rmse and r with k-NN (need to be adapted to new dataset) ###
def KRR(hyperparams,X,y,condition,gammas):
    # Assign hyperparameters
    if condition==1:
        gamma1 = gammas[0]
        alpha, gamma2, gamma3 = hyperparams
    elif condition==2:
        alpha, gamma1 = hyperparams
        gamma2 = gammas[0]
        gamma3 = gammas[1]
    elif condition==3:
        alpha, gamma1, gamma2, gamma3 = hyperparams
    # Build kernel function
    kernel = build_hybrid_kernel(gamma1=gamma1,gamma2=gamma2,gamma3=gamma3)
    # LOO
    y_predicted_krr=[]
    y_real_krr=[]
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    tr_errs = []
    counter=1
    # For each entry of LOO
    for train_index, test_index in loo.split(X, y):
        print('Step',counter," / ", Ndata,flush=True)
        counter=counter+1
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        krr = KernelRidge(alpha=alpha, kernel=kernel)
        krr.fit(X_train, y_train)
        y_pred_krr = krr.predict(X_test)
        y_predicted_krr.append(y_pred_krr.tolist())
        y_real_krr.append(y_test.tolist())
    # Put results together in a list
    y_real_krr_list=[]
    y_predicted_krr_list=[]
    y_real_krr_list = [item for caca in y_real_krr for item in caca ]
    y_predicted_krr_list = [item for caca in y_predicted_krr for item in caca ]
    y_real_krr_list_list=[]
    y_predicted_krr_list_list=[]
    y_real_krr_list_list = [item for caca in y_real_krr_list for item in caca ]
    y_predicted_krr_list_list = [item for caca in y_predicted_krr_list for item in caca ]
    # Calculate rmse and r
    weights = np.ones_like(y_real_krr_list_list)
    r_KRR, _ = pearsonr(y_real_krr_list_list, y_predicted_krr_list_list)
    rms_KRR  = sqrt(mean_squared_error(y_real_krr_list_list, y_predicted_krr_list_list,sample_weight=weights))

    y_real_KRR_list_list_array=np.array(y_real_KRR_list_list)
    y_predicted_KRR_list_list_array=np.array(y_predicted_KRR_list_list)
    plot_scatter(y_real_KRR_list_list_array, y_predicted_KRR_list_list_array)

    # Print results
    print('New KRR call:')
    print('gamma1:', gamma1, 'gamma2:', gamma2, 'gamma3:', gamma3, 'r KRR:', r_KRR, 'rmse KRR:',rms_KRR,flush=True)
    print('alpha:',krr.get_params(),flush=True)
    if print_log==True: f_out.write('New KRR call: \n')
    if print_log==True: f_out.write('gamma1: %f, gamma2: %f gamma3: %f, r KRR: %f, rmse KRR: %f \n' %(gamma1, gamma2, gamma3, r_KRR.tolist(), rms_KRR))
    if print_log==True: f_out.write('alpha: %s \n' %(str(krr.get_params())))
    if print_log==True: f_out.flush()
    return rms_KRR

### Function to calculate rmse and r with SVR ###
def func_SVR(hyperparams,X,y,condition,gammas):
    # Assign hyperparameters
    if condition==1:
        gamma1 = gammas[0]
        gamma2, gamma3 = hyperparams
    elif condition==2:
        gamma1 = hyperparams
        gamma2 = gammas[0]
        gamma3 = gammas[1]
    elif condition==3:
        gamma1, gamma2, gamma3 = hyperparams
    # Build kernel function
    svr = SVR(kernel=functools.partial(kernel_SVR, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3))
    # LOO
    y_predicted_svr=[]
    y_real_svr=[]
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    tr_errs = []
    counter=1
    # For each entry of LOO
    for train_index, test_index in loo.split(X, y):
        print('Step',counter," / ", Ndata,flush=True)
        X_train,X_test=X[train_index],X[test_index]
        y_train,y_test=y[train_index],y[test_index]
        svr.fit(X_train, y_train.ravel())
        y_pred_svr = svr.predict(X_test)
        y_predicted_svr.append(y_pred_svr.tolist())
        y_real_svr.append(y_test.tolist())
        counter=counter+1
    # Put results together in a list
    y_real_svr_list=[]
    y_predicted_svr_list=[]
    y_real_svr_list = [item for caca in y_real_svr for item in caca ]
    y_predicted_svr_list = y_predicted_svr
    y_real_svr_list_list=[]
    y_predicted_svr_list_list=[]
    y_real_svr_list_list = [item for caca in y_real_svr_list for item in caca ]
    y_predicted_svr_list_list = [item for caca in y_predicted_svr_list for item in caca ]
    # Calculate rmse and r
    weights = np.ones_like(y_real_svr_list_list)
    r_SVR, _ = pearsonr(y_real_svr_list_list, y_predicted_svr_list_list)
    rms_SVR  = sqrt(mean_squared_error(y_real_svr_list_list, y_predicted_svr_list_list,sample_weight=weights))
    y_real_SVR_list_list_array=np.array(y_real_SVR_list_list)
    y_predicted_SVR_list_list_array=np.array(y_predicted_SVR_list_list)
    plot_scatter(y_real_SVR_list_list_array, y_predicted_SVR_list_list_array)
   
    # Print results
    print('New SVR call:')
    print('gamma1:', gamma1, 'gamma2:', gamma2, 'gamma3:', gamma3, 'r SVR:', r_SVR, 'rmse SVR:',rms_SVR,flush=True)
    print('SVR parameters:',svr.get_params(),flush=True)
    if print_log==True: f_out.write('New SVR call: \n')
    if print_log==True: f_out.write('gamma1: %f, gamma2: %f gamma3: %f, r k-NN: %f, rmse k-NN: %f \n' %(gamma1, gamma2, gamma3, r_SVR.tolist(), rms_SVR))
    if print_log==True: f_out.write('alpha: %s \n' %(str(svr.get_params())))
    if print_log==True: f_out.flush()
    return rms_SVR

### SVR kernel function
def kernel_SVR(_x1, _x2, gamma1, gamma2, gamma3):
    # Initialize kernel values
    K_el   = 1.0
    K_fp_d = 1.0
    K_fp_a = 1.0
    size_matrix1=_x1.shape[0]
    size_matrix2=_x2.shape[0]
    ### K_el ###
    if gamma1 != 0.0:
        # define Xi_el
        Xi_el = [[] for j in range(size_matrix1)]
        for i in range(size_matrix1):
            for j in range(elec_descrip):
                Xi_el[i].append(_x1[i][j])
        Xi_el = np.array(Xi_el)
        # define Xj_el
        Xj_el = [[] for j in range(size_matrix2)]
        for i in range(size_matrix2):
            for j in range(elec_descrip):
                Xj_el[i].append(_x2[i][j])
        Xj_el = np.array(Xj_el)
        # calculate K_el
        D_el  = euclidean_distances(Xi_el, Xj_el)
        D2_el = np.square(D_el)
        K_el  = np.exp(-gamma1*D2_el)
    ### K_fp_d ###
    if gamma2 != 0.0:
        # define Xi_fp_d
        Xi_fp_d = [[] for j in range(size_matrix1)]
        for i in range(size_matrix1):
            for j in range(FP_length):
                Xi_fp_d[i].append(_x1[i][j+elec_descrip])
        Xi_fp_d = np.array(Xi_fp_d)
        # define Xj_fp_d
        Xj_fp_d = [[] for j in range(size_matrix2)]
        for i in range(size_matrix2):
            for j in range(FP_length):
                Xj_fp_d[i].append(_x2[i][j+elec_descrip])
        Xj_fp_d = np.array(Xj_fp_d)
        # calculate K_fp_d
        Xii_d = np.repeat(np.linalg.norm(Xi_fp_d, axis=1, keepdims=True)**2, size_matrix2, axis=1)
        Xjj_d = np.repeat(np.linalg.norm(Xj_fp_d, axis=1, keepdims=True).T**2, size_matrix1, axis=0)
        T_d = np.dot(Xi_fp_d, Xj_fp_d.T) / (Xii_d + Xjj_d - np.dot(Xi_fp_d, Xj_fp_d.T))
        D_fp_d  = 1 - T_d
        D2_fp_d = np.square(D_fp_d)
        K_fp_d = np.exp(-gamma2*D2_fp_d)
    ### K_fp_a ###
    if gamma3 != 0.0:
        # define Xi_fp_a
        Xi_fp_a = [[] for j in range(size_matrix1)]
        for i in range(size_matrix1):
            for j in range(FP_length):
                Xi_fp_a[i].append(_x1[i][j+elec_descrip])
        Xi_fp_a = np.array(Xi_fp_a)
        # define Xj_fp_a
        Xj_fp_a = [[] for j in range(size_matrix2)]
        for i in range(size_matrix2):
            for j in range(FP_length):
                Xj_fp_a[i].append(_x2[i][j+elec_descrip])
        Xj_fp_a = np.array(Xj_fp_a)
        # calculate K_fp_a
        Xii_a = np.repeat(np.linalg.norm(Xi_fp_a, axis=1, keepdims=True)**2, size_matrix2, axis=1)
        Xjj_a = np.repeat(np.linalg.norm(Xj_fp_a, axis=1, keepdims=True).T**2, size_matrix1, axis=0)
        T_d = np.dot(Xi_fp_a, Xj_fp_a.T) / (Xii_a + Xjj_a - np.dot(Xi_fp_a, Xj_fp_a.T))
        D_fp_a  = 1 - T_d
        D2_fp_a = np.square(D_fp_a)
        K_fp_a = np.exp(-gamma3*D2_fp_a)
    # Calculate final kernel
    K=K_el*K_fp_d*K_fp_a
    #K = np.exp(-gamma1*np.square(euclidean_distances(Xi_el, Xj_el)) - gamma2*np.square(1-(np.dot(Xi_fp_d, Xj_fp_d.T) / (Xii_d + Xjj_d - np.dot(Xi_fp_d, Xj_fp_d.T)))) - gamma3*np.square(1-(np.dot(Xi_fp_a, Xj_fp_a.T) / (Xii_a + Xjj_a - np.dot(Xi_fp_a, Xj_fp_a.T)))))
    #print('K just before return:', K)
    return K

### KRR Kernel function ###
def gaussian_kernel(Xi, Xj, gamma):
    '''
    Function to compute a gaussian kernel.

    Parameters
    ----------
    Xi: np.array.
        training data array
    Xj: np.array.
        training/testing data array.
    gamma: float.
        hyperparameter.

    Returns
    -------
    K: np.array.
        Kernel matrix.
    '''

    m1 = Xi.shape[0]
    m2 = Xi.shape[0]
    X1 = Xi[:,np.newaxis,:]
    X1 = np.repeat(X1, m2, axis=1)
    X2 = Xj[np.newaxis,:,:]
    X2 = np.repeat(X2, m1, axis=0)
    D2 = np.sum((X1 - X2)**2, axis=2)
    K = np.exp(-gamma * D2)

    return K

### KRR Kernel function ###
def tanimoto_kernel(Xi, Xj, gamma):
    '''
    Function to compute a Tanimoto kernel.

    Parameters
    ----------
    Xi: np.array.
        training data array
    Xj: np.array.
        training/testing data array.
    gamma: float.
        hyperparameter.

    Returns
    -------
    K: np.array.
        Kernel matrix.
    '''

    m1 = Xi.shape[0]
    m2 = Xj.shape[0]
    Xii = np.repeat(np.linalg.norm(Xi, axis=1, keepdims=True)**2, m2, axis=1)
    Xjj = np.repeat(np.linalg.norm(Xj, axis=1, keepdims=True).T**2, m1, axis=0)
    T = np.dot(Xi, Xj.T) / (Xii + Xjj - np.dot(Xi, Xj.T))
    K = np.exp(-gamma * (1 - T)**2)

    return K

### KRR Kernel function ###
def build_hybrid_kernel(gamma1,gamma2,gamma3):
    '''
    Parameters
    ----------
    gamma1: float.
        gaussian kernel hyperparameter.
    gamma2: float.
        Donor Tanimoto kernel hyperparameter.
    gamma3: float.
        Acceptor Tanimoto kernel hyperparameter.

    Returns
    -------
    hybrid_kernel: callable.
        function to compute the hybrid gaussian/Tanimoto kernel given values.
    '''

    def hybrid_kernel(_x1, _x2):
        '''
        Function to compute a hybrid gaussian/Tanimoto.

        Parameters
        ----------
        _x1: np.array.
            data point.
        _x2: np.array.
            data point.

        Returns
        -------
        K: np.array.
            Kernel matrix element.
        '''
        # Split electronic data from fingerprints
        ndesp1 = elec_descrip + FP_length
        Xi_el = _x1[:elec_descrip].reshape(1,-1)
        Xi_fp_d = _x1[elec_descrip:ndesp1].reshape(1,-1)
        Xi_fp_a = _x1[ndesp1:].reshape(1,-1)
        Xj_el = _x2[:elec_descrip].reshape(1,-1)
        Xj_fp_d = _x2[elec_descrip:ndesp1].reshape(1,-1)
        Xj_fp_a = _x2[ndesp1:].reshape(1,-1)
        # Compute kernels separately
        K_el = 1.0
        K_fp_d = 1.0
        K_fp_a = 1.0
        if gamma1 != 0.0: K_el = gaussian_kernel(Xi_el, Xj_el, gamma1)
        if gamma2 != 0.0: K_fp_d = tanimoto_kernel(Xi_fp_d, Xj_fp_d, gamma2)
        if gamma3 != 0.0: K_fp_a = tanimoto_kernel(Xi_fp_a, Xj_fp_a, gamma3)
        # Element-wise multiplication
        K = K_el * K_fp_d * K_fp_a
        return K

    return hybrid_kernel

### visualization and calculate pearsonr and spearmanr ###
def plot_scatter(x, y):

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)

    r, _ = pearsonr(x, y)
    rho, _ = spearmanr(x, y)

    ma = np.max([x.max(), y.max()]) + 1

    ax = plt.subplot(gs[0])
    ax.scatter(x, y, color="b")
    ax.tick_params(axis='both', which='major', direction='in', labelsize=22, pad=10, length=5)


    ax.set_xlabel(r"PCE / %", size=24, labelpad=10)
    ax.set_ylabel(r"PCE$^{k-NN}$ / %", size=24, labelpad=10)
    
   # ax.set_xlabel(r"Distance", size=24, labelpad=10)
   # ax.set_ylabel(r"RMSE$^{k-NN}$", size=24, labelpad=10)

    xtickmaj = ticker.MaxNLocator(5)
    xtickmin = ticker.AutoMinorLocator(5)
    ytickmaj = ticker.MaxNLocator(5)
    ytickmin = ticker.AutoMinorLocator(5)
    ax.xaxis.set_major_locator(xtickmaj)
    ax.xaxis.set_minor_locator(xtickmin)
    ax.yaxis.set_major_locator(ytickmaj)
    ax.yaxis.set_minor_locator(ytickmin)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', direction='in', labelsize=22, pad=10, length=2)
    ax.set_xlim(0, ma)
    ax.set_ylim(0, ma)
    ax.set_aspect('equal')

    # xmin, xmax = ax.get_xlim()
    # ymin, ymax = ax.get_ylim()
    ax.plot(np.arange(0, ma + 0.1, 0.1), np.arange(0, ma + 0.1, 0.1), color="k", ls="--")
    ax.annotate(u'$r$ = %.2f' % r, xy=(0.15,0.85), xycoords='axes fraction', size=22)
    # ax.annotate(u'$\\rho$ = %.2f' % rho, xy=(0.15,0.75), xycoords='axes fraction', size=22)
    plt.savefig('knn-basic-loo-PCE-results-model3.jpeg',dpi=600,bbox_inches='tight')

    return

##### END OTHER FUNCTIONS ######
################################################################################
################################################################################
################################################################################
### Run main program ###
start = time()
# Read input values
(ML,Neighbors,alpha,gamma1,gamma2,gamma3,optimize_hyperparams,alpha_lim,gamma_lim1,gamma_lim2,gamma_lim3,input_file,elec_descrip,xcols,ycols,Ndata,print_log,log_name,NCPU,f_out,FP_length,weight_RMSE) = read_initial_values(input_file_name)
# Execute main function
main(alpha,gamma1,gamma2,gamma3,alpha_lim,gamma_lim1,gamma_lim2,gamma_lim3)
# Print running time and close log file
time_taken = time()-start
print ('Process took %0.2f seconds' %time_taken,flush=True)
if print_log==True: f_out.write('Process took %0.2f seconds\n' %(time_taken))
if print_log==True: f_out.close()
