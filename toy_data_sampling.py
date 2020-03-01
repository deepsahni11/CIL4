import numpy as np
import sklearn
import random 
import pdb
from sklearn.metrics import *
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from imblearn.combine import *
from imblearn.ensemble import *
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from numpy import save
from numpy import load





seed = 0
samplers_all = [
    # Oversampling methods:
    RandomOverSampler(random_state=seed), 
    SMOTE(random_state=seed),             
    ADASYN(random_state=seed),            
    BorderlineSMOTE(random_state=seed),
    SVMSMOTE(random_state=seed),
    
    # Undersampling methods:
    RandomUnderSampler(random_state=seed),
    ClusterCentroids(random_state=seed),
    NearMiss(version=1, random_state=seed),
    NearMiss(version=2, random_state=seed),
    NearMiss(version=3, random_state=seed),
    TomekLinks(random_state=seed),
    EditedNearestNeighbours(random_state=seed),
    RepeatedEditedNearestNeighbours(random_state=seed),
    AllKNN(random_state=seed),
    CondensedNearestNeighbour(random_state=seed),
    OneSidedSelection(random_state=seed),
    NeighbourhoodCleaningRule(random_state=seed),
    InstanceHardnessThreshold(random_state=seed),
    
    
    # Combos:
    SMOTEENN(random_state=seed),
    SMOTETomek(random_state=seed)

]
samplers_array_all = np.array(samplers_all)


X_train_datasets_5d = [] 
y_train_datasets_5d = [] 
X_test_datasets_5d = []  
y_test_datasets_5d = []



X_train_datasets_5d_resampled = []
y_train_datasets_5d_resampled = []
X_test_datasets_5d_resampled = []
y_test_datasets_5d_resampled = []


with h5py.File("all_datasets.hdf5","r") as f:  
    dsList = f.attrs["names"].split()  
    for i in range(len(dsList)):   
        tr_x = ""                                                                                                                                                               
        tr_y = ""                                                                                                                                                               
        te_x = ""                                                                                                                                                               
        te_y = ""                                                                                                                                                               
        tr_x = tr_x + str(dsList[i]) + "/X"                                                                                                                                     
        tr_y = tr_y + str(dsList[i]) + "/y"                                                                                                                                     
        te_x = te_x + str(dsList[i]) + "/Xt"                                                                                                                                    
        te_y = te_y + str(dsList[i]) + "/yt"                                                                                                                                   
        #print(tr_x)                                                                                                                                                            
        Xtrain = np.array(f[ tr_x])                                                                                                                                            
        ytrain = np.squeeze(np.array(f[tr_y]))                                                                                                                                 
        Xtest = np.array(f[te_x])                                                                                                                                              
        ytest = np.squeeze(np.array(f[te_y]))                                                                                                                                  
        X_train_datasets_5d_resampled.append(Xtrain)
        y_train_datasets_5d_resampled.append(ytrain)
        X_test_datasets_5d_resampled.append(Xtest)
        y_test_datasets_5d_resampled.append(ytest)


        for i in range(len(samplers_array_all)):
            X_resampled, y_resampled = samplers_array_all[i].fit_sample(Xtrain, ytrain)
            X_train_datasets_5d_resampled.append(X_resampled)
            y_train_datasets_5d_resampled.append(y_resampled)
            X_test_datasets_5d_resampled.append(Xtest)
            y_test_datasets_5d_resampled.append(ytest)



save('../Data_metrics_toy_datasets_X_train.npy',X_train_datasets_5d_resampled)
save('../Data_metrics_toy_datasets_X_test.npy',X_test_datasets_5d_resampled)
save('../Data_metrics_toy_datasets_y_train.npy',y_train_datasets_5d_resampled)
save('../Data_metrics_toy_datasets_y_test.npy',y_test_datasets_5d_resampled)
