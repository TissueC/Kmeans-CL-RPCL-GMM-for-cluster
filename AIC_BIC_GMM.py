# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:42:23 2019

@author: Administrator
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import time
is_plot=False
test_iterations=10000
is_verbose=False


aic_right=0
bic_right=0
starttime=time.time()
for running_time in range(test_iterations):
    n_component=np.random.randint(2,11)
    n_samples = np.random.randint(40,200,size=n_component)
    n_dimension=np.random.randint(2,51)
    random_mu=100*np.random.random((n_component,n_dimension))
    random_sigma=10*np.random.random((n_component,n_dimension))
    
    
    X=np.zeros((0,n_dimension))
    label_y=np.zeros(0)
    for component_index in range(n_component):
        X=np.concatenate((
                X,random_sigma[component_index]*np.random.randn(
                        n_samples[component_index],n_dimension)+\
                        random_mu[component_index]),axis=0)
        label_y=np.concatenate((
                label_y,component_index*np.ones(
                        n_samples[component_index])
                ))
    
    if is_plot:
        plt.scatter(X[:,0],X[:,1],c=label_y,s=3)
    
    
    #start to fit
    aic = []
    bic = []
    component_range=range(2,11)
    for i in component_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=i,
                                          covariance_type='diag')
            gmm.fit(X)
            aic.append(gmm.aic(X))
            bic.append(gmm.bic(X))
    
    aic=np.array(aic)
    bic=np.array(bic)
    
    
    if is_plot:
        plt.figure()
        plt.plot(component_range,aic,c='r')
        plt.plot(component_range,bic,c='blue')
        plt.legend(['AIC','BIC'],fontsize=20)
        plt.show()
    if is_verbose:
        print("number of dimension: ",n_dimension)
        print("number of component: ",n_component)
        print('min aic: ',np.argmin(aic)+2)
        print('min bic: ',np.argmin(bic)+2)
    if np.argmin(aic)+2==n_component:
        aic_right+=1
    if np.argmin(bic)+2==n_component:
        bic_right+=1

print('aic k accurate:%0.5f'% (aic_right/test_iterations))
print('bic k accurate:%0.5f'% (bic_right/test_iterations))
print('time cost: ',time.time()-starttime)



