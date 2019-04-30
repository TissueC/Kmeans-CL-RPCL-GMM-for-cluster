# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:08:44 2019

@author: Administrator
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import time

is_plot=False
is_verbose=True
test_iterations=1

right=0
starttime=time.time()
for runnning_time in range(test_iterations):
    n_component=np.random.randint(2,11)
    n_samples = np.random.randint(50,200,size=n_component)
    n_dimension=np.random.randint(2,51)
    random_mu=100*np.random.random((n_component,n_dimension))
    random_sigma=10*np.random.random((n_component,n_dimension))
    
    X=np.zeros((0,n_dimension))
    label_y=np.zeros(0)
    for component_index in range(n_component):
        X=np.concatenate((
                X,random_sigma[component_index]*np.random.randn(
                        n_samples[component_index],
                        n_dimension)+random_mu[component_index]),axis=0)
        label_y=np.concatenate((
                label_y,component_index*np.ones(
                        n_samples[component_index])
                ))
    
    if is_plot:
        plt.scatter(X[:,0],X[:,1],c=label_y,s=3)
        plt.figure()
    
    #start to fit
    max_n_component=10
    gmm = mixture.BayesianGaussianMixture(n_components=max_n_component,
                                  covariance_type='diag',max_iter=500)
    gmm.fit(X)
    pred_y=gmm.predict(X)
    
    if is_verbose:
        print('label component number: ',n_component)
        print('pred component number: ',len(set(pred_y)))
    if is_plot:
        plt.scatter(X[:,0],X[:,1],c=pred_y,s=3)
        plt.show()
    if n_component==len(set(pred_y)):
        right+=1
    start_index=0
    pred_right=0
    for com in range(n_component):
        pred_right+=np.max(np.bincount(
                pred_y[start_index:start_index+n_samples[com]]))
        start_index+=n_samples[com]
    print('cluster accuracy: ',pred_right/np.sum(n_samples))
print('VBEM accuracy: %0.5f'%(right/test_iterations))
print('time cost: ',time.time()-starttime)