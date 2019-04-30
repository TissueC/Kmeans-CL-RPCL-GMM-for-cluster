# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 00:32:21 2019

@author: Administrator
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture
import time

is_plot=False
is_verbose=False
test_iterations=10

n_component_list=[2,6,10]
n_samples_list=[1000,2000]
n_dimension_list=[15,40]


for n_component in n_component_list:
    for n_samples_sum in n_samples_list:
        for n_dimension in n_dimension_list:
            #n_component=np.random.randint(2,11)
            n_samples = np.random.randint(50,200,size=n_component)
            n_samples = (n_samples * (n_samples_sum/np.sum(n_samples))).astype(np.int16)
            #n_dimension=np.random.randint(2,51)
            random_mu=100*np.random.random((n_component,n_dimension))
            random_sigma=10*np.random.random((n_component,n_dimension))
            
            print('n_samples:',np.sum(n_samples))
            print('n_dimension:',n_dimension)
            print('n_component:',n_component)
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
            
            
            
            #aic to get k, then fit.
            aic=[]
            bic=[]
            component_range=range(2,11)
            for i in component_range:
                    # Fit a Gaussian mixture with EM
                    gmm_aicbic = mixture.GaussianMixture(n_components=i,
                                                  covariance_type='diag')
                    gmm_aicbic.fit(X)
                    aic.append(gmm_aicbic.aic(X))
                    bic.append(gmm_aicbic.bic(X))
            aic=np.array(aic)
            bic=np.array(bic)
            print('aic k: ',np.argmin(aic)+2)
            print('bic k: ',np.argmin(bic)+2)
            
            
            
                
                
            max_n_component=10
            gmm_VBEM = mixture.BayesianGaussianMixture(n_components=max_n_component,
                                          covariance_type='diag',max_iter=500)
            gmm_VBEM.fit(X)
            pred_y=gmm_VBEM.predict(X)
            print('VBEM k: ',len(set(pred_y)))
    
    
    
    
    
    #bic to get k, then fit.
    # bic = []
    # component_range=range(2,11)
    # for i in component_range:
    #         # Fit a Gaussian mixture with EM
    #         gmm_bic = mixture.GaussianMixture(n_components=i,
    #                                       covariance_type='diag')
    #         gmm_bic.fit(X)
    #         bic.append(gmm_bic.bic(X))    
    # bic=np.array(bic)
    
    
    # if is_plot:
    #     plt.figure()
    #     plt.plot(component_range,aic,c='r')
    #     plt.plot(component_range,bic,c='blue')
    #     plt.legend(['AIC','BIC'],fontsize=20)
    #     plt.show()
    # if is_verbose:
    #     print("number of dimension: ",n_dimension)
    #     print("number of component: ",n_component)
    #     print('min aic: ',np.argmin(aic)+2)
    #     print('min bic: ',np.argmin(bic)+2)
    # if np.argmin(aic)+2==n_component:
    #     aic_right+=1
    # if np.argmin(bic)+2==n_component:
    #     bic_right+=1