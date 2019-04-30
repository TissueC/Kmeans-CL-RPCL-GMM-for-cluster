# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:45:02 2019

@author: Administrator
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import copy
init_k = 6
ratio=0.05
iteration_num=10
def competitive_k_means(save_plot=True):
    k=init_k
    plt.figure(figsize=(12, 12))
    X, y =generate_dataset()
    data_size=X.shape[0]
    plt.scatter(X[:,0],X[:,1],c=y,marker='+')
    if save_plot:
        plt.savefig("data.pdf")
    np.random.shuffle(X)
    center=X[np.random.choice(len(X),k)]
    for i in range(iteration_num):
        np.random.shuffle(X)
        y_pred,center,n_point_of_each_cluster=\
        distance(X,center,k)
        plt.figure()
        plt.scatter(X[:,0],X[:,1],c=y_pred,marker='+')
        plt.scatter(center[:,0],center[:,1],c='r')
        if save_plot:
            plt.savefig('iteration%d.pdf'%(i+1))
        print(n_point_of_each_cluster)
        chosen_index=np.where(n_point_of_each_cluster>data_size/3/k)[0]
        k=len(chosen_index)
        center=center[chosen_index]
    #plt.xlim([-3,3])0
    #plt.ylim([-3,3])    
    
    
    
    # k-mean
    print(center,k)
    y_pred=np.zeros(len(X))
    while True:
        for i in range(len(X)):
            tmp=np.zeros(k)
            for j in range(k):
                tmp[j]=np.sum((X[i]-center[j])**2)
            y_pred[i]=np.argmin(tmp)
        flag=True
        for i in range(k):
            record_x,record_y=center[i][0],center[i][1]
            center[i]=np.average(X[np.where(y_pred==i)[0]],axis=0)
            if (center[i][0]-record_x)/record_x>0.01 or \
            (center[i][1]-record_y)/record_y>0.01:
                flag=False
        print(center)
        plt.scatter(X[:,0],X[:,1],c=y_pred,marker='+')
        plt.scatter(center[:,0],center[:,1],c='r')
        if save_plot:
            plt.savefig('kmeans.pdf')
        if flag:
            break
    plt.show()

def distance(X,center,k):
    result=list()
    n_point_of_each_cluster=np.zeros(k)
    for i in range(len(X)):
        tmp=np.zeros(k)
        for j in range (k):
            tmp[j]=np.sum((X[i]-center[j])**2)
        tmp=n_point_of_each_cluster*tmp
        result.append(np.argmin(tmp))
        tmp[result[i]]=float('inf')
        center[result[i]]=center[result[i]]+(X[i]-center[result[i]])/(n_point_of_each_cluster[result[i]]+1)
        n_point_of_each_cluster[result[i]]+=1
        # push the competitive center
        rival=np.argmin(tmp)
        center[rival]=center[rival]-ratio*(X[i]-center[rival])/(n_point_of_each_cluster[rival]+1)
    return result,center,n_point_of_each_cluster



def generate_dataset():
    file='three_cluster.mat'
    data=loadmat(file)
    data_x=np.array(data['X'][0])[:,np.newaxis]
    data_y=np.array(data['X'][1])[:,np.newaxis]
    data_label=np.array(data['X'][2])
    X=np.concatenate((data_x,data_y),axis=1)
    return X,data_label

if __name__ == '__main__':
    competitive_k_means()