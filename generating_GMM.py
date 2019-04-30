import numpy as np
import matplotlib.pyplot as plt
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

# only show 2 dimension
plt.scatter(X[:,0],X[:,1],c=label_y,s=3)
