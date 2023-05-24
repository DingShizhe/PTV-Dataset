#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import collections as mc

import matplotlib as mpl
import matplotlib.pyplot as plt

# change the following to %matplotlib notebook for interactive plotting
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def load_data(fn):
    return np.load(fn)


# data = load_data("field_data/0000.npy")

# print(data.shape)


# In[3]:


# def vis_data(data, data2=None):
def vis_data(*data):
    
    def vis_lines_using_ax(lines, ax):
    
        ns = np.linalg.norm(lines[:,0] - lines[:,1], axis=1)

        lines = [[ (l[0][0], l[0][1]), (l[1][0], l[1][1]) ] for l in lines]

        c = np.sign(ns-10) * np.power(ns-10, 2.0) / 10 + 10
        linewidths = 1 + c / 30

        lc = mc.LineCollection(lines, array=c, cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=20), linewidths=linewidths)


        ax.add_collection(lc)

        ax.set_xlim(0, 512)
        ax.set_ylim(0, 720)
        ax.set_aspect(1)

        ax.invert_yaxis()
    
    V_NUM = len(data)
    
    fig, axs = plt.subplots(1, V_NUM, figsize=(4*V_NUM,8), dpi=150)

    if V_NUM == 1:
        axs = [axs]

    for d,a in zip(data, axs):
        vis_lines_using_ax(d, a)

    plt.show()

# vis_data(data)


# In[4]:


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def denoise(data):
    
    X = data[:, 0]
    Y = data[:, 1] - X
    
    y1 = Y[:, 0:1]
    y2 = Y[:, 1:2]
    
#     print("X.shape", X.shape)
#     print("y1.shape", y1.shape)
    
    kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + 4.0 * WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    gp1 = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y1)
    gp2 = GaussianProcessRegressor(kernel=kernel, alpha=0.0).fit(X, y2)
    
    y1_denoised = gp1.predict(X, return_cov=False)
#     print("y1_denoised.shape", y1_denoised.shape)
    y2_denoised = gp2.predict(X, return_cov=False)
    
    y_denoised = np.concatenate([y1_denoised, y2_denoised], axis=1)
    
    y_denoised += X
    
    data_denoised = np.concatenate([X[:,np.newaxis,:], y_denoised[:,np.newaxis,:]], axis=1)
    
    return data_denoised

# data_denoised = denoise(data)


# In[5]:


# error = data_denoised[:, 1:2] - data[:, 1:2]

# import matplotlib.pyplot as plt

# plt.hist(error[:,0], bins=50, alpha=0.75)
# # plt.hist(error[:,0,0], bins=50, alpha=0.75)
# # plt.hist(error[:,0,1], bins=50, alpha=0.75)
# # print(error[:,0,0].shape)
# plt.show()

# print(error.max())
# print(error.min())
# # print((error==0).sum())

# error = np.concatenate([data[:, 0:1], error+data[:, 0:1]], axis=1)

# # vis_data(data, data_denoised)
# vis_data(data, data_denoised, error)


# In[ ]:


import os

data_files = sorted(os.listdir("field_data"))

def denoise_and_save(fn):
    print(fn)
    data = np.load("field_data/" + fn)
    data_denoised = denoise(data)
    np.save("field_data_denoised/" + fn, data_denoised)


# import multiprocessing

# def driver_func():
#     PROCESSES = 40
#     with multiprocessing.Pool(PROCESSES) as pool:
#         pool.map(denoise_and_save, data_files)

# driver_func()


# for fn in data_files[200:400]:
for fn in data_files[700:800]:
# for fn in data_files[400:800]:
# for fn in data_files[986:1100]:
# for fn in data_files[1100:1200]:
    denoise_and_save(fn)

