import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv
from numpy import genfromtxt
from scipy import linalg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import preprocessing
from outliers import smirnov_grubbs as grubbs
import csv
from iteration_utilities import deepflatten
import scipy.stats as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GiniPca(object):
    
    def __init__(self, gini_param):
        self.gini_param = gini_param
        assert self.gini_param >= 0.1 and self.gini_param != 1
            
    def ranks(self, x):
        n, k = x.shape
        r = np.zeros_like(x)
        for i in range(k):
            r[:,i] = (n + 1 - ss.rankdata(x[:,i], method='average'))**(self.gini_param-1)
        r -= np.mean(r, axis=0)
        return r
    
    def gmd(self, x):
        n, k = x.shape
        G = np.zeros_like(x)
        rank = self.ranks(x)
        xc = x - np.mean(x, axis=0)
        G = -2/(n*(n - 1)) * self.gini_param * (xc.T @ rank)
        return G
    
    def scale_gini(self, x):
        G = self.gmd(x)
        Z = (x - x.mean(axis=0)) / np.diag(G)[np.newaxis,:]
        return Z
    
    def project(self, x):
        Z = self.scale_gini(x)
        GMD = self.gmd(Z)
        _, vecp = linalg.eig(GMD.T + GMD)
        F = np.real(Z @ vecp)
        return F
    
    def project_l2(self, x):
        n, k = x.shape
        Z = preprocessing.scale(x)
        rho = (1/n) * (Z.T @ Z)
        _, vecp = linalg.eig(rho)
        F = np.real(Z @ vecp)
        return F
    
    def gini_corr(self, F, x):
        Z = self.scale_gini(x)
        r1 = self.ranks(Z)
        GC = (F.T @ r1) / np.diag(Z.T @ r1)[np.newaxis,:]
        return GC
    
    def act(self, x):
        n, k = x.shape
        Z = self.scale_gini(x)
        GMD = self.gmd(Z)
        valp, vecp = linalg.eig(GMD.T + GMD)
        F = np.real(Z @ vecp)
        rZ = self.ranks(Z)
        CTA = ((-2/(n*(n-1)))* self.gini_param * (F*(rZ @ vecp))) / (valp/2) 
        return np.real(CTA)
    
    def rct(self, x):
        CTR = np.zeros_like(x)
        F = self.project(x)
        CTR = np.abs(F) / np.sum(abs(F), axis = 0)
        return CTR
    
    def u_stat(self, x):
        n, k = x.shape
        F = self.project(x)
        GC = self.gini_corr(F, x)
        Z = self.scale_gini(x)
        axe1 = np.zeros_like(F)
        axe2 = np.zeros_like(F)
        for i in range(n):
            F1 = np.delete(F, i, axis=0) 
            Z1 = np.delete(Z, i, axis=0)
            r_Z1 = self.ranks(Z1)         
            Stock1 = (F1.T @ r_Z1) / np.diag(Z1.T @ r_Z1)[np.newaxis,:]
            axe1[i,:] = Stock1[0,:]
            axe2[i,:] = Stock1[1,:]
        std_jkf = np.zeros((2, k))
        std_jkf[0, :] = np.sqrt(np.var(axe1, axis =0, ddof=1) * ((n - 1)**2 / n))
        std_jkf[1, :] = np.sqrt(np.var(axe2, axis =0, ddof=1) * ((n - 1)**2 / n))
        ratio = GC[:2, :] / std_jkf
        return ratio
    
    def u_stat_pca(self, x):
        n, k = x.shape
        Z = preprocessing.scale(x)
        R = (1/n)* Z.T @ Z
        _, vecp = linalg.eig(R)
        F = np.real(Z @ vecp)
        F = preprocessing.scale(F)
        rho = (1/n)* F.T @ Z
        axe1 = np.zeros_like(F)
        axe2 = np.zeros_like(F)
        for i in range(n):
            F1 = np.delete(F, i, axis=0) 
            Z1 = np.delete(Z, i, axis=0)
            Stock1 = (1/(n-1))* (F1.T @ Z1)
            axe1[i,:] = Stock1[0,:]
            axe2[i,:] = Stock1[1,:]
        std_jkf = np.zeros((2, k))
        std_jkf[0, :] = np.sqrt(np.var(axe1, axis =0, ddof=1) * ((n - 1)**2 / n))
        std_jkf[1, :] = np.sqrt(np.var(axe2, axis =0, ddof=1) * ((n - 1)**2 / n))
        ratio = rho[:2, :] / std_jkf
        return ratio
        
    def optimal_gini_param(self,x):
        n, k = x.shape
        a=[]
        for i in range (k):
            a.append(grubbs.max_test_indices(x[:,i], alpha=0.05))
        x_outlier = np.delete(x, list(deepflatten(a)), axis=0) 
        eigen_val = []
        for self.gini_param in np.arange(1.1, 6, 0.1):
            Z = self.scale_gini(x_outlier)
            GMD = self.gmd(Z)
            valp_outlier,_ = linalg.eig(GMD.T + GMD)
            Z = self.scale_gini(x)
            GMD = self.gmd(Z)
            valp,_ = linalg.eig(GMD.T + GMD)
            eigen_val.append(np.abs(np.real(valp[:2].sum())/np.real(valp).sum() - valp_outlier[:2].sum()/np.real(valp_outlier).sum()))
        if (np.argmin(np.asarray(eigen_val))+1)/10 == 1:
            self.gini_param = (np.argmin(np.asarray(eigen_val))+1)/10 + 0.1
        else:
            self.gini_param = (np.argmin(np.asarray(eigen_val))+1)/10
        return self.gini_param

    def hotelling(self, x):
        n, k = x.shape
        Z = self.scale_gini(x)
        F = self.project(x)
        Hotelling1 = (n**2)*(n-1)/((n**2-1)*(n-1)) * (F[:,0])**2 / np.var(F[:,0])
        Hotelling2 = (n**2)*(n-2)/((n**2-1)*(n-1)) * ((F[:,0])**2 / np.var(F[:,0]) + (F[:,1])**2 / np.var(F[:,1]))
        return Hotelling1, Hotelling2

    def plot3D(self,x, y):
        n, k = x.shape
        Z = self.scale_gini(x)
        F = self.project(x)
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        X_reduced = F[:,:3]
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2],c = y, cmap=plt.cm.Set1, edgecolor='k', s=40)
        ax.set_title("Gini PCA")
        ax.set_xlabel("1st component")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd component")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd component")
        ax.w_zaxis.set_ticklabels([])
        return plt.show()
    

    


