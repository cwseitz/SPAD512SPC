import numpy as np
import matplotlib.pyplot as plt
import os
import secrets
import string
import json
import scipy.sparse as sp
import torch

from skimage.io import imsave
from scipy.special import erf
from scipy.optimize import minimize
from scipy.special import factorial
from scipy.stats import bernoulli,poisson

from ..utils import *
from SPAD512SPC.psf.psf2d.psf2d import *

class SPAD2D_Ring:
    """Simulates a small ROI of a 2D spad array 
    (SPAD photon counting camera)"""
    def __init__(self,config):
        self.config = config

    def ring(self,n,radius=3,phase=0):
        thetas = np.arange(0,n,1)*2*np.pi/n
        xs = radius*np.cos(thetas+phase)
        ys = radius*np.sin(thetas+phase)
        return xs,ys

    def generate(self,ring_radius=3,show=False):
        theta = np.zeros((3,self.config['particles']))
        nx,ny = self.config['nx'],self.config['ny']
        xsamp,ysamp = self.ring(self.config['particles'],radius=ring_radius)
        x0 = nx/2; y0 = ny/2
        theta[0,:] = ysamp + x0
        theta[1,:] = xsamp + y0
        theta[2,:] = self.config['sigma']
        x = self.get_counts(theta)
        if show:
            self.show(theta,x)
        return x
        
    def add_signal(self,prob,x_signal):
        """Distribute photons over space"""
        prob_flat = prob.flatten()
        rows,cols = prob.shape
        x_signal_ = np.zeros((rows, cols))
        x_signal = np.sum(x_signal).astype(np.uint8)
        for n in range(x_signal):
            idx = np.random.choice(rows*cols,p=prob_flat)
            row = idx // cols
            col = idx % cols
            x_signal_[row,col] += 1
        return x_signal_
        
    def add_background(self):
        size = (self.config['nx'],self.config['ny'])
        return poisson.rvs(self.config['lamb'],size=size)
        
    def get_prob(self,theta,patch_hw=3):
        probsum = np.zeros((self.config['nx'],self.config['ny']),dtype=np.float32)
        x = np.arange(0,2*patch_hw); y = np.arange(0,2*patch_hw)
        X,Y = np.meshgrid(x,y)
        for n in range(self.config['particles']):
            x0,y0,sigma = theta[:,n]
            patchx, patchy = int(round(x0))-patch_hw, int(round(y0))-patch_hw
            x0p = x0-patchx; y0p = y0-patchy
            lam = lamx(X,y0p,sigma)*lamy(Y,x0p,sigma)
            lam /= lam.sum()
            prob = np.zeros_like(probsum)
            prob[patchx:patchx+2*patch_hw,patchy:patchy+2*patch_hw] += lam
            probsum += prob   
        probsum = probsum/np.sum(probsum)
        return probsum      
                  
    def get_counts(self,theta):
        nt,nx,ny = self.config['nt'],self.config['nx'],self.config['ny']
        photons = np.zeros((nt,nx,ny),dtype=np.uint8)
        prob = self.get_prob(theta)
        for t in range(self.config['nt']):
            print(f'Generating frame {t}')
            x_background = self.add_background()
            x_signal =\
            bernoulli.rvs(self.config['zeta'],size=(self.config['particles'],))
            x_signal = self.add_signal(prob,x_signal)
            x_signal = x_signal.astype(np.uint8)
            x_background = x_background.astype(np.uint8)
            photons[t,:,:] += x_signal
            photons[t,:,:] += x_background
        return photons


    def show(self,theta,counts):
        fig,ax=plt.subplots(1,3,figsize=(12,3))
        csum = np.sum(counts,axis=0)
        ax[0].scatter(theta[1,:],theta[0,:],color='black',s=5)
        ax[0].set_aspect(1.0)
        ax[1].imshow(csum,cmap=plt.cm.BuGn_r)
        ax[2].plot(np.sum(counts,axis=(1,2)),color='black')
        ax[2].set_xlabel('Time'); ax[2].set_ylabel('Counts')
        ax[0].set_xlim([0,self.config['nx']])
        ax[0].set_ylim([0,self.config['ny']])
        ax[0].invert_yaxis()
        plt.tight_layout()



