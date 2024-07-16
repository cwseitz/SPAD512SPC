import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from ..psf2d import *
from .mll2d import *
from .jac2d import *
from sklearn.mixture import BayesianGaussianMixture
import seaborn as sns
import pandas as pd
from multiprocessing import Pool

class MixMCMCParallel:
    def __init__(self, theta0, adu, sigma, N0):
        self.theta0 = theta0
        self.adu = adu
        self.sigma = sigma
        self.N0 = N0

    def log_prior(self, theta):
        if np.all(theta >= 2.0) and np.all(theta <= 8.0):
            return 0.0
        return -np.inf

    def log_likelihood(self, theta):
        try:
            log_like = -1 * mixloglike(theta, self.adu, sigma=self.sigma, N0=self.N0)
            if np.isnan(log_like) or np.isinf(log_like):
                return -np.inf
            return log_like
        except OverflowError:
            return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def run_mcmc(self, nwalkers=100, nsteps=1000, plot_fit=False):
        ndim = self.theta0.size
        pos = self.theta0 + 1e-3 * np.random.randn(nwalkers, ndim)

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability, pool=pool)
            sampler.run_mcmc(pos, nsteps, progress=True)

        samples = sampler.get_chain(discard=100, thin=5, flat=True)
        if plot_fit:
            self.plot_fit(samples)
        return samples

    def plot_fit(self, samples):
        fig = corner.corner(samples, labels=["param" + str(i) for i in range(samples.shape[1])])
        fig.show()
        plt.show()

    def find_modes_dpgmm(self, samples, max_components=10):
        dpgmm = BayesianGaussianMixture(
            n_components=max_components,
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1e-3,
            random_state=0
        )
        samples += np.random.normal(0, 0.01, size=samples.shape)
        dpgmm.fit(samples)
        modes = dpgmm.means_
        labels = dpgmm.predict(samples)

        print("Modes found by DPGMM:")
        print(modes)

        self.plot_dpgmm_fit(samples, dpgmm, labels)
        return modes

    def plot_dpgmm_fit(self, samples, dpgmm, labels):
        df = pd.DataFrame(samples, columns=[r'$x_0$', r'$y_0$'])
        df['cluster'] = labels
        sns.set_theme(font_scale=1.5, style='ticks')
        sns.pairplot(df, hue='cluster', diag_kind='kde', palette='tab10')
        plt.show()


class MixMCMC:
    def __init__(self,theta0,adu,sigma,N0):
        self.theta0 = theta0
        self.adu = adu
        self.sigma = sigma
        self.N0 = N0

    def log_prior(self, theta):
        if np.all(theta >= 2.0) and np.all(theta <= 8.0):
            return 0.0
        return -np.inf

    def log_likelihood(self,theta):
        try:
            log_like = -1*mixloglike(theta,self.adu,sigma=self.sigma,N0=self.N0)
            if np.isnan(log_like) or np.isinf(log_like):
                return -np.inf
            return log_like
        except OverflowError:
            return -np.inf

    def log_probability(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll
    
    def run_mcmc(self,nwalkers=100,nsteps=1000,plot_fit=False):
        ndim = self.theta0.size
        pos = self.theta0 + 1e-3*np.random.randn(nwalkers,ndim)
        sampler = emcee.EnsembleSampler(nwalkers,ndim,self.log_probability)
        sampler.run_mcmc(pos,nsteps,progress=True)
        samples = sampler.get_chain(discard=100,thin=5,flat=True)
        if plot_fit:
            self.plot_fit(samples)
        return samples
    
    def plot_fit(self,samples):
        fig = corner.corner(samples, labels=["param" + str(i) for i in range(samples.shape[1])])
        fig.show()
        plt.show()
        
    def find_modes_dpgmm(self, samples, max_components=10):
        dpgmm = BayesianGaussianMixture(n_components=max_components, covariance_type='full', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=1e-3, random_state=0)
        samples += np.random.normal(0,0.01,size=samples.shape)
        dpgmm.fit(samples)
        modes = dpgmm.means_
        labels = dpgmm.predict(samples)

        print("Modes found by DPGMM:")
        print(modes)

        self.plot_dpgmm_fit(samples, dpgmm, labels)
        return modes

    def plot_dpgmm_fit(self, samples, dpgmm, labels):
        df = pd.DataFrame(samples, columns=[r'$x_0$',r'$y_0$'])
        df['cluster'] = labels
        sns.set_theme(font_scale=1.5,style='ticks')
        sns.pairplot(df,hue='cluster',diag_kind='kde', palette='tab10')
        plt.show()

class MLE2DMix:
    def __init__(self,theta0,adu):
       self.theta0 = theta0
       self.adu = adu
                                         
    def optimize(self,max_iters=1000,lr=None,plot_fit=False,tol=1e-8,nparams=2):
        if plot_fit:
           thetat = []
        if lr is None:
           lr = 1e-5*np.ones_like(self.theta0)
        loglike = []
        theta = np.zeros_like(self.theta0)
        theta += self.theta0
        niters = 0
        converged = False
        while niters < max_iters:
            print(f'Iteration: {niters}')
            niters += 1
            loglike.append(mixloglike(theta,self.adu))
            jac = jacmix(theta,self.adu).flatten()
            theta = theta - lr*jac
            dd = lr[:-1]*jac[:-1]
            if np.all(np.abs(dd) < tol):
                converged = True
                break

        return theta, loglike, converged

        

