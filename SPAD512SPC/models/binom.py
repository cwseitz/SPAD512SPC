import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import factorial
from skimage.io import imread
from scipy.stats import binom, poisson
from scipy.signal import convolve
import time

class PoissonBinomialParallel:
    def __init__(self, x, lambd=1, zeta_mean=0.05, zeta_std=0.01):
        self.x = x
        self.lambd = lambd
        self.zeta_mean = zeta_mean
        self.zeta_std = zeta_std

    def log_likelihood(self, N_values, zeta_values, max_cts=10):
        x = np.arange(0, max_cts + 1)[:, None, None]
        N_values = N_values[None, :, None]
        zeta_values = zeta_values[None, None, :]
        binom_pmf = binom.pmf(x, N_values, zeta_values)
        poisson_pmf = poisson.pmf(np.arange(0,max_cts+1),self.lambd)
        sum_pmf = np.apply_along_axis(lambda m: np.convolve(m,poisson_pmf,mode='full'),axis=0,arr=binom_pmf)
        sum_pmf /= np.sum(sum_pmf, axis=0, keepdims=True)
        start = time.time()
        log_like = np.sum(np.log(1e-8+sum_pmf[self.x,:,:]),axis=0) #slow but O(n)
        end = time.time()
        #print(end-start)
        return log_like

    def gaussian_prior(self, zeta_values):
        return norm.pdf(zeta_values, loc=self.zeta_mean, scale=self.zeta_std)

    def integrate(self, num_samples, N_values, approx=False):
        zeta_samples = np.random.normal(self.zeta_mean, self.zeta_std, size=num_samples)
        zeta_samples = zeta_samples[zeta_samples > 0]
        zeta_samples = zeta_samples.astype(np.float32)
        log_likes = self.log_likelihood(N_values, zeta_samples)
        priors = self.gaussian_prior(zeta_samples)
        log_likes -= np.max(log_likes, axis=1, keepdims=True)
        likes = np.exp(log_likes)
        results = np.sum(likes*priors,axis=1) / num_samples

        return results


class PoissonBinomialSerial:
    def __init__(self, x, lambd=1, zeta_mean=0.05, zeta_std=0.01):
        self.x = x
        self.lambd = lambd
        self.zeta_mean = zeta_mean
        self.zeta_std = zeta_std

    def log_likelihood(self,N,zeta,max_cts=10):
        x = np.arange(0,max_cts+1)
        binom_pmf = binom.pmf(x,N,zeta)
        poisson_pmf = poisson.pmf(np.arange(0,max_cts+1),self.lambd)
        sum_pmf = np.convolve(binom_pmf,poisson_pmf,mode='full')
        sum_pmf /= np.sum(sum_pmf)
        log_like = np.sum(np.log(1e-8+sum_pmf[self.x]))
        return log_like

    def monte_log_likelihood(self,N,zeta,nsamples=1000,max_cts=200,plot=False):
        true_counts = np.random.binomial(N,zeta,size=nsamples)
        bg_counts = np.random.poisson(self.lambd,size=nsamples)
        mc_counts = true_counts + bg_counts
        bins = np.arange(0,max_cts,1)
        pmf, _ = np.histogram(self.x, bins=bins, density=True)
        mc_pmf, _ = np.histogram(mc_counts, bins=bins, density=True)
        if plot:
            plt.bar(bins[:-1],pmf,color='red');
            plt.bar(bins[:-1],mc_pmf,color='blue')
            plt.show()
        log_like = np.sum(np.log(1e-8+mc_pmf[self.x]))
        return log_like

    def gaussian_prior(self, zeta):
        return norm.pdf(zeta, loc=self.zeta_mean, scale=self.zeta_std)

    def integrate(self, num_samples, N, approx=False):
        zeta_samples = np.random.normal(self.zeta_mean,self.zeta_std,size=num_samples)
        zeta_samples = zeta_samples[zeta_samples > 0]
        likes = []; priors = []
        for n,zeta in enumerate(zeta_samples):
            log_like = self.log_likelihood(N,zeta)
            print(n)
            prior = self.gaussian_prior(zeta)
            likes.append(log_like); priors.append(prior)
        likes = np.array(likes)
        priors = np.array(priors)
        likes -= np.max(likes)
        likes = np.exp(likes)
        return np.sum(likes*priors)/num_samples

