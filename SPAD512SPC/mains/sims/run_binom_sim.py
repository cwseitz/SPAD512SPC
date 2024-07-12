import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from SPAD512SPC.models import *

class BinomialPoissonSimulation:
    def __init__(self):
        pass
    def prob_bi_greater_than_or_equal_two(self,n,zeta):
        return 1 - binom.pmf(0, n, zeta) - binom.pmf(1, n, zeta)
    def prob_bim_greater_than_or_equal_two(self,n,zeta):
        return (1 - binom.pmf(0, n, zeta))**2
    def sample(self,n,zeta,nframes,lamb=0.0075):
        xsignal = binom.rvs(n,zeta,size=(nframes,))
        xbackground = poisson.rvs(lamb,size=(nframes,)) 
        x = xsignal + xbackground  
        return x         
    def avg_g20(self,ns,zeta,lamb,nframes=500000):
        B = nframes*lamb*zeta
        avgG20 = nframes*self.prob_bi_greater_than_or_equal_two(ns,zeta)
        avgG2m = 100*self.prob_bim_greater_than_or_equal_two(ns,zeta)
        return (avgG20-B)/(avgG2m-B)
    def post(self,x,ns,nbatches=50,lamb=0.0075,zeta_mean=0.01,zet_std=0.005):
        x = np.split(x,nbatches)
        posts = []
        for n,this_x in enumerate(x):
            model = PoissonBinomialParallel(this_x,lambd=lamb,
                                            zeta_mean=zeta_mean,
                                            zeta_std=zeta_std)
            post = model.integrate(num_samples,ns)
            post = post/np.sum(post)
            posts.append(post)
            del model
        posts = np.array(posts)
        avg_post = np.mean(posts,axis=0)
        return avg_post
    def run_simulation(self,zeta,ns,nframes,numm,lamb=0.0075,iters=100):
        all_g20s = []; all_sigmas = []
        for i in range(iters):
            print(f'Iteration {i}')
            g20s = []; sigmas = []
            for n in ns:
                x = self.sample(n,zeta,nframes,lamb=lamb)
                g20,sigma = coincidence_ratio(x,B=nframes*lamb*zeta)
                g20s.append(g20); sigmas.append(sigma)
            all_g20s.append(np.array(g20s))
            all_sigmas.append(np.array(sigmas))
        avg_g20 = np.mean(all_g20s,axis=0)
        avg_sigma = np.mean(all_sigmas,axis=0)
        return avg_g20,avg_sigma


zetas = [0.015, 0.03, 0.05]
zetas = [0.015]
ns = np.arange(1,10,1)
nframes = 500000
numm = 100
lamb = 0.0075
colors = ['red','blue','lime']
colors = ['red']

fig,ax=plt.subplots(figsize=(3,3))
for n,zeta in enumerate(zetas):
    sim = BinomialPoissonSimulation()
    avg_g20,avg_sigma = sim.run_simulation(zeta,ns,nframes,numm,lamb=lamb)
    tavg = sim.avg_g20(ns,zeta,lamb,nframes=nframes)
    ax.errorbar(ns,avg_g20,avg_sigma,capsize=5,markersize=3, 
                marker='o',ls='none',color=colors[n],
                label=rf'$\zeta={zeta}$')
    ax.plot(ns,tavg,color=colors[n])
    ax.set_xticks([2,4,6,8,10])
    ax.set_xlabel('Number of Flurophores (N)')
    ax.set_ylabel(r'$g^{(2)}(0)$')
    ax.legend()
    plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/Figure-1.png', dpi=300)
plt.show()

