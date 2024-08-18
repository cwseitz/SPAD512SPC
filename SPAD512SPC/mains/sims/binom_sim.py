import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
from SPAD512SPC.models import *
from binom import BinomialPoissonSimulation

zetas = [0.02,0.05]
ns = np.arange(1,10,1)
nframes = 100000
lamb = 0.0075
colors = ['red','blue','lime']

fig,ax=plt.subplots(figsize=(4,2.5))
for n,zeta in enumerate(zetas):
    sim = BinomialPoissonSimulation()
    tavg = sim.avg_g20(ns,zeta,lamb,nframes=nframes) #theory
    avg_g20,avg_sigma = sim.run_simulation_g20(zeta,ns,nframes,lamb=lamb)
    ax.errorbar(ns,avg_g20,avg_sigma,capsize=5,markersize=3, 
                marker='o',ls='none',color=colors[n],
                label=rf'$\zeta={zeta}$')
    ax.plot(ns,tavg,color=colors[n]) #theory
    ax.set_xticks([2,4,6,8,10])
    ax.set_xlabel('Number of Flurophores (N)')
    ax.set_ylabel(r'$g^{(2)}(0)$')
    ax.legend()
    plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/Figure-1.png', dpi=300)
plt.show()

