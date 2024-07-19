from miniSMLM.generators import Disc2D
from miniSMLM.psf.psf2d.mix import MixMCMC
from SPAD512SPC.utils import match_coordinates
import matplotlib.pyplot as plt
import numpy as np

config_ = {
    'sigma': 0.8,
    'B0': 150,
    'eta': 1.0,
    'texp': 1.0,
    'gain': 1.0,
    'offset': 0.0,
    'var': 0.0
}

def run_simulation(N, N0, config, iters=100):
    xerrs = []
    yerrs = []
    config['N0'] = N0
    for n in range(iters):
        generator = Disc2D(10,10)
        adu, spikes, theta_true = generator.forward(1.0, N, **config, show=False)
        theta_true = np.squeeze(theta_true)[:2]
        adu = np.clip(adu - config['B0'], 0, None)
        theta0 = np.array([5.0, 5.0])
        theta0 = np.repeat(theta0[None, :], N, axis=0).flatten()
        sampler = MixMCMC(theta0, adu, config)
        samples = sampler.run_mcmc(plot_fit=False)
        samples = samples[:, :2]
        theta_est = sampler.cluster_samples(samples, N)
        sampler.plot_fit(samples,adu,theta_true,N)
        if N == 1:
            theta_true = theta_true[:,None]
        xerr, yerr = match_coordinates(theta_est, theta_true.T)
        xerrs += list(xerr)
        yerrs += list(yerr)
    return np.std(xerrs), np.std(yerrs)

N_values = [3]
N0_values = [5000]
iters = 100

fig,ax=plt.subplots()
for N in N_values:
    xerrs_std = []
    yerrs_std = []
    for N0 in N0_values:
        xerr_std, yerr_std = run_simulation(N,N0,config_.copy(),iters)
        xerrs_std.append(xerr_std)
        yerrs_std.append(yerr_std)
    ax.plot(N0_values,xerrs_std)
    
ax.set_xlabel('N0')
ax.set_ylabel('Error')
plt.show()

