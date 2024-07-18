from SPAD512SPC.generators import Disc2D
from SPAD512SPC.psf.psf2d.mix import MixMCMC,MixMCMCParallel
from SPAD512SPC.utils import match_coordinates
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np

config = {
'nx': 10,
'ny': 10,
'sigma': 0.8,
'particles': 1,
'lamb': 0.0003,
'N0': 200
}
muB = 150.0
iters = 1000
xerrs = []; yerrs = []

for n in range(iters):
    generator = Disc2D(config)
    adu,theta_true = generator.forward(radius=1.0)
    theta_true = theta_true[:2,:].T
    adu = np.clip(adu-muB,0,None)
    theta0 = np.array([5.0,5.0])
    theta0 = np.repeat(theta0[None,:],config['particles'],axis=0).flatten()
    sampler = MixMCMCParallel(theta0,adu,sigma=config['sigma'],
                      N0=config['N0'])
    samples = sampler.run_mcmc(plot_fit=False)
    samples = samples[:,:2]
    theta_est = sampler.find_modes_dpgmm(samples,max_components=config['particles'])
    theta_est[:,[0,1]] = theta_est[:,[1,0]]
    xerr,yerr = match_coordinates(theta_est,theta_true)
    xerrs += list(xerr); yerrs += list(yerr)

xerrs = np.array(xerrs); yerrs = np.array(yerrs)
print(np.std(xerrs),np.std(yerrs))
bins = np.linspace(-1.0,1.0,30)
fig,ax=plt.subplots(1,2)
ax[0].hist(xerrs,bins=bins)
ax[1].hist(yerrs,bins=bins)
plt.show()
