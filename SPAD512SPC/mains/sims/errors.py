from SPAD512SPC.generators import Ring2D
from SPAD512SPC.psf.psf2d.mix import MixMCMC,MixMCMCParallel
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np

config = {
'nx': 10,
'ny': 10,
'sigma': 0.8,
'particles': 1,
'lamb': 0.0003,
'N0': 5000
}

generator = Ring2D(config)
adu = generator.forward(ring_radius=1.0)
muB = 150.0
adu = np.clip(adu-muB,0,None)

theta0 = np.array([5.0,5.0])
theta0 = np.repeat(theta0[None,:],config['particles'],axis=0).flatten()

sampler = MixMCMC(theta0,adu,sigma=config['sigma'],
                  N0=config['N0'])
samples = sampler.run_mcmc(plot_fit=True)
samples = samples[:,:2]
theta_est = sampler.find_modes_dpgmm(samples,max_components=config['particles'])
