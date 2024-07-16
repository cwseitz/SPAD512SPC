from SPAD512SPC.generators import SPAD2D_Ring
from SPAD512SPC.psf.psf2d.mix import MLE2DMix, MixMCMC
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np

config = {
'nx': 10,
'ny': 10,
'sigma': 0.8,
'particles': 3,
'lamb': 0.0003,
'zeta': 0.01,
'nt': 100000,
}
generator = SPAD2D_Ring(config)
x,theta_true = generator.generate(ring_radius=0.5,show=True)

adu = np.sum(x,axis=0)
muB = config['lamb']*config['nt']
adu = np.clip(adu-muB,0,None)

theta0 = np.array([5.0,5.0])
theta0 = np.repeat(theta0[None,:],config['particles'],axis=0).flatten()

sampler = MixMCMC(theta0,adu,sigma=config['sigma'],
                  N0=config['zeta']*config['nt'])
samples = sampler.run_mcmc(plot_fit=True)
samples = samples[:,:2]
theta_est = sampler.find_modes_dpgmm(samples,max_components=6)

fig,ax=plt.subplots()
ax.scatter(theta_true[1,:],theta_true[0,:],color='red')
ax.invert_yaxis()
ax.imshow(adu,cmap='gray')
plt.show()
plt.show()


"""
counts = np.sum(x,axis=(1,2))
first_six = np.argwhere(counts >= 1.0)[:6]
fig,ax=plt.subplots(2,3)
ax = ax.ravel()
for n,idx in enumerate(first_six):
    ax[n].imshow(np.squeeze(x[idx]),cmap='gray',vmin=0,vmax=1)
    ax[n].set_xticks([]); ax[n].set_yticks([])
    ax[n].set_title(f'Frame {idx}')
plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/BinaryImages.png',dpi=300)
plt.show()
"""


