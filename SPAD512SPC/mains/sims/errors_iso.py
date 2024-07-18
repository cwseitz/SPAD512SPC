from miniSMLM.generators import Disc2D
from miniSMLM.psf.psf2d import MLE2D_BFGS
from SPAD512SPC.utils import match_coordinates
from skimage.io import imread,imsave
import matplotlib.pyplot as plt
import numpy as np

config = {
'sigma': 0.8,
'N0': 1000,
'B0': 0,
'eta': 1.0,
'texp': 1.0,
'gain': 1.0,
'offset': 0.0,
'var': 0.0
}

iters = 1000
xerrs = []; yerrs = []
for n in range(iters):
    generator = Disc2D(10,10)
    adu,spikes,theta_true = generator.forward(1.0,1,**config,show=False)
    theta_true = np.squeeze(theta_true)[:2]
    adu = np.clip(adu-config['B0'],0,None)
    theta0 = np.array([5.0,5.0,config['N0']])
    theta0 += np.random.normal(0,0.5,size=theta0.shape)
    opt = MLE2D_BFGS(theta0,adu,config)
    theta,loglike,converged,_ = opt.optimize(max_iters=1000,plot_fit=False)
    xerr,yerr = theta[:2]-theta_true[:2]
    xerrs.append(xerr); yerrs.append(yerr)


xerrs = np.array(xerrs); yerrs = np.array(yerrs)
print(np.std(xerrs),np.std(yerrs))
bins = np.linspace(-5.0,5.0,50)
fig,ax=plt.subplots(1,2)
ax[0].hist(xerrs,bins=bins)
ax[1].hist(yerrs,bins=bins)
plt.show()

