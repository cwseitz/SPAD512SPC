import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.io import imsave
from miniSMLM.psf.psf2d.mix import crlb2d

class CRB2D:
    def __init__(self):
        pass   
    def forward(self,N0space,sigma=0.8,B0=0):
        cam_params = [1.0,1.0,1.0,0.0,0.0]
        crlb_N0 = np.zeros((len(N0space),2))
        for i,N0 in enumerate(N0space):
            theta0 = np.array([5,5])
            crlb_N0[i] = crlb2d(theta0,cam_params,sigma=sigma,N0=N0,B0=B0)
        return crlb_N0
        
N0space_ = np.array([1000,5000,10000])
N1 = np.array([0.236,0.06,0.036])
N2 = np.array([0.732,0.1732,0.111])
N3 = np.array([0.747,0.262,0.197])

pixel_size=125
fig,ax=plt.subplots(figsize=(3.2,3.2))
N0space = np.linspace(1000,10000,500)
crb = CRB2D()
crlb_N0 = crb.forward(N0space,B0=150)
ax.plot(N0space,crlb_N0[:,0]*pixel_size,
          color='gray',linestyle='--',label=r'$\sigma_{CRLB}$')
ax.scatter(N0space_,N1*pixel_size,marker='x',color='red',label=r'$N=1$')
ax.scatter(N0space_,N2*pixel_size,marker='x',color='blue',label=r'$N=2$')
ax.scatter(N0space_,N3*pixel_size,marker='x',color='lime',label=r'$N=3$')

ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlabel('Intensity (photons)')
ax.set_ylabel('Localization uncertainty (nm)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/Errors.png',dpi=300)
plt.show()

