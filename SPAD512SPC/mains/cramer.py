import numpy as np
import json
import matplotlib.pyplot as plt
from skimage.io import imsave
from SPAD512SPC.psf.psf2d import crlb2d

class CRB2D:
    def __init__(self):
        pass   
    def forward(self,N0space,sigma=0.8,B0=0):
        crlb_N0 = np.zeros((len(N0space),2))
        for i,N0 in enumerate(N0space):
            theta0 = np.array([5,5])
            crlb_N0[i] = crlb2d(theta0,sigma=sigma,N0=N0,B0=B0)
        return crlb_N0

pixel_size=125
fig,ax=plt.subplots(figsize=(3.2,3.2))
N0space = np.linspace(500,5000,500)
crb = CRB2D()
crlb_N0 = crb.forward(N0space,B0=0)
ax.loglog(N0space,crlb_N0[:,0]*pixel_size,
          color='black',linestyle='--',label=r'$\langle n_{background}\rangle=0$')
crlb_N0 = crb.forward(N0space,B0=20)
ax.loglog(N0space,crlb_N0[:,0]*pixel_size,
          color='red',linestyle='--',label=r'$\langle n_{background}\rangle=20$')
crlb_N0 = crb.forward(N0space,B0=150)
ax.loglog(N0space,crlb_N0[:,0]*pixel_size,
          color='blue',linestyle='--',label=r'$\langle n_{background}\rangle=150$')

ax.set_xlabel('Intensity (photons)')
ax.set_ylabel('Localization uncertainty (nm)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

