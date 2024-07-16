import numpy as np
import matplotlib.pyplot as plt
from pipes import Pipeline
from skimage.io import imread
from SPAD512SPC.utils import estimate_background
from SPAD512SPC.models import PoissonBinomialParallel
from SPAD512SPC.psf.psf2d.mix import MixMCMC

path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images_2/'
patch_file = path+'patch_data.npz'

acqs = [
'acq00003/',
'acq00004/',
'acq00005/',
'acq00006/',
'acq00007/'
]

pipe = Pipeline(path,acqs)
summ = imread(path+'SUM.tif')

#stack,det = pipe.detect_and_read(summ,threshold=0.0003)
#np.savez(patch_file,stack=stack,det=det)

npz = np.load(patch_file)
stack = npz['stack']
coords = npz['det']
counts = np.sum(stack,axis=(2,3))
nt,ndet = counts.shape
Ns = np.arange(1,20,1)
patchw=2

for n in range(ndet):
    this_counts = counts[:,n]
    g20,sigma,conf = pipe.coincidence(this_counts)
    patch_sum = np.sum(stack[:,n,:,:],axis=0)
    x0,y0 = coords[n,:].astype(np.int16)
    total_counts = np.sum(patch_sum)
    avg_post,Nmap = pipe.post(this_counts,Ns)    
    print(g20,sigma,conf)
    if Nmap == 4:
        muB = np.min(patch_sum)
        theta_est =\
        pipe.fit(patch_sum,Nmap,muB,N0=5000,
                 max_components=Nmap,plot=True)
        pipe.plot_post(Ns,avg_post,patch_sum)
        plt.show()
            



