import numpy as np
import matplotlib.pyplot as plt
from pipes import Pipeline
from skimage.io import imread
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from SPAD512SPC.utils import estimate_background

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
bg = estimate_background(summ)

#stack,det = pipe.detect_and_read(summ,threshold=0.0003)
#np.savez(patch_file,stack=stack,det=det)

npz = np.load(patch_file)
stack = npz['stack']
counts = np.sum(stack,axis=(2,3))
nt,ndet = counts.shape
Ns = np.arange(1,20,1)

for n in range(ndet):
    this_counts = counts[:,n]
    g20,sigma,conf = pipe.coincidence(this_counts)
    patch_sum = np.sum(stack[:,n,:,:],axis=0)
    total_counts = np.sum(patch_sum)
    print(g20,sigma,conf)
    if g20 < 1.0:
        avg_post = pipe.post(this_counts,lambd=0.0075)
        fig,ax=plt.subplots(figsize=(3,3))
        ax.bar(Ns,avg_post,color='white',edgecolor='black')
        ax_inset = inset_axes(ax,width="40%",height="40%",loc='upper right')
        ax.set_xlim([0,20])
        ax.set_xticks(np.arange(0,20,2))
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$p(N|x)$')
        ax.set_yticks([])
        ax.spines[['right','top']].set_visible(False)
        ax_inset.imshow(patch_sum,cmap='gray')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        plt.tight_layout()
        plt.savefig(f'/home/cwseitz/Desktop/Fig3/Figure-3-{n}.png',dpi=300)
        #plt.show()



