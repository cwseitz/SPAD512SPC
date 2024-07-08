import numpy as np
import matplotlib.pyplot as plt
from pipes import Pipeline
from skimage.io import imread

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
stack,det = pipe.detect_and_read(summ,threshold=0.0003)
np.savez(patchfile,stack=stack,det=det)

npz = np.load(patch_file)
stack = npz['stack']
counts = np.sum(stack,axis=(2,3))
nt,ndet = counts.shape
Ns = np.arange(1,20,1)

for n in range(ndet):
    avg_post = pipe.post(counts[:,n])
    fig,ax=plt.subplots(1,3,figsize=(9,3))
    patch_sum = np.sum(stack[:,n,:,:],axis=0)
    total_counts = np.sum(patch_sum)
    ax[0].imshow(patch_sum,cmap='gray')
    ax[0].set_title(f'{total_counts} cts')
    ax[1].plot(counts[:,n],color='black')
    ax[2].bar(Ns,avg_post,alpha=0.3, color='red')
    ax[2].set_xlim([0,20])
    ax[2].set_xticks(np.arange(0,20,2))
    ax[2].set_xlabel('N')
    ax[2].set_ylabel('Posterior Probability')
    plt.tight_layout()
plt.show()


