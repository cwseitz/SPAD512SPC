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
#stack,det = pipe.read(summ)
#np.savez(patchfile,stack=stack,det=det)
npz = np.load(patch_file)
stack = npz['stack']
counts = np.sum(stack,axis=(2,3))
print(counts.shape)

"""
fig,ax=plt.subplots(1,3,figsize=(9,3))
ax[0].set_title(f'{np.sum(counts)} cts')
ax[0].imshow(np.sum(stack1bit,axis=0),cmap='gray')
ax[1].plot(counts,color='black')
ax[2].bar(Ns,avg_post,alpha=0.3, color='red')
ax[2].set_xlim([0,20])
ax[2].set_xticks(np.arange(0,20,2))
ax[2].set_xlabel('N')
ax[2].set_ylabel('Posterior Probability')
plt.tight_layout()
plt.show()
"""

