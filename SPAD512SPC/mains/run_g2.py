import numpy as np
import matplotlib.pyplot as plt
from pipes import Pipeline
from skimage.io import imread
from SPAD512SPC.utils import color_sum_g20
from SPAD512SPC.models import correlate
from skimage.filters import gaussian

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
counts = np.sum(stack,axis=(2,3))
nt,ndet = counts.shape
Nmax = 20
Ns = np.arange(1,Nmax,1)

plot=True
g20s = []
for n in range(ndet):
    this_counts = counts[:,n]
    g20,sigma,conf = pipe.coincidence(this_counts)
    patch_sum = np.sum(stack[:,n,:,:],axis=0)
    total_counts = np.sum(patch_sum)
    g20s.append(g20)
    
    avg_post = np.zeros_like(Ns)
    if g20 < 0.5:
        #avg_post = pipe.post(this_counts,Nmax=Nmax)
        this_stack = stack[:,n,:,:]
        #print(this_stack.shape)
        #this_stack = this_stack.reshape(-1, 1000, 5, 5)
        #this_stack = this_stack.sum(axis=1)
        #corr = correlate(this_stack)
        #fig,ax=plt.subplots(1,3)
        #ax[0].imshow(patch_sum)
        #ax[1].imshow(corr)
        #ax[2].imshow(gaussian(corr,sigma=0.5))
        #plt.show()
        
        fig,ax=plt.subplots(1,2)
        this_stack_ = this_stack.reshape(500000, 25)
        cmatrix = np.dot(this_stack_.T,this_stack_)
        np.fill_diagonal(cmatrix,0)
        print(np.sum(cmatrix),g20,total_counts)
        ax[0].imshow(np.sum(this_stack,axis=0))
        ax[1].imshow(cmatrix,vmin=0,vmax=5)
        plt.show()
        
        """
        if plot:
            fig,ax=plt.subplots(1,3,figsize=(10,3))
            im = ax[0].imshow(patch_sum,cmap='gray')
            ax[0].set_title(f'{total_counts} cts')
            ax[1].plot(counts[:,n],color='black')
            valstr = r'$g^{(2)}(0)=$' + str(g20) +\
             r' $\sigma=$' + str(sigma) + f' confidence={conf}'
            ax[1].set_xlabel('Frame')
            ax[1].set_ylabel('cts')
            ax[1].set_title(valstr)
            ax[2].bar(Ns,avg_post,alpha=0.3, color='red')
            ax[2].set_xlim([0,Nmax])
            ax[2].set_xticks(np.arange(0,Nmax,2))
            ax[2].set_xlabel('N')
            ax[2].set_ylabel('Posterior Probability')
            plt.colorbar(im,ax=ax[0],label='cts',
                         fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()

g20s = np.array(g20s)
coords = npz['det']
color_sum_g20(summ,coords,g20s)
"""



