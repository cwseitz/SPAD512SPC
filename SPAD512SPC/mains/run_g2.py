import numpy as np
import matplotlib.pyplot as plt
from pipes import Pipeline
from skimage.io import imread
from SPAD512SPC.utils import color_sum_g20
#from SPAD512SPC.models import correlate
from scipy.signal import correlate,correlation_lags
from skimage.filters import gaussian
import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 200

path = '/research2/shared/cwseitz/Data/SPAD/240814/data/intensity_images/'
patch_file = path+'patch_data.npz'

acqs = [
'acq00002/'
]

pipe = Pipeline(path,acqs)
summ = imread(path+'SUM.tif')
stack,det = pipe.detect_and_read(summ,threshold=0.0005)
np.savez(patch_file,stack=stack,det=det)

npz = np.load(patch_file)
stack = npz['stack']
counts = np.sum(stack,axis=(2,3))
nt,ndet = counts.shape
Nmax = 20
Ns = np.arange(1,Nmax,1)

plot=True
g20s = []
numt = 100000
for n in range(ndet):
    this_counts = counts[:numt,n]
    summ = np.sum(stack[:,n,:,:],axis=0)
    Best = np.min(summ)
    g20,sigma,conf,G20,G2ms = pipe.coincidence(this_counts)
    
    if g20 < 0.2:
        lags = np.arange(-50,50,1)
        G2ms[np.argmax(G2ms)] = G20
        Gmavg = np.mean(G2ms)
        g20 = (G20 - Best*0.01)/(Gmavg - Best*0.01)
        print(G20/Gmavg,g20)
        fig,ax=plt.subplots(1,2,figsize=(6,3))
        im1 = ax[0].imshow(summ,cmap='gray')
        plt.colorbar(im1,ax=ax[0],fraction=0.046,pad=0.04,label='cts')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[1].plot(lags,G2ms,color='red')
        ax[1].scatter(lags,G2ms,s=20,facecolor='white',edgecolor='black')
        ax[1].hlines(Gmavg,-50,50,color='blue',
                 linestyle='--',label=r'$\langle G^{(2)}(m)\rangle$')
        ax[1].set_xlabel(r'Lag $m$ (frames)',fontsize=10)
        ax[1].set_ylabel(r'$G^{(2)}(m)$',fontsize=10)
        ax[1].spines[['right', 'top']].set_visible(False)
        ax[1].legend(frameon=False,bbox_to_anchor=(0.5,1.2))
        plt.tight_layout()
        plt.show()
        
"""
    g20,sigma,conf = pipe.coincidence(this_counts)
    patch_sum = np.sum(stack[:,n,:,:],axis=0)
    total_counts = np.sum(patch_sum)
    g20s.append(g20)
    avg_post = np.zeros_like(Ns)
    if g20 < 0.5:
        avg_post = pipe.post(this_counts,Nmax=Nmax)
        this_stack = stack[:,n,:,:]
        print(this_stack.shape)
        this_stack = this_stack.reshape(-1, 1000, 5, 5)
        this_stack = this_stack.sum(axis=1)
        corr = correlate(this_stack)
        fig,ax=plt.subplots(1,3)
        ax[0].imshow(patch_sum)
        ax[1].imshow(corr)
        ax[2].imshow(gaussian(corr,sigma=0.5))
        plt.show()
                
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



