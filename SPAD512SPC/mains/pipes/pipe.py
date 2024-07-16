import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from scipy.ndimage import median_filter
from scipy import stats
from SPAD512SPC.utils import IntensityReaderSparse
from SPAD512SPC.models import PoissonBinomialParallel, coincidence_ratio
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from SPAD512SPC.psf.psf2d.mix import MixMCMC

class Pipeline: 
    def __init__(self,basepath,acqs):
        self.basepath = basepath
        self.acqs = acqs
    def detect(self,summ,threshold=0.0005,show=True):
        med = median_filter(summ/summ.max(),size=3)
        det = blob_log(med,threshold=threshold,min_sigma=1,max_sigma=5,
                       num_sigma=5,exclude_border=True)
        det = det[:,:2]
        if show:
            plt.imshow(med,cmap='gray',vmin=0,vmax=0.005)
            plt.scatter(det[:,1],det[:,0],marker='x',s=3,color='red')
            plt.show()
        print(f'Num detections: {det.shape[0]}')
        return det 
    def extract_patches(self,frame,det,patchw=2):
        patches = []
        patch_size = 2*patchw + 1
        ndet,_ = det.shape
        for n in range(ndet):
            x,y = det[n]
            xmin = int(x)-patchw; xmax = int(x)+patchw+1
            ymin = int(y)-patchw; ymax = int(y)+patchw+1
            patch = frame[xmin:xmax,ymin:ymax]
            patches.append(patch)
        return np.array(patches)
    def detect_and_read(self,summ,threshold=0.0005,patchw=2):
        coords = self.detect(summ,threshold=threshold)
        w = 2*patchw + 1
        ndet,_ = coords.shape
        stack = []
        for this_acq in self.acqs:
            print('Reading: ' + this_acq)
            reader = IntensityReaderSparse(self.basepath+this_acq)
            generator = reader.read_1bit_sparse_frames()
            for frame in generator:
                stack.append(self.extract_patches(frame,coords))
            del reader
        stack = np.array(stack)
        return stack,coords
    def read_frames(self,coords,max_frames=None):
        stack = []
        for this_acq in self.acqs:
            print('Reading: ' + this_acq)
            reader = IntensityReaderSparse(self.basepath+this_acq)
            generator = reader.read_1bit_sparse_frames()
            n = 0
            for frame in generator:
                print(f'Reading frame {n}')
                if max_frames is not None:
                    if n > max_frames:
                        break
                stack.append(self.extract_patches(frame,coords))
                n += 1
            del reader
        stack = np.array(stack)
        return stack
    def coincidence(self,counts,B=37.5,conf_thres=0.5):
        g20,sigma = coincidence_ratio(counts,B=B)
        conf = stats.norm.cdf(conf_thres,loc=g20,scale=sigma)
        g20 = np.round(g20,2)
        sigma = np.round(sigma,2)
        conf = np.round(conf,2)
        return g20,sigma,conf
    def get_post(self,counts,Nmax=20,zeta_mean=0.01,zeta_std=0.005,
                lambd=0.01,num_samples=100,nbatches=50):
                
        Ns = np.arange(1,Nmax,1)
        counts = np.split(counts,nbatches) #minibatch
        posts = []
        for n,this_count in enumerate(counts):
            model = PoissonBinomialParallel(this_count,lambd=lambd,
                                            zeta_mean=zeta_mean,
                                            zeta_std=zeta_std)
            post = model.integrate(num_samples,Ns)
            post = post/np.sum(post)
            posts.append(post)
            del model
        posts = np.array(posts)
        avg_post = np.mean(posts,axis=0)
        return avg_post
    def plot_post(self,Ns,avg_post,patch_sum):
        fig,ax=plt.subplots(figsize=(3,3))
        ax.bar(Ns,avg_post,color='white',edgecolor='black')
        ax_inset = inset_axes(ax,width="40%",
                              height="40%",loc='upper right')
        ax.set_xlim([0,10])
        ax.set_xticks(np.arange(0,10,2))
        ax.set_xlabel(r'$N$')
        ax.set_ylabel(r'$p(N|n)$')
        ax.set_yticks([])
        ax.spines[['right','top']].set_visible(False)
        ax_inset.imshow(patch_sum,cmap='gray')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        plt.tight_layout()    
    def post(self,counts,Ns,lambd=0.0075,plot=False):
        avg_post = self.get_post(counts,lambd=lambd)
        Nmap = np.argmax(avg_post)+1
        model = PoissonBinomialParallel(None,lambd=lambd,
                                        zeta_mean=None,
                                        zeta_std=None)
        return avg_post,Nmap
        
    def fit(self,adu,Nmap,muB,sigma=0.8,
            N0=5000,max_components=5,plot=False):
        adu = np.clip(adu-muB,0,None)
        adu = np.pad(adu,((2,2),(2,2)))
        nx,ny = adu.shape
        theta0 = np.array([nx/2,nx/2])
        theta0 = np.repeat(theta0[None,:],Nmap,axis=0).flatten()
        sampler = MixMCMC(theta0,adu,sigma=sigma,N0=N0)
        samples = sampler.run_mcmc(plot_fit=True)
        samples = samples[:,:2]
        theta_est = sampler.find_modes_dpgmm(samples,max_components=max_components)
        if plot:
            fig,ax=plt.subplots(figsize=(3,3))
            ax.scatter(theta_est[:,0],theta_est[:,1],color='red')
            ax.invert_yaxis()
            ax.imshow(adu,cmap='gray')
            ax.set_xticks([]); ax.set_yticks([])
        return theta_est

