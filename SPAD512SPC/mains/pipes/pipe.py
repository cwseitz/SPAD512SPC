import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import blob_log
from scipy.ndimage import median_filter
from SPAD512SPC.utils import IntensityReaderSparse

class Pipeline: 
    def __init__(self,basepath,acqs):
        self.basepath = basepath
        self.acqs = acqs
    def detect(self,summ,show=True):
        med = median_filter(summ/summ.max(),size=3)
        det = blob_log(med,threshold=0.0005,min_sigma=1,max_sigma=5,
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
    def read(self,summ,patchw=2):
        det = self.detect(summ)
        w = 2*patchw + 1
        ndet,_ = det.shape
        stack = []
        for this_acq in self.acqs:
            print('Reading: ' + this_acq)
            reader = IntensityReaderSparse(self.basepath+this_acq)
            generator = reader.read_1bit_sparse_frames()
            for frame in generator:
                stack.append(self.extract_patches(frame,det))
            del reader
        stack = np.array(stack)
        return stack,det
    def post(self,counts,Nmax=20,zeta_mean=0.01,zeta_std=0.005,
                lambd=0.01,num_samples=100,nbatches=1):
                
        Ns = np.arange(1,Nmax,1)
        counts = np.split(counts,nbatches) #minibatch
        model = PoissonBinomialParallel(this_count,lambd=lambd,
                                        zeta_mean=zeta_mean,zeta_std=zeta_std)
        posts = []
        for n,this_count in enumerate(counts):
            post = model.integrate(num_samples,Ns)
            post = post/np.sum(post)
            posts.append(post)
        posts = np.array(posts)
        avg_post = np.mean(posts,axis=0)
        return avg_post

