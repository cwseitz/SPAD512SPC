import numpy as np
import matplotlib.pyplot as plt
from pipes import Pipeline
from skimage.io import imread
from SPAD512SPC.models import PoissonBinomialParallel

class PostImage:
    def __init__(self,path,acqs,npix=100,nframes=100000,xmin=130,ymin=140):
        self.path = path
        self.patch_file = path+'patch_data_g2_image.npz'
        self.acqs = acqs
        self.npix = npix
        self.nframes = nframes
        self.xmin = xmin
        self.ymin = ymin
        self.pipe = Pipeline(path, acqs)
        self.x = np.arange(xmin, xmin + npix)
        self.y = np.arange(ymin, ymin + npix)
        self.xx, self.yy = np.meshgrid(self.x, self.y)
        self.coords = np.stack((self.xx.ravel(),self.yy.ravel()),axis=-1)
        self.summ = None
        self.stack = None
        self.coords = None
        self.log_like_image = np.zeros((npix,npix))

    def write_patch_file(self):
        stack = self.pipe.read_frames(self.coords, max_frames=self.nframes)
        np.savez(self.patch_file, stack=stack, det=self.coords)

    def get_image(self,zeta=0.01,Nmax=20,thresh=150,lambd=0.01875):
        self.summ = imread(self.path+'SUM.tif')[self.xmin:self.xmin+self.npix, self.ymin:self.ymin+self.npix]
        npz = np.load(self.patch_file)
        self.stack = npz['stack']
        self.coords = npz['det']
        for n in range(self.npix**2):
            counts = np.sum(self.stack[:,n,:,:],axis=(1, 2))
            total_counts = np.sum(self.stack[:,n,2,2])
            if total_counts > thresh:
                model = PoissonBinomialParallel(counts,lambd=lambd)
                Ns = np.arange(1,Nmax,1)
                zetas = np.array([zeta])
                log_like = model.log_likelihood(Ns,zetas)
                log_like = np.squeeze(log_like)
                log_like = np.exp(log_like-log_like.max())
                log_like /= np.sum(log_like)
                prob = np.sum(log_like[:2]) 
                if np.isnan(log_like).any():
                    prob = 0.0
                if prob < 1e-8:
                    prob = 0.0
                print(f'Pixel {n}: {total_counts, prob}')
                self.log_like_image[self.coords[n,0]-self.xmin,self.coords[n,1] - self.ymin] = prob

    def show_image(self):
        fig, ax = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=True)
        im1 = ax[0].imshow(self.log_like_image,cmap='coolwarm')
        im2 = ax[1].imshow(self.summ,cmap='gray',vmin=0.0,vmax=1000)
        plt.colorbar(im1,ax=ax[1],fraction=0.046,pad=0.04,label=r'$\mathrm{Pr}(N<2)$')
        plt.colorbar(im2,ax=ax[2],fraction=0.046,pad=0.04,label='cts')
        plt.tight_layout()
        plt.show()

path = '/research3/shared/cwseitz/Data/SPAD/240817/data/intensity_images/'
acqs = [
'acq00001/'
]

npix = 100
nframes = 100000
xmin, ymin = 130,140
#xmin, ymin = 360,340
post_image = PostImage(path,acqs,npix=npix,nframes=nframes,xmin=xmin,ymin=ymin)     
post_image.get_image()
post_image.show_image()

