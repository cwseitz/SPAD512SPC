import numpy as np
import matplotlib.pyplot as plt
from pipes import Pipeline
from skimage.io import imread
from SPAD512SPC.utils import color_sum_g20
from SPAD512SPC.models import correlate
from skimage.filters import gaussian

path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images_2/'
patch_file = path+'patch_data_g2_image.npz'

acqs = [
'acq00003/',
#'acq00004/',
#'acq00005/',
#'acq00006/',
#'acq00007/'
]

npix = 50
nframes = 100000
lamb = 0.0075
zeta_est = 0.01
xmin, ymin = 360,340
pipe = Pipeline(path,acqs)


x = np.arange(xmin,xmin+npix)
y = np.arange(ymin,ymin+npix)
xx, yy = np.meshgrid(x, y)
coords = np.stack((xx.ravel(), yy.ravel()), axis=-1)
#stack = pipe.read_frames(coords,max_frames=nframes)
#nt,_,nx,ny = stack.shape
#np.savez(patch_file,stack=stack,det=coords)

summ = imread(path+'SUM.tif')
summ = summ[xmin:xmin+npix,ymin:ymin+npix]
summ[summ > 400] = 200

npz = np.load(patch_file)
stack = npz['stack']; coords = npz['det']
counts = np.sum(stack,axis=0)
hotpix = counts > 150
stack[:,hotpix] = 0


g2_image = np.zeros((npix,npix))
conf_image = np.zeros_like(g2_image)
sum_image = np.zeros_like(g2_image)
for n in range(npix**2):
    counts = np.sum(stack[:,n,:,:],axis=(1,2))
    g20,sigma,conf = pipe.coincidence(counts,B=nframes*lamb*zeta_est)
    print(f'Pixel {n}: {g20}')
    g2_image[coords[n,0]-xmin,coords[n,1]-ymin] = g20
    conf_image[coords[n,0]-xmin,coords[n,1]-ymin] = conf
    sum_image[coords[n,0]-xmin,coords[n,1]-ymin] = np.sum(stack[:,n,2,2])
    
g2_image = gaussian(g2_image,sigma=0.75)
conf_image = gaussian(conf_image,sigma=0.75)
summ = gaussian(summ,sigma=0.5)
fig,ax=plt.subplots(1,3,figsize=(12,3),sharex=True,sharey=True)
im1 = ax[1].imshow(g2_image,cmap='coolwarm',vmin=0.0,vmax=1.0)
im2 = ax[2].imshow(conf_image,cmap='coolwarm',vmin=0.0,vmax=1.0)
im3 = ax[0].imshow(summ,cmap='gray',vmin=0.0,vmax=300)
plt.colorbar(im1,ax=ax[1],fraction=0.046,pad=0.04,label=r'$g^{(2)}(0)$')
plt.colorbar(im2,ax=ax[2],fraction=0.046,pad=0.04,label=r'$\mathrm{Pr}(g^{(2)}(0) < 1/2)$')
plt.colorbar(im3,ax=ax[0],fraction=0.046,pad=0.04,label='cts')
plt.tight_layout()
plt.savefig('/home/cwseitz/Desktop/Figure-2.png',dpi=300)
plt.show()





