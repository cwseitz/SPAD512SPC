from skimage.restoration import rolling_ball
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

def estimate_background(image,radius=10,plot=True):
    image = median_filter(image,size=2)
    background = rolling_ball(image,radius=radius)
    if plot:
        fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
        ax[0].imshow(image,vmin=0,vmax=1000,cmap='gray')
        ax[1].imshow(image-background,vmin=0,vmax=1000,cmap='gray')
        ax[2].imshow(background,vmin=0,vmax=150,cmap='gray')
        plt.tight_layout()
        plt.show()
    return background
