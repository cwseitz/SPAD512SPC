import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

def color_sum_g20(image,coords,g20_values,alpha=0.5,marker_size=500):
    colormap = cm.plasma
    colors = colormap(g20_values)
    fig, ax = plt.subplots()
    med = median_filter(image/image.max(),size=2)
    ax.imshow(med,cmap='gray',vmin=0,vmax=0.005)
    for (x0, y0), color in zip(coords, colors):
        ax.scatter(y0,x0,s=marker_size,color=color,alpha=0.5, edgecolors='none')
    norm = plt.Normalize(vmin=0,vmax=1)
    sm = plt.cm.ScalarMappable(cmap=colormap,norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=ax)
    cbar.set_label(r'$g^{(2)}(0)$')
    plt.show()
    
