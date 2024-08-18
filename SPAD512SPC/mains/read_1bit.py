from SPAD512SPC.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240813/data/intensity_images/',
'prefix': ''
}

base_prefix = '240813_SPAD-QD-500kHz-1k-20ns-1bit-'
base_path = '/research2/shared/cwseitz/Data/SPAD/240813/data/intensity_images/'

acqs = [
'acq00000',
'acq00001',
'acq00002',
'acq00003',
'acq00004',
'acq00005',
'acq00006',
'acq00007',
'acq00008',
'acq00009',
'acq00010'
]

for n,acq in enumerate(acqs):
    acqn = str(int(acqs[n][4:]))
    path = base_path + acq + '/'
    prefix = base_prefix + acqn
    config['path'] = path
    config['prefix']  = prefix
    reader = IntensityReader(config)
    stack = reader.stack_1bit(nframes=1000)
    summed = np.sum(stack,axis=0)
    summed = summed.astype(np.uint8)
    imsave(config['savepath']+base_prefix+acqn+'-sum.tif',summed)
    del stack
