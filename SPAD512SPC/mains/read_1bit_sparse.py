from SPAD512SPC.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images_2/',
'prefix': ''
}

base_prefix = '240702_SPAD-QD-500kHz-100k-1us-1bit-'
base_path = '/research2/shared/cwseitz/Data/SPAD/240702/data/intensity_images_2/'

acqs = [
'acq00003',
'acq00004',
'acq00005',
'acq00006',
'acq00007'
]

for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    acqn = str(int(acqs[n][4:]))
    prefix = base_prefix + acqn
    config['path'] = path
    config['prefix']  = prefix
    reader = IntensityReaderSparse(config)
    summed = reader.read_1bit_summed()
    imsave(config['savepath']+base_prefix+acqn+'-sum.tif',summed)
    del summed
    
    
