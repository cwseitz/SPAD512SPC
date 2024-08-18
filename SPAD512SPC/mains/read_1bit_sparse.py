from SPAD512SPC.utils import *
from skimage.io import imread,imsave
import matplotlib.pyplot as plt

config = {
'path': '',
'savepath': '/research3/shared/cwseitz/Data/SPAD/240817/data/intensity_images/',
'prefix': ''
}

base_prefix = '240817_SPAD-QD-500kHz-100k-50ns-1bit-'
base_path = '/research3/shared/cwseitz/Data/SPAD/240817/data/intensity_images/'

acqs = [
'acq00000',
'acq00001'
]

for n,acq in enumerate(acqs):
    path = base_path + acq + '/'
    acqn = str(int(acqs[n][4:]))
    prefix = base_prefix + acqn
    config['path'] = path
    config['prefix']  = prefix
    reader = IntensityReaderSparse(path)
    summed = reader.read_1bit_summed()
    imsave(config['savepath']+base_prefix+acqn+'-sum.tif',summed)
    del summed
    
    
