import mne
import numpy as np
import os
import sys
from numpy.fft import fft as npfft

#get list of bdf objects
datadir = sys.argv[1]
classfile = sys.argv[2]

os.chdir(targdir)

data_files = []
not_read = []
markers = []

for f in os.listdir('.'):
    if 'bdf' in f and not 'mrk' in f:
        try:
            data_files.append(mne.io.read_raw_edf(f))
        except UnicodeDecodeError:
            not_read.append(f)
            pass

#get raw data out of bdfs
data_arrs = []
for f in data_files:
    data_arrs.append(f.get_data())

#read in markers
for f in os.listdir('.'):
    if 'mrk' in f and not any([f[:-4] in i for i in notlist]):
        fp = open(f, 'rb')
        

#get classes
