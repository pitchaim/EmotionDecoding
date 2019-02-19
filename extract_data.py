import mne
import numpy as np
import os
import sys
import scipy
from scipy import signal
import pickle as pkl

SR = 1024
PRES_TIME = 2.5
IMAGE_START = SR*3

#get list of bdf objects
datadir = sys.argv[1]
classfile = sys.argv[2]

print('Using data directory: %s' % datadir)
print('Class file: %s' % classfile)

data_files = []
filenames = []
not_read = []
markers = [] #list of lists of event onset markers

print('Extracting data from BDFs')
for f in os.listdir(datadir):
    if 'bdf' in f and not 'mrk' in f:
        keep = True
        try:
            data_files.append(mne.io.read_raw_edf(os.path.join(datadir,f)))
        except UnicodeDecodeError:
            not_read.append(f)
            keep = False
            pass
        if keep:
            filenames.append(f)

#get raw data out of bdfs
data_arrs = []
for f in data_files:
    data_arrs.append(f.get_data())

print('Number of data arrays: %d' % len(data_arrs))

#read in markers
print('Reading event markers')
for f in os.listdir(datadir):
    if 'mrk' in f and not any([f[:-4] in i for i in not_read]):
        mark = []
        fp = open(os.path.join(datadir,f), 'rb')
        lines = fp.readlines()
        for l in range(1,len(lines)):
            parts = lines[l].split()
            if '254' in str(parts[-1]) and '255' in str(lines[l-1].split()[-1]):
                mark.append(int(parts[0].decode("utf-8")))
        markers.append(mark)

print('Number of event markers: %s' % np.shape(markers))

#get classes, save dict with one-hots for lookup
print('Reading in class labels')
classes = [[], [], []]
cf = open(classfile, 'rb')
cl = cf.readlines()
for cc in cl:
    cp = cc.split()
    classes[0].append(cp[0].decode("utf-8"))
    classes[1].append(cp[1].decode("utf-8"))
    classes[2].append(cp[2].decode("utf-8"))
classdict = {}
classdict['Pos'] = [1,0,0]
classdict['Neg'] = [0,1,0]
classdict['Calm'] = [0,0,1]

#slice up data arrays into event-based windows
master_X_raw = []
master_Y = []
#loop over each separate session datafile
print('Slicing data by event windows')
for s in range(len(data_arrs)):
    print('Session %d' % s)
    session = 0
    if 'SES2' in filenames[s]:
        session = 1
    elif 'SES3' in filenames[s]:
        session = 2
    m = markers[s]
    #loop over block start points (30 blocks)
    for time in range(len(m)):
        print('Block %d' % time)
        start = m[time] + IMAGE_START
        print('Got start time %d' % start)
        for im in range(5):
            print('Image pres %d' % im)
            st = start + int(im*SR*PRES_TIME)
            end = start + int(SR*PRES_TIME*(im+1))
            data_window = data_arrs[s][:,st:end]
            print('Checking data window slice:')
            print(np.shape(data_window))
            if np.shape(data_window)[1] > 0:
                print('Nonzero array - added')
                master_X_raw.append(data_window)
                cl = classes[session][time]
                class_vec = classdict[cl]
                master_Y.append(class_vec)

#perform Welch-method spectral power analysis on event windows
master_X = []
print('Performing power spectrum analysis')
for ev in master_X_raw:
    #find data center, get rid of any outlier channels
    print('Event-windowed array shape:')
    print(np.shape(ev))
    ev_av = np.average(ev,axis=0)
    m_av = np.mean(ev_av)
    m_std = np.std(ev_av)
    mark = np.zeros((1,len(ev))).astype('bool')
    for l in range(len(ev)):
        if np.average(ev[l]) > m_av + 3*m_std:
            mark[0][l] = True
    cleaned = [ev[i] for i in range(len(ev)) if not mark[0][i]]
    av_cl = np.average(cleaned,axis=0)
    #ww, Pxx = scipy.signal.welch(av_cl, fs=SR)
    #feat = Pxx/np.average(Pxx)
    master_X.append(av_cl)

print('Number of features extracted: %d' % len(master_X))
print('Number of class vectors extracted: %d' % len(master_Y))
pkl.dump(master_X, open('features.pkl','wb'))
pkl.dump(master_Y, open('classes.pkl','wb'))
