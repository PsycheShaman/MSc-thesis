import numpy as np
import pandas as pd
import os

def subdir_(directory):
    fileNames = []
    for r, d, f in os.walk(directory):
        for file in f:
            if 'pythonDict.txt' in file:
                fileNames.append(os.path.join(r, file))
    fileNames.sort()
    return fileNames

def process_tracklet_(raw_data, raw_info, min_tracklet=1.0, min_adcvalue=10.0, min_momentum=0.0, max_momentum=100.0):
    """
    raw_info[0] = label
    raw_info[1] = nsigmae
    raw_info[2] = nsigmap
    raw_info[3] = PT
    raw_info[4] = dEdX
    raw_info[5] = P
    raw_info[6] = eta
    raw_info[7] = theta
    raw_info[8] = phi
    raw_info[9] = run_number
    raw_info[10] = event
    raw_info[11] = trackid
    raw_info[12] = trackval
    raw_info[13] = num_tracklets
    raw_info[14:20] = dets
    raw_info[20:26] = rows
    raw_info[26:32] = cols
    raw_info[32:38] = present_map
    """
    mask_tracklet = raw_info[:,13] > min_tracklet                          #Discriminate tracks based on no. of tracklets
    mask_momentum = (raw_info[:,5] > min_momentum) & (raw_info[:,5] < max_momentum) #Select momentum range
    mask_total = np.logical_and(mask_tracklet,mask_momentum)

    raw_info = raw_info[mask_total]
    raw_data = raw_data[mask_total]

    numtracklets = raw_info[:,13].astype(int)                                  #Tracklets per track
    infoset = np.zeros((numtracklets.sum(), raw_info[:,:13].shape[1]))
    k = 0
    for i in range(len(numtracklets)):
        t = i
        for j in range(numtracklets[i]):
            infoset[k] = raw_info[i,:13]
            k += 1

    present = raw_info[:,-6:].flatten('C').astype('bool')
    dataset = raw_data.reshape(-1,17,24,1)[present]  #NHWC array
    coordinates = raw_info[:,14:32].reshape(-1,3,6).swapaxes(1,2).reshape(-1,3)[present].astype('int') #[tracklet, [det, row, col]]
    mask_adcvalue = dataset.sum(axis=(1,2,3)) > min_adcvalue              #Sum of ADC per tracklet

    return dataset[mask_adcvalue], infoset[mask_adcvalue], coordinates[mask_adcvalue]

def process_track_(raw_data, raw_info, num_tracklet=6.0, min_adcvalue=10.0, min_momentum=0.0, max_momentum=100.0):
    mask_tracklet = raw_info[:,13] == num_tracklet                         #Discriminate tracks based on no. of tracklets
    mask_adcvalue = raw_data.sum(axis=(1,2,3)) > min_adcvalue              #Sum of ADC per tracklet
    mask_momentum = (raw_info[:,5] > min_momentum) & (raw_info[:,5] < max_momentum) #Select momentum range

    infoset = raw_info[mask_tracklet & mask_adcvalue & mask_momentum][:,:13]
    dataset = raw_data[mask_tracklet & mask_adcvalue & mask_momentum].swapaxes(1,2).swapaxes(2,3)
    coordinates = raw_info[mask_tracklet & mask_adcvalue & mask_momentum][:,14:32].reshape(-1,3,
        int(num_tracklet)).swapaxes(1,2).astype('int')
    return dataset, infoset, coordinates

def calib_tracklet_(dataset, infoset, coordinates, ocdbdir):
    R = infoset[:,9].astype('int')
    runs = set(R)

    for run in runs:
        print(run)
        gainglob = pd.read_csv(ocdbdir + 'chamber_info_2016_%d.txt'%run, header = None).values[:,3]
        gainlocl = pd.read_csv(ocdbdir + 'local_gains_2016_%d.txt'%run,
            header = None).values.reshape((540, 16,-1))[:,:,2:]           #(detector, row, column)

        gainG = np.ones(dataset.shape)                   #gain for chambers
        gainP = np.ones(dataset.shape)
        mask = np.where(R==run, range(dataset.shape[0]), -1)

        for i, [d, r, c] in enumerate(coordinates):
            if i == mask[i]:
                gainP[i,:,:,0] = np.tile(gainlocl[d, r, c-8:c+9],(24,1)).T
                gainG[i,:,:,0] = np.tile(gainglob[d],(17,24))

        dataset = np.multiply(np.multiply(dataset, gainP),gainG)
    return dataset

def calib_track_(dataset, infoset, coordinates, ocdbdir):
    R = infoset[:,9].astype('int')
    runs = set(R)

    for run in runs:
        print(run)
        gainglob = pd.read_csv(ocdbdir + 'chamber_info_2016_%d.txt'%run, header = None).values[:,3]
        gainlocl = pd.read_csv(ocdbdir + 'local_gains_2016_%d.txt'%run,
            header = None).values.reshape((540, 16,-1))[:,:,2:]           #(detector, row, column)

        gainG = np.ones(dataset.shape)                   #gain for chambers
        gainP = np.ones(dataset.shape)
        mask = np.where(R==run, range(dataset.shape[0]), -1)

        for i in range(coordinates.shape[0]):
            if i == mask[i]:
                for j, [d, r, c] in enumerate(coordinates[i]):
                    gainP[i,:,:,j] = np.tile(gainlocl[d, r, c-8:c+9],(24,1)).T
                    gainG[i,:,:,j] = np.tile(gainglob[d],(17,24))

        dataset = np.multiply(np.multiply(dataset, gainP),gainG)
    return dataset
    
def bin_time_(dataset, bins = 8):
    num = int(dataset.shape[2]/bins)
    shapearr  = np.array(dataset.shape)
    shapearr[2] = bins
    datasetbinned = np.zeros(tuple(shapearr))
    for i in range(num):
        datasetbinned += dataset[:,:,i::num]
    return datasetbinned

def shuffle_(dataset, infoset):
    #   Apply random permutation to given dataset.  #
    perm = np.random.permutation(dataset.shape[0])
    dataset = dataset[perm]
    infoset = infoset[perm]
    return dataset, infoset

def elec_strip_(dataset, infoset):
    targets = infoset[:,0].astype('int')
    dataset = dataset[targets==1]
    infoset = infoset[targets==1]
    return dataset, infoset

def pion_strip_(dataset, infoset):
    targets = infoset[:,0].astype('int')
    dataset = dataset[targets==0]
    infoset = infoset[targets==0]
    return dataset, infoset

def batch_(dataset, targets, batch_size, pos):
    batch_dataset = dataset[(pos-1)*batch_size:pos*batch_size]
    batch_targets = targets[(pos-1)*batch_size:pos*batch_size]
    return batch_dataset, batch_targets

def elec_pion_split_(dataset, infoset):
    targets = infoset[:,0].astype('int')
    elec_dataset = dataset[targets.astype(bool)]
    pion_dataset = dataset[(1-targets).astype(bool)]
    elec_infoset = infoset[targets==1]
    pion_infoset = infoset[targets==0]
    return [elec_dataset, elec_infoset], [pion_dataset, pion_infoset]

def TVT_split_(dataset, infoset, test_split=0.1, valid_split=0.2):
    #   Create training and validation sets   #
    train_split = test_split + valid_split
    N1 = int((1-train_split)*dataset.shape[0])
    N2 = int((1-test_split )*dataset.shape[0])
    train_dataset = dataset[:N1]
    train_infoset = infoset[:N1]
    valid_dataset = dataset[N1:N2]
    valid_infoset = infoset[N1:N2]
    test_dataset  = dataset[N2:]
    test_infoset  = infoset[N2:]
    return [train_dataset, train_infoset], [valid_dataset, valid_infoset], [test_dataset, test_infoset]