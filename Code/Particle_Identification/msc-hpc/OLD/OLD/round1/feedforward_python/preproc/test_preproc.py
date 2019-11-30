import glob

files = glob.glob('C:\\Users\\gerhard\\Documents\\msc-thesis-data-master\\msc-thesis-data-master\\unprocessed\\000265342' + '\\**\\*.txt', recursive=True)

a = list(range(1,len(files)-1))


files_in_order = [files[i] for i in a]

from ast import literal_eval

def file_reader1(i):
    di = open(i)
    di = di.read()
    if di == "}":
        pass
    else:
        di = di + "}"
        di = literal_eval(di)
        ki = list(di.keys())
        pdgCode = [di.get(k).get('pdgCode') for k in ki]
        return(pdgCode)

def file_reader2(i,l):
    di = open(i)
    di = di.read()
    if di == "}":
        pass
    else:
        di = di + "}"
        di = literal_eval(di)
        ki = list(di.keys())
        layer = [di.get(k).get(l) for k in ki]
        return(layer)
        
pdgCode0 = [file_reader1(i) for i in files_in_order]

layer0 = [file_reader2(i,"layer 0") for i in files_in_order]

import numpy as np

layer0 = np.array([item for sublist in layer0 for item in sublist])

pdgCode0 = np.array([item for sublist in pdgCode0 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer0])

print(layer0.shape)
print(pdgCode0.shape)

layer0 = np.delete(layer0, empties)

layer0 = np.stack(layer0)

pdgCode0 = np.delete(pdgCode0, empties)

print(layer0.shape)
print(pdgCode0.shape)

x = np.vstack([np.array([np.sum(i,axis=0) for i in layer0]),\
               np.array([np.sum(i,axis=0) for i in layer0])])

print("layer 1........................................................................................")

layer1 = [file_reader2(i,"layer 1") for i in files_in_order]

pdgCode1 = [file_reader1(i) for i in files_in_order]

pdgCode1 = np.concatenate(pdgCode1).ravel()

layer1 = np.concatenate(layer1,axis=None)

empties = np.where([np.array(i).shape!=(17,24) for i in layer1])

layer1 = np.delete(layer1, empties)

layer1 = np.stack(layer1)

pdgCode1 = np.delete(pdgCode1, empties)




print("layer 2........................................................................................")

layer2 = [file_reader2(i,"layer 2") for i in files_in_order]

pdgCode2 = [file_reader1(i) for i in files_in_order]

pdgCode2 = np.concatenate(pdgCode2).ravel()

layer2 = np.concatenate(layer2,axis=None)

empties = np.where([np.array(i).shape!=(17,24) for i in layer2])

layer2 = np.delete(layer2, empties)

layer2 = np.stack(layer2)

pdgCode2 = np.delete(pdgCode2, empties)


print("layer 3........................................................................................")

layer3 = [file_reader2(i,"layer 3") for i in files_in_order]

pdgCode3 = [file_reader1(i) for i in files_in_order]

pdgCode3 = np.concatenate(pdgCode3).ravel()

layer3 = np.concatenate(layer3,axis=None)

empties = np.where([np.array(i).shape!=(17,24) for i in layer3])

layer3 = np.delete(layer3, empties)

layer3 = np.stack(layer3)

pdgCode3 = np.delete(pdgCode3, empties)


print("layer 4........................................................................................")

layer4 = [file_reader2(i,"layer 4") for i in files_in_order]

pdgCode4 = [file_reader1(i) for i in files_in_order]

pdgCode4 = np.concatenate(pdgCode4).ravel()

layer4 = np.concatenate(layer4,axis=None)

empties = np.where([np.array(i).shape!=(17,24) for i in layer4])

layer4 = np.delete(layer4, empties)

layer4 = np.stack(layer4)

pdgCode4 = np.delete(pdgCode4, empties)


print("layer 5........................................................................................")

layer5 = [file_reader2(i,"layer 5") for i in files_in_order]

pdgCode5 = [file_reader1(i) for i in files_in_order]

pdgCode5 = np.concatenate(pdgCode5).ravel()

layer5 = np.concatenate(layer5,axis=None)

empties = np.where([np.array(i).shape!=(17,24) for i in layer5])

layer5 = np.delete(layer5, empties)

layer5 = np.stack(layer5)

pdgCode5 = np.delete(pdgCode5, empties)

print("mapped out files to useful elements....................................................................")

print("concatenate pdgs and layers....................................................................")

pdgCode = np.concatenate([pdgCode0,pdgCode1,pdgCode2,pdgCode3,pdgCode4,pdgCode5]).ravel()

x = np.vstack([layer0,layer1,layer2,layer3,layer4,layer5])

np.array([np.sum(i,axis=0) for i in layer0])

def pdg_code_to_elec(i):
    if np.abs(i)==11:
        return(1)
    else:
        return(0)
        
y = np.array([pdg_code_to_elec(i) for i in pdgCode])

print("mapped out electrons....................................................................")

#print(y)

print("y.shape....................................................................")

print(y.shape)

print("x.shape....................................................................")

print(x.shape)

mu = np.mean(x)
#x /= mu

x = np.true_divide(x,mu)