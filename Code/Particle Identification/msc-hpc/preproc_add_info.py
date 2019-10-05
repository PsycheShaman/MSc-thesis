print("==============================================================================================")

#import argparse
#
#parser = argparse.ArgumentParser()
#parser.add_argument("run", help="enter the specific run you need to process",type=str)
#args = parser.parse_args()
#
#run = str(args.run)

print("starting........................................................................................")

import glob

print("imported glob........................................................................................")

#run = '000265383'

run = '000265309'

#files_in_order = glob.glob("/scratch/vljchr004/data/msc-thesis-data/unprocessed/" + run + '/**/*.txt', recursive=True)

files_in_order = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/unprocessed/" + run + '/**/*.txt', recursive=True)

#a = list(range(1,len(files_in_order)-1))
#
#files_in_order = [files_in_order[i] for i in a]

print("read files list........................................................................................")

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
        P = [di.get(k).get('P') for k in ki]
        return(P)
        
def file_reader2(i,l):
    di = open(i)
    print(i)
    di = di.read()
    if di == "}":
        pass
    else:
        di = di + "}"
        di = literal_eval(di)
        ki = list(di.keys())
        layer = [di.get(k).get(l) for k in ki]
        return(layer)


import numpy as np

print("pdg........................................................................................")
        
P0 = [file_reader1(i) for i in files_in_order]


print("layer 0........................................................................................")

layer0 = [file_reader2(i,"layer 0") for i in files_in_order]

layer0 = np.array([item for sublist in layer0 for item in sublist])

P0 = np.array([item for sublist in P0 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer0])

layer0 = np.delete(layer0, empties)

layer0 = np.stack(layer0)

P0 = np.delete(P0, empties)

print("layer 1........................................................................................")

layer1 = [file_reader2(i,"layer 1") for i in files_in_order]

P1 = [file_reader1(i) for i in files_in_order]

layer1 = np.array([item for sublist in layer1 for item in sublist])

P1 = np.array([item for sublist in P1 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer1])

layer1 = np.delete(layer1, empties)

layer1 = np.stack(layer1)

P1 = np.delete(P1, empties)


print("layer 2........................................................................................")

layer2 = [file_reader2(i,"layer 2") for i in files_in_order]

P2 = [file_reader1(i) for i in files_in_order]

layer2 = np.array([item for sublist in layer2 for item in sublist])

P2 = np.array([item for sublist in P2 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer2])

layer2 = np.delete(layer2, empties)

layer2 = np.stack(layer2)

P2 = np.delete(P2, empties)


print("layer 3........................................................................................")

layer3 = [file_reader2(i,"layer 3") for i in files_in_order]

P3 = [file_reader1(i) for i in files_in_order]

layer3 = np.array([item for sublist in layer3 for item in sublist])

P3 = np.array([item for sublist in P3 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer3])

layer3 = np.delete(layer3, empties)

layer3 = np.stack(layer3)

P3 = np.delete(P3, empties)


print("layer 4........................................................................................")

layer4 = [file_reader2(i,"layer 4") for i in files_in_order]

P4 = [file_reader1(i) for i in files_in_order]

layer4 = np.array([item for sublist in layer4 for item in sublist])

P4 = np.array([item for sublist in P4 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer4])

layer4 = np.delete(layer4, empties)

layer4 = np.stack(layer4)

P4 = np.delete(P4, empties)


print("layer 5........................................................................................")

layer5 = [file_reader2(i,"layer 5") for i in files_in_order]

P5 = [file_reader1(i) for i in files_in_order]

layer5 = np.array([item for sublist in layer5 for item in sublist])

P5 = np.array([item for sublist in P5 for item in sublist])

empties = np.where([np.array(i).shape!=(17,24) for i in layer5])

layer5 = np.delete(layer5, empties)

layer5 = np.stack(layer5)

P5 = np.delete(P5, empties)

print("mapped out files to useful elements....................................................................")

print("concatenate pdgs and layers....................................................................")

P = np.concatenate([P0,P1,P2,P3,P4,P5]).ravel()

#x = np.vstack([layer0,layer1,layer2,layer3,layer4,layer5])
#
#nz = np.array([np.count_nonzero(i) for i in x])
#
#zeros = np.where(nz==0)
#
#P = np.delete(P,zeros)
  
#np.savetxt('/scratch/vljchr004/data/msc-thesis-data/cnn/P_' + run + '.csv',P,delimiter=", ")

np.savetxt('C:/Users/gerhard/Documents/msc-thesis-data/cnn/P_' + run + '.csv',P,delimiter=", ")

print("done.........................................................................................")

print("==============================================================================================")











