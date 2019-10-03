import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run", help="enter the specific run you need to process",type=str)
args = parser.parse_args()

run = str(args.run)

print("starting")

import glob

print("imported glob")

files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/unprocessed/" + run + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

files_in_order = []
for i in a:
    files_in_order.append(files[i])

print("read files list")

from ast import literal_eval

d = {}

for i in range(0,len(files_in_order)):
#practice run on 5 files:
#for i in range(0,5):
    print(files_in_order[i])
    di = open(files_in_order[i])
    di = di.read()
    if di == "}":
        continue
    else:
        di = di + "}"
        di = literal_eval(di)
        ki = list(di.keys())
        j = len(d)
        for k in ki:
            j += 1000000000000000000000000000000123
            di[j] = di.pop(k)
            d.update(di)
        print(str(100*(i/len(files_in_order))))

print("layer and pdg")

print("get pdg")

def pdg_getter(d,i):

    print(str(i))

    pdgCode_i = d.get(i).get('pdgCode')

    return(pdgCode_i)

import multiprocessing as mp

k = d.keys()

print("get pdg")

pool = mp.Pool(mp.cpu_count())

pdgCode = [pool.apply(pdg_getter, args=(d,i)) for i in k]

pool.close()

electron = []

print("electron one-hot encoding")

import numpy as np

for i in pdgCode:
    if np.abs(i)==11:
        electron.append(1)
    else:
        electron.append(0)

print("get layer 0")

def layer0_getter(d,i):

    print(str(i))

    layer0_i = d.get(i).get('layer 0')

    return(layer0_i)


print("get layer 0")

pool = mp.Pool(mp.cpu_count())

layer0 = [pool.apply(layer0_getter, args=(d,i)) for i in k]

pool.close()


def layer1_getter(d,i):

    print(str(i))

    layer1_i = d.get(i).get('layer 1')

    return(layer1_i)


print("get layer 1")

pool = mp.Pool(mp.cpu_count())

layer1 = [pool.apply(layer1_getter, args=(d,i)) for i in k]

pool.close()

def layer2_getter(d,i):

    print(str(i))

    layer2_i = d.get(i).get('layer 2')

    return(layer2_i)


print("get layer 2")

pool = mp.Pool(mp.cpu_count())

layer2 = [pool.apply(layer2_getter, args=(d,i)) for i in k]

pool.close()

def layer3_getter(d,i):

    print(str(i))

    layer3_i = d.get(i).get('layer 3')

    return(layer3_i)


print("get layer 3")

pool = mp.Pool(mp.cpu_count())

layer3 = [pool.apply(layer3_getter, args=(d,i)) for i in k]

pool.close()

def layer4_getter(d,i):

    print(str(i))

    layer4_i = d.get(i).get('layer 4')

    return(layer4_i)


print("get layer 4")

pool = mp.Pool(mp.cpu_count())

layer4 = [pool.apply(layer4_getter, args=(d,i)) for i in k]

pool.close()

def layer5_getter(d,i):

    print(str(i))

    layer5_i = d.get(i).get('layer 5')

    return(layer5_i)


print("get layer 5")

pool = mp.Pool(mp.cpu_count())

layer5 = [pool.apply(layer5_getter, args=(d,i)) for i in k]

pool.close()

print("get x and y in parallel")

print("get x from layer 0")

def x_0_getter(i):
    import numpy as np

    layer0 = i
    if type(layer0)==type(None) or np.array(layer0).shape==(17,0):
        pass
    else:
        x0 = np.array(layer0)
        x0 = np.sum(x0,axis=0)

    if 'x0' in locals():
        return(x0)
#

print("get y from layer 0")

def y_0_getter(electron,i):
    import numpy as np

    layer0 = i
    if type(layer0)==type(None) or np.array(layer0).shape==(17,0):
        pass
    else:
        y0 = np.array(electron)

    if 'y0' in locals():
        return(y0)

pool = mp.Pool(mp.cpu_count())

x0 = [pool.apply(x_0_getter, args=(i)) for i in (layer0)]

pool.close()


pool = mp.Pool(mp.cpu_count())

y0 = [pool.apply(y_0_getter, args=(i,electron)) for i in (layer0)]

pool.close()

pool = mp.Pool(mp.cpu_count())

x1 = [pool.apply(x_0_getter, args=(i)) for i in (layer1)]

pool.close()


pool = mp.Pool(mp.cpu_count())

y1 = [pool.apply(y_0_getter, args=(i,electron)) for i in (layer1)]

pool.close()

pool = mp.Pool(mp.cpu_count())

x2 = [pool.apply(x_0_getter, args=(i)) for i in (layer2)]

pool.close()


pool = mp.Pool(mp.cpu_count())

y2 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer2)]

pool.close()

pool = mp.Pool(mp.cpu_count())

x3 = [pool.apply(x_0_getter, args=(i)) for i in (layer3)]

pool.close()


pool = mp.Pool(mp.cpu_count())

y3 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer3)]

pool.close()

pool = mp.Pool(mp.cpu_count())

x4 = [pool.apply(x_0_getter, args=(i)) for i in (layer4)]

pool.close()


pool = mp.Pool(mp.cpu_count())

y4 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer4)]

pool.close()

pool = mp.Pool(mp.cpu_count())

x5 = [pool.apply(x_0_getter, args=(i)) for i in (layer5)]

pool.close()


pool = mp.Pool(mp.cpu_count())

y5 = [pool.apply(y_0_getter, args=(electron,i)) for i in (layer5)]

pool.close()

#

x = np.concatenate((x0,x1,x2,x3,x4,x5),axis=None)

y = np.concatenate((y0,y1,y2,y3,y4,y5),axis=None)

print("reshape x and y")

import numpy as np

x = np.reshape(x,(len(y),24))
x = x.astype('float32')

mu = np.mean(x)
x /= mu

print("pickling files")

import pickle

with open('/scratch/vljchr004/data/msc-thesis-data/x_' + run + '.pkl', 'wb') as x_file:
  pickle.dump(x, x_file)

with open('/scratch/vljchr004/msc-thesis-data/y_' + run + '.pkl', 'wb') as y_file:
  pickle.dump(y, y_file)


print("done.")
