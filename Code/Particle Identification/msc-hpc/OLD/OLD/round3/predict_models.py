print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

#x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/x_*.pkl")
#y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/y_*.pkl")

x_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\ff\\x_*.pkl")
y_files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\ff\\y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
    with open(i,'rb') as x_file:
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)

from tensorflow.keras.models import Sequential

print("<-----------------------------predicting models: 1------------------------------------------>")

model = Sequential()

model.load('C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model1_.h5')  # creates a HDF5 file 'my_model.h5'

model.probs = model.predict_proba(x)

model.probs = np.array(model.probs)

model.probs = np.hstack(model.probs,y)

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model1_full_preds.csv", np.array(model.probs), fmt="%s")


print("<-----------------------------predicting models: 2------------------------------------------>")

model = Sequential()

model.load('C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model2_.h5')  # creates a HDF5 file 'my_model.h5'

model.probs = model.predict_proba(x)

model.probs = np.array(model.probs)

model.probs = np.hstack(model.probs,y)

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model2_full_preds.csv", np.array(model.probs), fmt="%s")

print("<-----------------------------predicting models: 3------------------------------------------>")

model = Sequential()

model.load('C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model3_.h5')  # creates a HDF5 file 'my_model.h5'

model.probs = model.predict_proba(x)

model.probs = np.array(model.probs)

model.probs = np.hstack(model.probs,y)

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model3_full_preds.csv", np.array(model.probs), fmt="%s")

print("<-----------------------------predicting models: 4------------------------------------------>")

model = Sequential()

model.load('C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model4_.h5')  # creates a HDF5 file 'my_model.h5'

model.probs = model.predict_proba(x)

model.probs = np.array(model.probs)

model.probs = np.hstack(model.probs,y)

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model4_full_preds.csv", np.array(model.probs), fmt="%s")

print("<-----------------------------predicting models: 5------------------------------------------>")

model = Sequential()

model.load('C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model2_.h5')  # creates a HDF5 file 'my_model.h5'

model.probs = model.predict_proba(x)

model.probs = np.array(model.probs)

model.probs = np.hstack(model.probs,y)

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/round3/feedforward/local/round3_model2_full_preds.csv", np.array(model.probs), fmt="%s")













print("<-----------------------------done------------------------------------------>")

















