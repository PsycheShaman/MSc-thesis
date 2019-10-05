print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

#x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/x_*.pkl")
#y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/cnn/y_*.pkl")

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

mu = np.mean(x)

sd = np.std(x)

x = x-mu

x=x/sd

#x.shape = (x.shape[0],x.shape[1]*x.shape[2])

electrons = np.where(y==1)

electrons = electrons[0]

pions = np.where(y==0)

pions = pions[0]

pions = pions[0:electrons.shape[0]]

x_1 = x[electrons,:]
x_2 = x[pions,:]

x = np.vstack((x_1,x_2))

y_1 = y[electrons]
y_2 = y[pions]

y = np.concatenate((y_1,y_2),axis=None)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier(silent=False,
                      objective='binary:logistic', 
                      n_estimators=10000)

eval_set = [(x_train, y_train), (x_test, y_test)]
eval_metric = ["auc","error"]
model.fit(x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
	
# make predictions for test data
y_pred = model.predict_proba(x_test)
predictions = [round(value) for value in y_pred[:,1]]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

 	
# save model to file
pickle.dump(model, open("C:/Users/gerhard/Documents/msc-hpc/model12.pkl", "wb"))

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/model12_preds.csv",y_pred,delimiter=", ")
np.savetxt("C:/Users/gerhard/Documents/msc-hpc/model12_y_test.csv",y_test,delimiter=", ")

