print("==============================================================================================")

print("starting........................................................................................")

import glob

import numpy as np

print("imported glob, np........................................................................................")

#x_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/x_*.pkl")
#y_files = glob.glob("/scratch/vljchr004/data/msc-thesis-data/ff/y_*.pkl")

x_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/ff/x_*.pkl")
y_files = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/ff/y_*.pkl")

import pickle

print("loading first x pickle........................................................................................")

with open(x_files[0], 'rb') as x_file0:
    x = pickle.load(x_file0)
    
print("loading first y pickle........................................................................................")

with open(y_files[0], 'rb') as y_file0:
   y = pickle.load(y_file0)
   
print("recursively adding x pickles........................................................................................")

for i in x_files[1:]:
#for i in x_files[1:2]:
    with open(i,'rb') as x_file:
        xi = pickle.load(x_file)
        x = np.concatenate((x,xi),axis=0)
        
print("recursively adding y pickles........................................................................................")
        
for i in y_files[1:]:
#for i in y_files[1:2]:
    with open(i,'rb') as y_file:
        yi = pickle.load(y_file)
        y = np.concatenate((y,yi),axis=None)
        
nz = np.array([np.count_nonzero(i) for i in x])

zeros = np.where(nz==0)

x = np.delete(x,zeros,axis=0)
y = np.delete(y,zeros)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=123456)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=4, 
gamma=10)

eval_set = [(x_train, y_train), (x_test, y_test)]
eval_metric = ["auc","error"]
model.fit(x_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)

#import matplotlib.pyplot as plt
#
## summarize history for accuracy
##plt.plot(model.eval_metric['auc'])
#plt.plot(eval_metric['error'])
#plt.title('Error')
#plt.ylabel('error')
#plt.xlabel('n_trees')
#plt.legend(['train', 'test'], loc='upper left')
#
#plt.show()
##plt.savefig('/home/vljchr004/msc-hpc/feedforward_python/fig/feed_forward_2_history1.png', bbox_inches='tight')
#
#plt.close()
	
# make predictions for test data
y_pred = model.predict_proba(x_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

 	
# save model to file
pickle.dump(model, open("C:/Users/gerhard/Documents/msc-hpc/xgb/xgb0.pkl", "wb"))

np.savetxt("C:/Users/gerhard/Documents/msc-hpc/xgb/xgb0_preds.csv",y_pred,delimiter=", ")
np.savetxt("C:/Users/gerhard/Documents/msc-hpc/xgb/xgb0_y_test.csv",y_test,delimiter=", ")

