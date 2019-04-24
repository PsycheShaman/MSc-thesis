# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:23:36 2019

@author: gerhard
"""

import glob

files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\unprocessed\\000265309" + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

#a.append(0)

files_in_order = []
for i in a:
    files_in_order.append(files[i])
    
from ast import literal_eval

for i in range(0,1):#len(files_in_order)):
            print(files_in_order[i])
            appendFile = open(files_in_order[i], 'a') # file object, notice 'a' mode
            appendFile.close()
            d = open(files_in_order[i])
            d = d.read()
            d = d + "}"
            d = literal_eval(d)
            print(d[1])
            
a = d.keys()

print(a)

d.get(1).keys()

d.get(0).get('P')
