# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

os.chdir("/Users/gerhard/Thesis-data")

import glob

files = glob.glob("/Users/gerhard/Thesis-data/JS/000265377" + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

#a.append(0)

files_in_order = []
for i in a:
    files_in_order.append(files[i])

#with open('/Users/gerhard/Thesis-data/265377', 'w') as outfile:
#    for fname in files_in_order:
#        with open(fname) as infile:
#            outfile.write(infile.read())
#outfile.close()
    
from ast import literal_eval
import json
            
for i in range(0,len(files_in_order)):
            print(files_in_order[i])
            appendFile = open(files_in_order[i], 'a') # file object, notice 'a' mode
            appendString = "}"
            appendFile.write(appendString)
            appendFile.close()
            d = open(files_in_order[i])
            d = d.read()
            d = literal_eval(d)
            jayson = json.dumps(d,indent=4,sort_keys=True)
            name1="/Users/gerhard/Thesis-data/JS/000265377/js"
            name2=".json"
            name=name1+str(i)+name2
            outfile = open(name,"w")
            outfile.write(jayson)
            print(name)