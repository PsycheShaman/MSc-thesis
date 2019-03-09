# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os

os.chdir("/Users/gerhard/Thesis-data")

import glob

files = glob.glob("/Users/gerhard/Thesis-data" + '/**/*.txt', recursive=True)

a = list(range(1,len(files)-1))

a.append(0)

files_in_order = []
for i in a:
    files_in_order.append(files[i])

with open('/Users/gerhard/Thesis-data/265377', 'w') as outfile:
    for fname in files_in_order:
        with open(fname) as infile:
            outfile.write(infile.read())
            
from ast import literal_eval

import json
            
d = open('/Users/gerhard/Thesis-data/265377')
d = d.read()
d = literal_eval(d)
jayson = json.dumps(d,indent=4,sort_keys=True)
name1="265377"
name2=".json"
name=name1+name2
outfile = open(name,"w")
outfile.write(jayson)
print(name)