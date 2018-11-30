# -*- coding: utf-8 -*-
"""
This script extracts Python Dictionaries with Tracklet information 
"""

import os
from ast import literal_eval
import json

os.chdir("/Users/gerhard/MSc-thesis/data")

dat_files = os.listdir("/Users/gerhard/MSc-thesis/data")
dat_files.pop(16)

for i in range(0,len(dat_files)):
        print(dat_files[i])
        d = open(dat_files[i])
        d = d.read()
        d = literal_eval(d)
        jayson = json.dumps(d,indent=4,sort_keys=True)
        name1="dat"
        name2=".json"
        name=name1+str(i)+name2
        outfile = open(name,"w")
        outfile.write(jayson)