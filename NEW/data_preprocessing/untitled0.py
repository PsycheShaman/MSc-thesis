# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:06:25 2019

@author: gerhard
"""

from ast import literal_eval
import json

import glob

files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\hijing-sim/**/pythonDict.txt", recursive=True)
 
j=0   
for i in files[120:]:
    j=j+1
    d = open(i)
    d = d.read()
    #d = d + "}"
    d = literal_eval(d)
    jayson = json.dumps(d,indent=4,sort_keys=True)
    name1="C:\\Users\\gerhard\\Documents\\msc-thesis-data\\hijing-sim\\" + str(j)
    name2=".json"
    name=name1+name2
    outfile = open(name,"w")
    outfile.write(jayson)
    print(name)