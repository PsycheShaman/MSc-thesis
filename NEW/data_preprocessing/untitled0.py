# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:06:25 2019

@author: gerhard
"""

from ast import literal_eval
import json

d = open("C:/Users/gerhard/Documents/msc-thesis-data/hijing-sim/pythonDict.txt")
d = d.read()
#d = d + "}"
d = literal_eval(d)
jayson = json.dumps(d,indent=4,sort_keys=True)
name1="C:\\Users\\gerhard\\Documents\\msc-thesis-data\\hijing-sim\\" + str(1)
name2=".json"
name=name1+name2
outfile = open(name,"w")
outfile.write(jayson)
print(name)