# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:14:22 2019

@author: gerhard
"""

run = '265309'

import glob

files_in_order = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/unprocessed/000" + run + '/**/*.txt', recursive=True)

i = files_in_order[0]

i = open(i)
i = i.read()
i = i+"}"

from ast import literal_eval

i = literal_eval(i)

k = list(i.keys())

len(k)

i.get(1).keys()

RunNumber
Event
V0TrackID
track
pdgCode
nSigmaElectron
nSigmaPion
PT
dEdX
P
Eta
Theta
Phi
det0
row0
col0
layer0
det1
row1
col1
layer1
det2
row2
col2
layer2
det3
row3
col3
layer3
det4
row4
col4
layer4
det5
row5
col5
layer5

for(i in k)


