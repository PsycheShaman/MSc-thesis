# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 08:45:58 2019

@author: gerhard
"""

with open("C:/Users/gerhard.22seven/downloads/rosalind_dna(2).txt",'r') as infile:
    x=infile.read()
    
str(x.count('A'))+' '+str(x.count('C'))+' '+str(x.count('G'))+' '+str(x.count('T'))    
    
    
with open("C:/Users/gerhard.22seven/downloads/rosalind_rna.txt",'r') as infile:
    x=infile.read()
    
x.replace('T','U')
    
with open("C:/Users/gerhard.22seven/downloads/rosalind_revc(1).txt",'r') as infile:
    x=infile.read()
    
x = ''.join(reversed(x))

d = {'A':'T','T':'A','C':'G','G':'C'}

x = x.strip()

x2 = ''
for i in x:
    x2 = x2+d[i]

x2

with open("C:/Users/gerhard.22seven/downloads/rosalind_gc.txt",'r') as infile:
    x=infile.read()


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)
        
t = list(find_all(x,'>'))

t2 = t[1:]

for i in range(len(t2)):
    t2[i]=t2[i]-1
    
t2.append(len(x))
d = {}

for i in range(len(t)) :
    seqname = 'seq'+str(i)
    seqval = x[t[i]:t2[i]]
    d.update({seqname:seqval})
    
import re

for i in d:
    print(re.finditer('Rosalind_[0-9]{4}',d[i]))






    
    
    
    
    
    