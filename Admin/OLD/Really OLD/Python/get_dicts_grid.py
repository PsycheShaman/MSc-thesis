#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 11:55:27 2018

@author: gerhard
"""

#in order to get a list of python dictionaries on the grid,
#the following commands were run in a terminal:

'''
@my_pc_terminal > ssh -Y -l *myusername* hep01.phy.uct.ac.za
@hep01_terminal > alienv enter AliPhysics::latest
@hep01_terminal_with_aliphysics_initialized > aliensh
@alien_shell_terminal > find . pythonDict.txt
'''

#since the alien shell is not a bash terminal, it is not possible (or at least not straight-forward) to simply pipe the stdout to a file,
#therefore: the output of this alien command was manually copied and saved to a text file on my local machine, called 'dict_loc.txt'


import os

#here I tested the viability of calling bash commands via python, but automated password submission didn't work

#import subprocess as shell

#test_shell = str(shell.check_output("ssh-pass -p password shh -Y -l gviljoen hep01.phy.uct.ac.za"))
#test_shell = test_shell.replace('b','').replace("'","").replace('\\n','')
#print(test_shell)


#I therefore created a file from which I could copy and paste the contents into an alienshell:

os.chdir("/Users/gerhard/MSc-thesis/")

bash_command_1 = "initialize_aliroot"
bash_command_2 = "alien_token_init cviljoen"
#bash_command_3 = "aliensh"

bash_commands = [bash_command_1,bash_command_2]

with open("dict_loc.txt","r") as f:
    for line in f:
        line = str(line)
        
        new_filename = line.replace('/alice/cern.ch/user/c/cviljoen/myWorkDir/myOutDir/', '').replace('/', '').replace('pythonDict.txt','').replace('\n','')
        new_filename = ''.join(new_filename.split())
        new_filename = new_filename+".txt"
        
        location = "/home/gviljoen/Thesis_New_Data/"+new_filename
        find_me = line
        new_bash = "cp " + find_me + " file:" + location
        new_bash = new_bash.replace('   \n','')
        bash_commands.append(new_bash)
        
with open("bashcommands.txt", "a") as bashcommandfile:
    for com in bash_commands:
        com = str(com+"\n")
        bashcommandfile.write(com)
        
bashcommandfile.close()

#then, it was needed to append a '}' to the end of each file:

for filename in os.listdir("/Users/gerhard/MSc-thesis/SemiFullData"): # filename is a string
    if filename.endswith(".txt"): # notice the indent
        appendFile = open(filename, 'a') # file object, notice 'a' mode
        appendString = "}" # could be done out of the loop if constant
        appendFile.write(appendString)
        appendFile.close()





