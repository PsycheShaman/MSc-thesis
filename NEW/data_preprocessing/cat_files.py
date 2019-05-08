# -*- coding: utf-8 -*-

import glob

for run_number in [265338,265339,265342,265343,265344,265377,265378,265381,265383,265385,265388,265419,265420,265425,265426,265499]:
    files = glob.glob("C:\\Users\\gerhard\\Documents\\msc-thesis-data\\unprocessed\\000"+ str(run_number) + '/**/*.txt', recursive=True)
    
    a = list(range(1,len(files)-1))
    
    files_in_order = []
    for i in a:
        files_in_order.append(files[i])
        
    from ast import literal_eval
    import json
                
    for i in range(0,len(files_in_order)):
                print(files_in_order[i])
                d = open(files_in_order[i])
                d = d.read()
                d = d + "}"
                d = literal_eval(d)
                jayson = json.dumps(d,indent=4,sort_keys=True)
                name1="C:\\Users\\gerhard\\Documents\\msc-thesis-data\\processed\\000" + str(run_number) + "\\"
                name2=".json"
                name=name1+str(i)+name2
                outfile = open(name,"w")
                outfile.write(jayson)
                print(name)