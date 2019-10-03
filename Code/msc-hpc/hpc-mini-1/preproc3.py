#for run in ('265332','265334','265335','265336','265338','265339','265342')

#for run in ('265343','265344','265377','265378','265381','265383','265385','265388'):

for run in ('265377','265378'):
    print(run)


    print("==============================================================================================")

    #import argparse
    #
    #parser = argparse.ArgumentParser()
    #parser.add_argument("run", help="enter the specific run you need to process",type=str)
    #args = parser.parse_args()
    #
    #run = str(args.run)

    print("starting........................................................................................")

    import glob

    print("imported glob........................................................................................")

#    run = '000265309'

    files_in_order = glob.glob("/scratch/vljchr004/unprocessed/000" + run + '/**/*.txt', recursive=True)

#    files_in_order = glob.glob("C:/Users/gerhard/Documents/msc-thesis-data/unprocessed/000" + run + '/**/*.txt', recursive=True)

    print("read files list........................................................................................")

    from ast import literal_eval

    def file_reader1(i):
        di = open(i)
        di = di.read()
        if di == "}":
            pass
        else:
            di = di + "}"
            di = literal_eval(di)
            ki = list(di.keys())
            pdgCode = [di.get(k).get('pdgCode') for k in ki]
            return(pdgCode)

    def file_reader2(i,l):
        di = open(i)
        di = di.read()
        if di == "}":
            pass
        else:
            di = di + "}"
            di = literal_eval(di)
            ki = list(di.keys())
            layer = [di.get(k).get(l) for k in ki]
            return(layer)

    def file_reader3(i):
        di = open(i)
        di = di.read()
        if di == "}":
            pass
        else:
            di = di + "}"
            di = literal_eval(di)
            ki = list(di.keys())
            P = [di.get(k).get('P') for k in ki]
            return(P)

    print("pdg........................................................................................")

    pdgCode0 = [file_reader1(i) for i in files_in_order]

    P0 = [file_reader3(i) for i in files_in_order]

    import numpy as np

    print("layer 0........................................................................................")

    layer0 = [file_reader2(i,"layer 0") for i in files_in_order]

    layer0 = np.array([item for sublist in layer0 for item in sublist])

    pdgCode0 = np.array([item for sublist in pdgCode0 for item in sublist])

    P0 = np.array([item for sublist in P0 for item in sublist])

    empties = np.where([np.array(i).shape!=(17,24) for i in layer0])

    layer0 = np.delete(layer0, empties)

    layer0 = np.stack(layer0)

    pdgCode0 = np.delete(pdgCode0, empties)

    P0 = np.delete(P0, empties)



    print("layer 1........................................................................................")

    layer1 = [file_reader2(i,"layer 1") for i in files_in_order]

    pdgCode1 = [file_reader1(i) for i in files_in_order]

    P1 = [file_reader3(i) for i in files_in_order]

    layer1 = np.array([item for sublist in layer1 for item in sublist if sublist is not None])

    pdgCode1 = np.array([item for sublist in pdgCode1 for item in sublist])

    P1 = np.array([item for sublist in P1 for item in sublist])

    empties = np.where([np.array(i).shape!=(17,24) for i in layer1])

    layer1 = np.delete(layer1, empties)

    layer1 = np.stack(layer1)

    pdgCode1 = np.delete(pdgCode1, empties)
    P1 = np.delete(P1, empties)


    print("layer 2........................................................................................")

    layer2 = [file_reader2(i,"layer 2") for i in files_in_order]

    pdgCode2 = [file_reader1(i) for i in files_in_order]
    P2 = [file_reader3(i) for i in files_in_order]

    layer2 = np.array([item for sublist in layer2 for item in sublist])

    pdgCode2 = np.array([item for sublist in pdgCode2 for item in sublist])
    P2 = np.array([item for sublist in P2 for item in sublist])

    empties = np.where([np.array(i).shape!=(17,24) for i in layer2])

    layer2 = np.delete(layer2, empties)

    layer2 = np.stack(layer2)

    pdgCode2 = np.delete(pdgCode2, empties)
    P2 = np.delete(P2, empties)


    print("layer 3........................................................................................")

    layer3 = [file_reader2(i,"layer 3") for i in files_in_order]

    pdgCode3 = [file_reader1(i) for i in files_in_order]
    P3 = [file_reader3(i) for i in files_in_order]

    layer3 = np.array([item for sublist in layer3 for item in sublist])

    pdgCode3 = np.array([item for sublist in pdgCode3 for item in sublist])
    P3 = np.array([item for sublist in P3 for item in sublist])

    empties = np.where([np.array(i).shape!=(17,24) for i in layer3])

    layer3 = np.delete(layer3, empties)

    layer3 = np.stack(layer3)

    pdgCode3 = np.delete(pdgCode3, empties)
    P3 = np.delete(P3, empties)


    print("layer 4........................................................................................")

    layer4 = [file_reader2(i,"layer 4") for i in files_in_order]

    pdgCode4 = [file_reader1(i) for i in files_in_order]
    P4 = [file_reader3(i) for i in files_in_order]

    layer4 = np.array([item for sublist in layer4 for item in sublist])

    pdgCode4 = np.array([item for sublist in pdgCode4 for item in sublist])
    P4 = np.array([item for sublist in P4 for item in sublist])

    empties = np.where([np.array(i).shape!=(17,24) for i in layer4])

    layer4 = np.delete(layer4, empties)

    layer4 = np.stack(layer4)

    pdgCode4 = np.delete(pdgCode4, empties)
    P4 = np.delete(P4, empties)


    print("layer 5........................................................................................")

    layer5 = [file_reader2(i,"layer 5") for i in files_in_order]

    pdgCode5 = [file_reader1(i) for i in files_in_order]
    P5 = [file_reader3(i) for i in files_in_order]

    layer5 = np.array([item for sublist in layer5 for item in sublist])

    pdgCode5 = np.array([item for sublist in pdgCode5 for item in sublist])
    P5 = np.array([item for sublist in P5 for item in sublist])

    empties = np.where([np.array(i).shape!=(17,24) for i in layer5])

    layer5 = np.delete(layer5, empties)

    layer5 = np.stack(layer5)

    pdgCode5 = np.delete(pdgCode5, empties)
    P5 = np.delete(P5, empties)

    print("mapped out files to useful elements....................................................................")

    print("concatenate pdgs and layers....................................................................")

    pdgCode = np.concatenate([pdgCode0,pdgCode1,pdgCode2,pdgCode3,pdgCode4,pdgCode5]).ravel()
    P = np.concatenate([P0,P1,P2,P3,P4,P5]).ravel()

    x = np.vstack([layer0,layer1,layer2,layer3,layer4,layer5])

    def pdg_code_to_elec(i):
        if np.abs(i)==11:
            return(1)
        else:
            return(0)

    y = np.array([pdg_code_to_elec(i) for i in pdgCode])

    P_1_8_2_2 = np.where(np.logical_and(P>=1.8, P<=2.2))

    P = P[P_1_8_2_2]
    y = y[P_1_8_2_2]
    x = x[P_1_8_2_2,:,:]



    import pickle

    #with open('/scratch/vljchr004/data/msc-thesis-data/ff/x_' + run + '.pkl', 'wb') as x_file:
    #  pickle.dump(x, x_file)

    with open('/scratch/vljchr004/1_8_to_2_2_GeV/P_' + run + '.pkl', 'wb') as P_file:
      pickle.dump(P, P_file)

    with open('/scratch/vljchr004/1_8_to_2_2_GeV/x_' + run + '.pkl', 'wb') as x_file:
      pickle.dump(x, x_file)

    with open('/scratch/vljchr004/1_8_to_2_2_GeV/y_' + run + '.pkl', 'wb') as y_file:
      pickle.dump(y, y_file)


print("done.........................................................................................")

print("==============================================================================================")
