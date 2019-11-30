#!/bin/sh

mkdir test2
cd test2
#i=0
#while ["${i}" != 10000]
for i in {1..100000}
do
# set up the test directoru
mkdir -p test${i}
echo "...................Doing sim for loop number $i"
#i = ${i}+1
cd test${i}
ln -sf ../../Config.C Config.C
ln -sf ../../sim.C sim.C
ln -sf ../../rec.C rec.C
ln -sf ../../CreateESDChain.C CreateESDChain.C
ls -la

# clean up
rm -rf *.root *.dat *.log fort* hlt hough raw* recraw/*.root recraw/*.log

# run simulation
aliroot -b -q sim.C      2>&1 | tee sim.log
mv syswatch.log simwatch.log

# run reconstruction
aliroot -b -q rec.C      2>&1 | tee rec.log
mv syswatch.log recwatch.log

#aliroot -b -q ${ALICE_ROOT}/STEER/macros/CheckESD.C 2>&1 | tee check.log

# skip AOD generation
#aliroot -b -q aod.C 2>&1 | tee aod.log

# run the raw reconstruction
#mkdir recraw
#cd recraw
#ln -s ../raw.root
#ln -s ../../CreateESDChain.C
#ln -sf ../../AliAnalysisTaskTom.cxx AliAnalysisTaskTom.cxx
#ln -sf ../../AliAnalysisTaskTom.h AliAnalysisTaskTom.h
#aliroot -b -q ../../rawrec.C 2>&1 | tee rawrec.log
#aliroot -b -q  2>&1 aod.C | tee aod.log
#cd ..
# analyse the data
ln -sf ../../AliAnalysisTaskTom.C
ln -sf ../../AliAnalysisTaskTom.cxx
ln -sf ../../AliAnalysisTaskTom.h
ln -sf ../../AddExtractTask.C
ln -sf ../../AliTRDdigitsExtract.cxx
ln -sf ../../AliTRDdigitsExtract.h
ln -sf ../../AliTRDdigitsTask.cxx
ln -sf ../../AliTRDdigitsTask.h
ln -sf ../../CreateESDChain.C
aliroot -b -q ../../ana.C 2>&1 | tee ana.log
rm -rf *.root *.dat *.log fort* hlt hough raw* recraw/*.root recraw/*.log
cd ..
done
