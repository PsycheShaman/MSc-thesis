#!/bin/sh

#alien_mkdir /alice/cern.ch/user/t/tdietel/trdpid/
#alien_mkdir /alice/cern.ch/user/t/tdietel/trdpid/ana


for i in AliTRDPIDrawData.cxx AliTRDPIDrawData.h train.C; do 
	 alien_cp $i alien:/alice/cern.ch/user/t/tdietel/trdpid/ana/
done
