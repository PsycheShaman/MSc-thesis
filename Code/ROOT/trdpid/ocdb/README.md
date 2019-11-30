
OCDB Access for PID Studies
===========================

The signal shape in the TRD depends on the propagation of the
ionization electrons in the detector gas, which can be parametrized
with a gain factor, the drift velocity and the Lorentz angle.

These three parameters in turn depend on environmental conditions
(temperature, pressure, gas composition) and detector setting (drift
and anode voltage).

All of these parameters are stored in the OCDB. The scripts in this
directory help in querying the OCDB to retrieve the parameters for a
given run or set of runs.

Scripts
=======

The script **dump.C** calls another script from AliRoot that dumps the
information for all the runs listed in a text file (default:
pPb-2016.txt) into a ROOT TTree. The called script, `DumpOCDBtoTree.`,
is a good reference how to access the data in the OCDB. An example
runlist `pPb-2016.txt` has been included.

The script **query.C** is meant to query the OCDB for a single run and
fill a data structure with the most relevant data. This script is a
starting point, e.g. to dump this information in a Python-readable
formal like YAML.

Usage
=====

The scripts in this directory run in AliRoot. You will therefore need
to setup AliRoot, e.g. with
```
/cvmfs/alice.cern.ch/bin/alienv enter VO_ALICE@AliPhysics::vAN-20190603-1
```
if you have CVMFS installed.

Both scripts have default options for all arguments. In the simplest case, you can just run them with
```
aliroot dump.C
aliroot query.C
```
You probably want to adjust run numbers, run list or similar.

