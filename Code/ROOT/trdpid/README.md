# Introduction

This repository is meant as a collection and working area for stuff
going into PID studies for the ALICE TRD. It is mainly a collection of
my (Tom Dietel) personal work, but I might pull in other people's work
as I see fit.

Some code has been (and will be) moved from here to AliRoot and
AliPhysics, to be used in production.

In the following you will find some hints what some parts of the
software are supposed to do. 


# Reconstruction [rec]

Reconstruct data (or simulated) files. This is e.g. necessary if you
want to prepare some raw data files to play with raw data filtering.


## full reconstruction: 'rec/rec.C'

   This runs the full reconstruction. This script was taken from
   alien:///alice/cern.ch/user/a/alidaq/LHC13f/rec.C. The macro will
   produce write TRD digits as well as the standard output. The OCDB
   will be taken from CVMFS.

   The easiest way to run reconstruction is to create a new directory,
   create a symlink *raw.root* in this directory with `ln -sf
   /path/to/your/raw/file.root raw.root` and run the reconstruction
   from there with `aliroot -b -q -l path/to/rec.C`

## extraction of TRD digits from a raw file: rec/rec_digits.C

   Reads a raw data file and extracts the TRD digits into a file
   TRD.Digits.root. Can be used to re-run the MCM emulator and for
   detailed PID and other studies.

# Filtering [filter]

This part deals with raw data filtering for TRD PID studies. The
purpose of this is to get a reasonable sample of raw data from
high-pT electrons, that can be used to optimize the PID.

A large part of this is now available in AliPhysics, and this is
mainly an example how to call these parts. The main work is done in
the class AliTRDdigitsFilter (in
$ALICE_PHYSICS/PWGPP/TRD/AliTRDdigitsFilter), which is instantiated by 
$ALICE_PHYSICS/PWGPP/TRD/macros/AddTRDdigitsFilter.C).

# Analysis [ana]

This is an example analysis, which at the moment produces a spectrum
of TRD tracks (and all tracks, for comparison), and an ADC spectrum
for the TRD digits.

The analysis is run via the macro *ana.C*, which in turn loads
analysis tasks either from Aliroot/AliPhysics, or from local
directories. If you want to use a local analysis task, it must be
accessible for gROOT->LoadMacro(), i.e. either be in the current
working directory where you run the analysis, or you must make it
available in some other way.

You can run the analysis in the same folder as the reconstruction was
run (with *rec/rec.C*, see above). You can do this my symlinking all
necessary files to the directory where the reconstruction produced its
output: `ln -sf path/to/ana/*.C path/to/ana/*.cxx path/to/ana/*.h
.`. Afterwards you can run the analysis with `aliroot -b -q -l ana.C`.
