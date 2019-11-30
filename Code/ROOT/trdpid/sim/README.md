
Simulation of TRD data
======================

This directory contains simulation code for AliRoot. The code was
taken from a Hons project that ran in 2016 and needs some cleanup and
tuning, but it should be a starting point.

Running the simulation
----------------------

The simulation can be run with the script `runtest.sh`, which
* creates a directory `test`
* symlinks sim.C, rec.C and Config.C into the test directory
* runs sim.C and rec.C

A quick look at Config.C suggests that the simulation has not been
tuned for TRD PID/sim studies. Ideally it should use a box generator
for electrons and pions instead of a mix of all different kinds of
things...

The output should be usable by normal analysis macros - but I did not
try this.


