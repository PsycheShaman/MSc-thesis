#!/bin/sh

echo "========================================================="
echo "| Environment"
echo "========================================================="
printenv

echo "========================================================="
echo "| Working directory"
echo "========================================================="
pwd
ls -l


echo "========================================================="
echo "| Run reconstruction"
echo "========================================================="
echo -n "start reconstruction: "
date
aliroot -b -q -l rec.C

echo "========================================================="
echo "| Run digits filter"
echo "========================================================="
echo -n "start filtering: "
date
aliroot -b -q -l filter.C

echo -n "finished: "
date
