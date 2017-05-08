#!/bin/bash

# user parameter
TARGET="HullSpace"

# cmake parameters
export CC=clang
export CXX=clang++

# build directory
cd /Users/cazeaux/Dropbox/Workplace/Projets_actuels/Minnesota/Calgebras_1D/Cpp/hullspace/build

# automatic makefile generation and compilation
echo "====================================================================================="
echo "                                      CMAKE                                          "
echo "====================================================================================="
cmake ..

echo "====================================================================================="
echo "                                       MAKE                                          " 
echo "====================================================================================="
make ${TARGET}
# make


# additional run
echo "====================================================================================="
echo "                                     EXECUTION                                       "
echo "====================================================================================="

InputFile=../app/cfg/tblg_hstruct.in
ExportFile=../test_output
rm ${ExportFile}.out
mpirun -n 4 ./app/${TARGET} -i ${InputFile} > ${ExportFile}.out
open ${ExportFile}.out