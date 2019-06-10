#!/bin/bash

# user parameter
TARGET="Moire"

# cmake parameters
export CC=mpicc
export CXX=mpic++

# build directory
cd ./build

# automatic makefile generation and compilation
echo "====================================================================================="
echo "                                      CMAKE                                          "
echo "====================================================================================="
cmake ..

echo "====================================================================================="
echo "                                       MAKE                                          " 
echo "====================================================================================="
make ${TARGET} -j 8
make test
# additional run
echo "====================================================================================="
echo "                                     EXECUTION                                       "
echo "====================================================================================="

InputFile=../app/cfg/1d_toymodel.in
ExportFile=../output

mpirun -n 8 --use-hwthread-cpus ./app/${TARGET} -i ${InputFile}