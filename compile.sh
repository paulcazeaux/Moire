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
make ${TARGET} -j 2
# make test -j
# additional run
echo "====================================================================================="
echo "                                     EXECUTION                                       "
echo "====================================================================================="

InputFile=../app/cfg/twisted_blg.in
ExportFile=../output

# mpirun -n 2 ./app/${TARGET} -i ${InputFile}
