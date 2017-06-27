#!/bin/bash

# user parameter
TARGET="Moire"

# cmake parameters
# export CC=/usr/lib/llvm/4/bin/clang
# export CXX=/usr/lib/llvm/4/bin/clang++
export CC=gcc-7
export CXX=g++-7

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
make ${TARGET} -j 44


# additional run
echo "====================================================================================="
echo "                                     EXECUTION                                       "
echo "====================================================================================="

InputFile=../app/cfg/1d_toymodel.in
ExportFile=../output

rm ${ExportFile}.out
mpirun -n 2 ./app/${TARGET} -i ${InputFile} #  > ${ExportFile}.out
