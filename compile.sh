#!/bin/bash

# user parameter
TARGET="Moire"

# cmake parameters
export CC=clang
export CXX=clang++

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
# make


# additional run
echo "====================================================================================="
echo "                                     EXECUTION                                       "
echo "====================================================================================="

# InputFile=../app/cfg/twisted_blg.in
InputFile=../app/cfg/1d_toymodel.in
ExportFile=../output
rm ${ExportFile}.out
mpirun -n 44 ./app/${TARGET} -i ${InputFile}  > ${ExportFile}.out
# open ${ExportFile}.out
