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
make ${TARGET}
# make


# additional run
echo "====================================================================================="
echo "                                     EXECUTION                                       "
echo "====================================================================================="

# InputFile=../app/cfg/twisted_blg.in
InputFile=../app/cfg/1d_toymodel.in
ExportFile=../output
rm ${ExportFile}.out
mpirun -n 8 ./app/${TARGET} -i ${InputFile} -draw_pause -1 > ${ExportFile}.out
# open ${ExportFile}.out