#!/bin/zsh

# user parameter
TARGET="Moire"

# cmake parameters
source /opt/intel/mpi/intel64/bin/mpivars.sh
export CC=/opt/intel/mpi/intel64/bin/mpigcc
export CXX=/opt/intel/mpi/intel64/bin/mpigxx

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
make ${TARGET} -j 16
make test
# additional run
echo "====================================================================================="
echo "                                     EXECUTION                                       "
echo "====================================================================================="

InputFile=../app/cfg/twisted_blg.in
ExportFile=../output

# mpirun -n 2 ./app/${TARGET} -i ${InputFile}
