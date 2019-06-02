#!/bin/zsh

# user parameter
TARGET="Moire"

# Environment variables
source /opt/intel/mpi/intel64/bin/mpivars.sh
source /opt/intel/mkl/bin/mklvars.sh intel64 mod lp64
TBBROOT=/opt/intel/tbb; export TBBROOT
LD_LIBRARY_PATH="$TBBROOT/lib:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
LIBRARY_PATH="$TBBROOT/lib:${LIBRARY_PATH}"; export LIBRARY_PATH
CPATH="${TBBROOT}/include:$CPATH"; export CPATH

export I_MPI_PIN_DOMAIN=cache3
export I_MPI_PIN_PROCESSOR_EXCLUDES=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export OMP_NUM_THREADS=8
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
