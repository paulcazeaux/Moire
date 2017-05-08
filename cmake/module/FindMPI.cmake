# - Find MPI
# Find the native MPI includes and library
# 
#  MPI_INCLUDES    - where to find mpi.h
#  MPI_LIBRARIES   - List of libraries when using MPI.
#  MPI_FOUND       - True if MPI found.

if (MPI_INCLUDES)
  # Already in cache, be silent
  set (MPI_FIND_QUIETLY TRUE)
endif (MPI_INCLUDES)

find_path (MPI_INCLUDES mpi.h)

find_library (MPI_LIBRARIES NAMES mpi)

# handle the QUIETLY and REQUIRED arguments and set MPI_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (MPI DEFAULT_MSG MPI_LIBRARIES MPI_INCLUDES)

mark_as_advanced (MPI_LIBRARIES MPI_INCLUDES)