# - Find METIS
# Find the native METIS includes and library
# 
#  METIS_INCLUDES    - where to find metis.h
#  METIS_LIBRARIES   - List of libraries when using METIS.
#  METIS_FOUND       - True if METIS found.

if (METIS_INCLUDES)
  # Already in cache, be silent
  set (METIS_FIND_QUIETLY TRUE)
endif (METIS_INCLUDES)

find_path (METIS_INCLUDES metis.h)

find_library (METIS_LIBRARIES NAMES metis)

# handle the QUIETLY and REQUIRED arguments and set METIS_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (METIS DEFAULT_MSG METIS_LIBRARIES METIS_INCLUDES)

mark_as_advanced (METIS_LIBRARIES METIS_INCLUDES)