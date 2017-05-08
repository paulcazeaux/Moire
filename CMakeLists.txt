######################################################################################
### 0) Set your application properties
######################################################################################
# check cmake version compatibility
cmake_minimum_required(VERSION 2.8.8)

# Here define your project name
project(HullSpace)

# Here define the name and the version of your package
SET(PACKAGE_NAME “HullSpace” CACHE STRING "package_name" FORCE)
SET(PACKAGE_VERSION “1.0” CACHE STRING “package_version” FORCE)

# set a language for the entire project.
enable_language(CXX)

# Add first is for debugging second is for performance
SET(GCC_COMPILE_FLAGS "-std=c++11")
#SET(GCC_COMPILE_FLAGS "-std=c++11 -DNDEBUG -O3 -funroll-loops")
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/module")
find_package(FFTW)
find_package(MPI)

find_package(deal.II 8.5.0 REQUIRED
		HINTS ${/usr/local/})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

######################################################################################

######################################
### include required modules & utilities
#####################################################################################

include_directories(${FFTW_INCLUDES} ${MPI_INCLUDES})
include(CMakeBackwardCompatibilityCXX)
include(CheckLibraryExists)
           
######################################################################################
### activate test mode
######################################################################################

ENABLE_TESTING()   

#####################################################################################
### Manage the build type
#####################################################################################

# the user should choose the build type on windows environments,excepted under cygwin (default=none)
set(CMAKE_DEFAULT_BUILD_TYPE "Release" CACHE STRING "Variable that stores the default CMake build type" FORCE)

find_program(MEMORYCHECK_COMMAND
    NAMES purify valgrind
    PATHS
    "/usr/local/bin /usr/bin [HKEY_LOCAL_MACHINE\\SOFTWARE\\Rational Software\\Purify\\Setup;InstallFolder]"
    DOC "Path to the memory checking command, used for memory error detection.") 
       

if(CMAKE_BUILD_TYPE MATCHES Debug)
    add_definitions(-DCMAKE_VERBOSE_MAKEFILE=ON)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

add_definitions(-DDEPRECATED_MESSAGES)

######################################################################################
### 3) Link the librairies for your executable
######################################################################################

add_subdirectory(src)
add_subdirectory(app)

######################################################################################

