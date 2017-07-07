######################################################################################
### 0) Set your application properties
######################################################################################
# check cmake version compatibility
cmake_minimum_required(VERSION 2.8.8)

# Here define your project name
project(Moire)
SET(CMAKE_BUILD_TYPE Debug)

# Here define the name and the version of your package
SET(PACKAGE_NAME “Moire” CACHE STRING "package_name" FORCE)
SET(PACKAGE_VERSION “1.0” CACHE STRING “package_version” FORCE)

# set a language for the entire project.
enable_language(CXX)

SET(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} "-std=c++1z -Wno-unused-variable -Wno-unused-parameter")
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/module")


find_package(FFTW)
find_package(MPI)
find_package(Trilinos 12.10.1 REQUIRED
		HINTS ${/usr/local/})
find_package(deal.II 8.5.0 REQUIRED
		HINTS ${/usr/local/})

# Fix dealii.II flags for Clang to avoid warnings
IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    STRING (REPLACE "-funroll-all-loops"    "" DEAL_II_CXX_FLAGS_RELEASE ${DEAL_II_CXX_FLAGS_RELEASE})
    STRING (REPLACE "-Wno-placement-new"    "" DEAL_II_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
    STRING (REPLACE "-Wno-literal-suffix"   "" DEAL_II_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
    STRING (REPLACE "-Wno-long-double"      "" DEAL_II_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
ENDIF()

######################################
### include required modules & utilities
#####################################################################################
include_directories(${FFTW_INCLUDES} ${MPI_INCLUDES} ${Trilinos_INCLUDE_DIRS} ${Trilinos_TPL_INCLUDE_DIRS})
link_directories(${Trilinos_LIBRARY_DIRS} ${Trilinos_TPL_LIBRARY_DIRS})
include(CMakeBackwardCompatibilityCXX)
include(CheckLibraryExists)

add_definitions(-DDEPRECATED_MESSAGES)

######################################################################################
### 3) Link the librairies for your executable
######################################################################################

add_subdirectory(src)
add_subdirectory(app)

######################################################################################
### 4) Enable testing
######################################################################################

enable_testing()
add_subdirectory(tests)


