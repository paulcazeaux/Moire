######################################################################################
### 0) Set your application properties
######################################################################################
# check cmake version compatibility
cmake_minimum_required(VERSION 2.8.8)

# Here define your project name
project(Moire)
SET(CMAKE_BUILD_TYPE Release)

# Here define the name and the version of your package
SET(PACKAGE_NAME “Moire” CACHE STRING "package_name" FORCE)
SET(PACKAGE_VERSION “1.0” CACHE STRING “package_version” FORCE)

# set a language for the entire project.
enable_language(CXX)

# Add first is for debugging second is for performance
#SET(GCC_COMPILE_FLAGS "-std=c++1z -Wno-c++1z-extensions -O0 -g")
SET(GCC_COMPILE_FLAGS "-std=c++1z -Wno-c++1z-extensions -DNDEBUG -O3 -funroll-loops")
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

add_definitions(-DDEPRECATED_MESSAGES)

######################################################################################
### 3) Link the librairies for your executable
######################################################################################

add_subdirectory(src)
add_subdirectory(app)

######################################################################################

