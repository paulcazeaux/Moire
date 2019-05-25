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

# Debug options to catch segfaults
# add_compile_options(-fsanitize=address -fsanitize-address-use-after-scope)
# add_link_options(-fsanitize=address -fsanitize-address-use-after-scope)

add_compile_options(-std=c++1z -Wno-unused-variable -Wno-unused-parameter)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/module")


find_package(MPI)
find_package(METIS)
find_package(Kokkos REQUIRED
        PATHS /usr/local /pool)
find_package(Teuchos REQUIRED
        PATHS /usr/local /pool)
find_package(Tpetra REQUIRED
		PATHS /usr/local /pool)
find_package(Thyra REQUIRED
        PATHS /usr/local /pool)
find_package(Belos REQUIRED
        PATHS /usr/local /pool)
find_package(deal.II 9.0.0 REQUIRED
		PATHS /usr/local /pool)
find_package(METIS REQUIRED)

# Fix dealii.II flags for Clang to avoid warnings
IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    STRING (REPLACE "-funroll-all-loops"    "" DEAL_II_CXX_FLAGS_RELEASE ${DEAL_II_CXX_FLAGS_RELEASE})
    STRING (REPLACE "-Wno-placement-new"    "" DEAL_II_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
    STRING (REPLACE "-Wno-literal-suffix"   "" DEAL_II_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
    STRING (REPLACE "-Wno-long-double"      "" DEAL_II_CXX_FLAGS ${DEAL_II_CXX_FLAGS})
    add_compile_options(-Wno-c++1z-extensions)
ENDIF()

# Avoid warnings with GCC 7
IF(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wno-int-in-bool-context -Wno-return-type)
ENDIF()

######################################
### include required modules & utilities
#####################################################################################
include_directories(${MPI_INCLUDES} ${Metis_INCLUDES} ${Tpetra_INCLUDE_DIRS} ${Tpetra_TPL_INCLUDE_DIRS} ${Belos_INCLUDE_DIRS} ${Belos_TPL_INCLUDE_DIRS} ${Teuchos_INCLUDE_DIRS} ${Teuchos_TPL_INCLUDE_DIRS} ${Kokkos_INCLUDE_DIRS} ${Kokkos_TPL_INCLUDE_DIRS})
link_directories(${MPI_LIBRARIES} ${Tpetra_LIBRARY_DIRS} ${Tpetra_TPL_LIBRARY_DIRS} ${Thyra_LIBRARY_DIRS} ${Thyra_TPL_LIBRARY_DIRS} ${Belos_LIBRARY_DIRS} ${Belos_TPL_LIBRARY_DIRS} ${Teuchos_LIBRARY_DIRS} ${Teuchos_TPL_LIBRARY_DIRS} ${Kokkos_LIBRARY_DIRS} ${Kokkos_TPL_LIBRARY_DIRS})
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


