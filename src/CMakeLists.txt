cmake_minimum_required(VERSION 2.8.8) # Pour éviter un warning au lancement de CMake, on doit préciser la version minimale
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/)

######################################################################################
### 3) Define your target(s): just the flow-shop lib here
######################################################################################

SET(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

FILE( GLOB_RECURSE source_files *.cpp )

ADD_LIBRARY( Calgebra STATIC ${source_files})


######################################################################################
### 3) Link the librairies for your library
######################################################################################
DEAL_II_SETUP_TARGET(Calgebra)
TARGET_LINK_LIBRARIES( Calgebra ${MPI_LIBRARIES} ${METIS_LIBRARIES}
                                ${Kokkos_LIBRARIES} ${Kokkos_TPL_LIBRARIES} 
                                ${Teuchos_LIBRARIES} ${Teuchos_TPL_LIBRARIES} 
                                ${Tpetra_LIBRARIES} ${Tpetra_TPL_LIBRARIES} 
                                ${Thyra_LIBRARIES} ${Thyra_TPL_LIBRARIES} 
                                ${Belos_LIBRARIES} ${Belos_TPL_LIBRARIES} )

######################################################################################