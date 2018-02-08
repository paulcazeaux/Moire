cmake_minimum_required(VERSION 2.8.8) # Pour éviter un warning au lancement de CMake, on doit préciser la version minimale
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/tests/)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/)

######################################################################################
### Define a macro to set up a  test:
######################################################################################


SET(ALL_TESTS) # clean variable

MACRO(make_test test_name mpi_run)

  LIST(APPEND ALL_TESTS "${test_name}")
  ADD_EXECUTABLE(${test_name} EXCLUDE_FROM_ALL ${test_name}.cpp)
  ADD_DEPENDENCIES(${test_name} Calgebra)
  TARGET_LINK_LIBRARIES(${test_name} Calgebra)
  DEAL_II_SETUP_TARGET(${test_name})

  IF("${mpi_run}" STREQUAL "")
    SET(_command ./${test_name})
  ELSE()
    SET(_command ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${mpi_run} ${MPIEXEC_PREFLAGS} ./${test_name})
  ENDIF()
  ADD_CUSTOM_TARGET(${test_name}.run
    DEPENDS ${test_name}
    COMMAND
      ${_command} > ${test_name}-OK 2>&1
      ||(echo "${test_name}: RUN failed. Output:"
         && cat ${test_name}-OK
         && rm ${test_name}-OK
         && echo "${CMAKE_CURRENT_BINARY_DIR}"
         && exit 1)
    COMMAND echo "${_target}: PASSED."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

  # this is a hack to make sure the -OK file is deleted
  # even if compilation fails.
  ADD_CUSTOM_TARGET(kill-${test_name}-OK
        COMMAND rm -f ${test_name}-OK 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  ADD_DEPENDENCIES(${test_name} kill-${test_name}-OK)

  ADD_TEST(NAME ${test_name}
    COMMAND ${CMAKE_COMMAND} --build . --target ${test_name}.run
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  SET_TESTS_PROPERTIES(${test_name} PROPERTIES LABEL "Sanity checks")
ENDMACRO()
######################################################################################
### 3) Define your target(s): a few tests
######################################################################################
MESSAGE(STATUS "Setting up tests")

# Test Materials: 1D model coefficients
make_test("materials.1d_toy_model" "")

# Test Materials: Graphene coefficients
make_test("materials.graphene" "")

# Test Tools: Periodic translation FFT helper
# make_test("tools.periodic_translation" "")

# Test Bilayer Hamiltonian assembly and symmetry
# make_test("bilayer.test_H_sym" "")

# Test Bilayer Hamiltonian rotation invariance
# make_test("bilayer.test_rotation_invariance" "")

# Print sparsity patterns
make_test("bilayer.print_sparsity_patterns" "3")

######################################################################################
### 4) Wrap up: define a custom test target
######################################################################################
IF(POLICY CMP0037)
  # allow to override "test" target for quick tests
  CMAKE_POLICY(SET CMP0037 OLD)
ENDIF()

ADD_CUSTOM_TARGET(test
  COMMAND ${CMAKE_COMMAND} -D ALL_TESTS="${ALL_TESTS}" -P ${CMAKE_CURRENT_SOURCE_DIR}/run.cmake
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Running tests..."
  )

MESSAGE(STATUS "Setting up tests - Done")
