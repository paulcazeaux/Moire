/* 
 * Common header for all test files:   tests.h
 * Author: Paul Cazeaux
 *
 * Created on June 16, 2017, 10:00 AM
 */


#ifndef moire__tests_h
#define moire__tests_h



/*      A few lines from the tests.h file from the dealii library     */
// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2017 by the deal.II authors
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

// common definitions used in all the tests

#include <deal.II/base/config.h>
#include <deal.II/base/job_identifier.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/multithread_info.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
 
#if defined(DEBUG) && defined(DEAL_II_HAVE_FP_EXCEPTIONS)
#  include <cfenv>
#endif

// silence extra diagnostics in the testsuite
#ifdef DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
DEAL_II_DISABLE_EXTRA_DIAGNOSTICS
#endif

#ifdef DEAL_II_WITH_PETSC
#include <petscsys.h>

namespace dealii {

namespace
{
  void check_petsc_allocations()
  {
#if DEAL_II_PETSC_VERSION_GTE(3, 2, 0)
    PetscStageLog stageLog;
    PetscLogGetStageLog(&stageLog);

    // I don't quite understand petsc and it looks like
    // stageLog->stageInfo->classLog->classInfo[i].id is always -1, so we look
    // it up in stageLog->classLog, make sure it has the same number of entries:
    Assert(stageLog->stageInfo->classLog->numClasses == stageLog->classLog->numClasses,
           dealii::ExcInternalError());

    bool errors = false;
    for (int i=0; i<stageLog->stageInfo->classLog->numClasses; ++i)
      {
        if (stageLog->stageInfo->classLog->classInfo[i].destructions !=
            stageLog->stageInfo->classLog->classInfo[i].creations)
          {
            errors = true;
            std::cerr << "ERROR: PETSc objects leaking of type '"
                      << stageLog->classLog->classInfo[i].name << "'"
                      << " with "
                      << stageLog->stageInfo->classLog->classInfo[i].creations
                      << " creations and only "
                      << stageLog->stageInfo->classLog->classInfo[i].destructions
                      << " destructions." << std::endl;
          }
      }

    if (errors)
      throw dealii::ExcMessage("PETSc memory leak");
#endif
  }
}
#endif


// Function to initialize deallog. Normally, it should be called at
// the beginning of main() like
//
// initlog();
//
// This will open the correct output file, divert log output there and
// switch off screen output. If screen output is desired, provide the
// optional second argument as 'true'.
std::string deallogname;
std::ofstream deallogfile;

void
initlog(bool console=false)
{
  deallogname = "output";
  deallogfile.open(deallogname.c_str());
  deallog.attach(deallogfile);
  deallog.depth_console(console?10:0);

//TODO: Remove this line and replace by test_mode()
  deallog.threshold_float(1.e-8);
}


inline
void
mpi_initlog(bool console=false)
{
#ifdef DEAL_II_WITH_MPI
  unsigned int myid = Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);
  if (myid == 0)
    {
      deallogname = "output";
      deallogfile.open(deallogname.c_str());
      deallog.attach(deallogfile);
      deallog.depth_console(console?10:0);

//TODO: Remove this line and replace by test_mode()
      deallog.threshold_float(1.e-8);
    }
#else
  (void)console;
  // can't use this function if not using MPI
  Assert (false, ExcInternalError());
#endif
}


/* helper class to include the deallogs of all processors
   on proc 0 */
struct MPILogInitAll
{
  MPILogInitAll(bool console=false)
  {
#ifdef DEAL_II_WITH_MPI
    unsigned int myid = Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);
    deallogname = "output";
    if (myid != 0)
      deallogname = deallogname + Utilities::int_to_string(myid);
    deallogfile.open(deallogname.c_str());
    deallog.attach(deallogfile);
    deallog.depth_console(console?10:0);

//TODO: Remove this line and replace by test_mode()
    deallog.threshold_float(1.e-8);
    deallog.push(Utilities::int_to_string(myid));
#else
    (void)console;
    // can't use this function if not using MPI
    Assert (false, ExcInternalError());
#endif
  }

  ~MPILogInitAll()
  {
#ifdef DEAL_II_WITH_MPI
    unsigned int myid = Utilities::MPI::this_mpi_process (MPI_COMM_WORLD);
    unsigned int nproc = Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD);

    deallog.pop();

    if (myid!=0)
      {
        deallog.detach();
        deallogfile.close();
      }

    MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEAL_II_WITH_PETSC
    check_petsc_allocations();
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if (myid==0)
      {
        for (unsigned int i=1; i<nproc; ++i)
          {
            std::string filename = "output" + Utilities::int_to_string(i);
            std::ifstream in(filename.c_str());
            Assert (in, ExcIO());

            while (in)
              {
                std::string s;
                std::getline(in, s);
                deallog.get_file_stream() << s << "\n";
              }
            in.close();
            std::remove (filename.c_str());
          }
      }
#else
    // can't use this function if not using MPI
    Assert (false, ExcInternalError());
#endif
  }

};

} // namespace dealii


// ------------------------------ Adjust global variables in deal.II -----------------------


DEAL_II_NAMESPACE_OPEN
/*
 * Now, change some global behavior of deal.II and supporting libraries:
 */

/* Disable stack traces: */

struct SwitchOffStacktrace
{
  SwitchOffStacktrace ()
  {
    deal_II_exceptions::suppress_stacktrace_in_exceptions ();
  }
} deal_II_stacktrace_dummy;



/* Enable floating point exceptions in debug mode and if we have
   detected that they are usable: */

struct EnableFPE
{
  EnableFPE ()
  {
#if defined(DEBUG) && defined(DEAL_II_HAVE_FP_EXCEPTIONS)
    // enable floating point exceptions
    feenableexcept(FE_DIVBYZERO|FE_INVALID);
#endif
  }
} deal_II_enable_fpe;


/* Set grainsizes for parallel mode smaller than they would otherwise be.
 * This is used to test that the parallel algorithms in lac/ work alright:
 */

namespace internal
{
  namespace Vector
  {
    extern unsigned int minimum_parallel_grain_size;
  }
  namespace SparseMatrix
  {
    extern unsigned int minimum_parallel_grain_size;
  }
}

struct SetGrainSizes
{
  SetGrainSizes ()
  {
    internal::Vector::minimum_parallel_grain_size = 2;
    internal::SparseMatrix::minimum_parallel_grain_size = 2;
  }
} set_grain_sizes;

DEAL_II_NAMESPACE_CLOSE

/*
 * Do not use a template here to work around an overload resolution issue with clang and
 * enabled  C++11 mode.
 *
 * - Maier 2013
 */
dealii::LogStream &
operator << (dealii::LogStream &out,
             const std::vector<unsigned int> &v)
{
  for (unsigned int i=0; i<v.size(); ++i)
    out << v[i] << (i == v.size()-1 ? "" : " ");
  return out;
}

dealii::LogStream &
operator << (dealii::LogStream &out,
             const std::vector<long long unsigned int> &v)
{
  for (unsigned int i=0; i<v.size(); ++i)
    out << v[i] << (i == v.size()-1 ? "" : " ");
  return out;
}

dealii::LogStream &
operator << (dealii::LogStream &out,
             const std::vector<double> &v)
{
  for (unsigned int i=0; i<v.size(); ++i)
    out << v[i] << (i == v.size()-1 ? "" : " ");
  return out;
}

#endif