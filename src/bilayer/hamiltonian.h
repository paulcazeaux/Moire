/* 
* File:   hamiltonian.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace ::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace ::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
}

#ifndef BILAYER_HAMILTONIAN_H
#define BILAYER_HAMILTONIAN_H

#include <memory>
#include <vector>
#include <array>
#include <utility>

#include "deal.II/base/mpi.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/index_set.h"
#include "deal.II/base/utilities.h"

#include "deal.II/lac/dynamic_sparsity_pattern.h"
#include "deal.II/lac/petsc_parallel_vector.h"

#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unitcell.h"
#include "bilayer/pointdata.h"



namespace Bilayer {


template <int dim, int degree, int derivation>
class Hamiltonian : public Multilayer<dim, 2>
{
public:
	Hamiltonian(const Multilayer<dim, 2>& bilayer, MPI_Comm mpi_communicator);

	void				initialize(const DoFHandler&);
	void 				adjoint() {}; 	// self-adjoint

	friend class Observable<dim, degree>;
private:
	MPI_Comm 							mpi_communicator;
	LA::MPI::SparseMatrix 				right_action;

}

}/* Namespace Bilayer */

#endif /* BILAYER_HAMILTONIAN_H */