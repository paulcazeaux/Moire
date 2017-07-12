/* 
* File:   bilayer/base_algebra.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/



#ifndef moire__bilayer_base_algebra_h
#define moire__bilayer_base_algebra_h

#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <fstream>

#include <complex>
#include "fftw3.h"

#include <Tpetra_DefaultPlatform.hpp>
#include <Kokkos_Core.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_Comm.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>

#include "deal.II/base/exceptions.h"
#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/conditional_ostream.h"
#include "deal.II/base/timer.h"

#include "tools/types.h"
#include "tools/numbers.h"
#include "tools/periodic_translation_unit.h"
#include "bilayer/dof_handler.h"



namespace Bilayer {
    
    /**
    * A class encapsulating the basic operations in a C* algebra equipped with a discretized Hamiltonian.
    * It is intended as a base class creating the necessary operations for use in further computations 
    * (density of states, conductivity, etc.)
    *
    * Its three template parameters are respectively 
    * - dim: the dimension of the problem at hand 
    *       (required to be 1 or 2 at the moment),
    * - degree: the degree of the finite elements used to discretize
    *       and interpolate the functions over the unit cell,
    *       (which can take the values 1, 2 or 3),
    * - Scalar: the main number type used in the computation.
    *       This should be a complex double at the moment when the
    *       computation of the adjoint is needed due the implementation 
    *       details of the FFTW computations, but we could also 
    *       implement real types (and then use the appropriate
    *       FFT operations specialized for reals.)
    */
    template <int dim, int degree, typename Scalar>
    class BaseAlgebra
    {
    public:
        /**
         *  Public typedefs.
         * These types correspond to the main Trilinos containers
         * used in the discretization.
         */
        typedef typename Tpetra::MultiVector<Scalar,types::loc_t, types::glob_t, Kokkos::Compat::KokkosSerialWrapperNode>  MultiVector;
        typedef typename Tpetra::CrsMatrix<Scalar, types::loc_t, types::glob_t, Kokkos::Compat::KokkosSerialWrapperNode>   Matrix;

        /**
         *  Default constructor.
         * Takes in a particular instance of a Multilayer object with
         * the desired geometrical and material parameters.
         *
         * Initializes 
         * - the underling MPI communicator to MPI_COMM_WORLD,
         * - the dof_handler object in charge of managing the interplay
         *     between geometry and degrees of freedom,
         * - raw input/output memory arrays used during the DFT 
         *     computations  with the special aligned allocation 
         *     fftw_malloc as recommended by the FFTW library,
         * - the FFTW plans for computation of said Fourier transforms.
         */
        BaseAlgebra(const Multilayer<dim, 2>& bilayer);

    protected:
        void                base_setup();
        void                assemble_base_matrices();
        void                assemble_hamiltonian_action();
        void                assemble_adjoint_interpolant();


        /* update (wrapper around the multivector update function): A = alpha * A + beta * B. */
        void                linear_combination(const Scalar alpha, const std::array<MultiVector, 2> A,
                                                const Scalar beta, std::array<MultiVector, 2> & B);
        /* Assemble identity observable */
        void                create_identity(std::array<MultiVector, 2>& Id);
        /* Application of the hamiltonian action, representing the right-product in the C* algebra */
        void                hamiltonian_rproduct(const std::array<MultiVector, 2> A, std::array<MultiVector, 2> & B);
        /* Adjoint operation on an observable */
        void                adjoint(const std::array<MultiVector, 2> A, std::array<MultiVector, 2>& tA);
        /* Diagonal of an observable, available on all processes */
        std::array<std::vector<Scalar>,2>
                            diagonal(const std::array<MultiVector, 2> A);
        /* Trace of an observable, available on all processes */
        Scalar              trace(const std::array<MultiVector, 2> A);

        /* MPI communication environment and utilities */
        Teuchos::RCP<const Teuchos::Comm<int> >             mpi_communicator;

        /* DoF Handler object and local indices range */
        DoFHandler<dim,degree>                              dof_handler;

        /* Matrices representing the sparse linear action of the two main operations, acting on blocks */
        std::array<std::array<Teuchos::RCP<Matrix>, 2>, 2>  adjoint_interpolant;
        std::array<Teuchos::RCP<Matrix>, 2>                 hamiltonian_action;

        /* Data structures allocated for additional local computations in the adjoint operation */
        std::array<PeriodicTranslationUnit<dim, Scalar>, 2> torus;
        std::array<std::array<MultiVector, 2>, 2>           helper;

        /* Convenience functions */
        const LayerData<dim>&       layer(const int & idx)      const { return dof_handler.layer(idx); }
        const Lattice<dim>&         lattice(const int & idx)    const { return dof_handler.lattice(idx); };
        const UnitCell<dim,degree>& unit_cell(const int & idx)  const { return dof_handler.unit_cell(idx); };
    };

}/* End namespace Bilayer */
#endif
