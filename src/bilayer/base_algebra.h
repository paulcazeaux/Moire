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
#include "RTOpPack_Types.hpp"

#include <Kokkos_Core.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_Operator.hpp>

#include "tools/types.h"
#include "tools/numbers.h"
#include "bilayer/dof_handler.h"



namespace Bilayer {
    using Teuchos::RCP;
    
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
    */
    template <int dim, int degree, typename Scalar, class Node>
    class BaseAlgebra
    {

        /** 
         *  Advanced protected typedefs, for specific kind of operators representing the useful operations in the C* algebra.
         */
    protected:
        class RangeBlockOp;
        class TransposeOp;
        class LiouvillianOp;
        class DerivationOp;

    public:
        /**
         *  Public typedefs.
         * These types correspond to the main Trilinos containers
         * used in the discretization.
         */
            typedef typename 
            Tpetra::MultiVector<Scalar,types::loc_t, types::glob_t, Node>  
        Vector;
            typedef typename 
            Tpetra::MultiVector<Scalar,types::loc_t, types::glob_t, Node>  
        MultiVector;
            typedef typename 
            Tpetra::CrsMatrix<Scalar, types::loc_t, types::glob_t, Node>   
        Matrix;
            typedef typename 
            Tpetra::Operator<Scalar, types::loc_t, types::glob_t, Node>    
        Operator;

        /**
         *  Default constructor.
         * Takes in a particular instance of a Multilayer object with
         * the desired geometrical and material parameters.
         *
         * Initializes 
         * - the underling MPI communicator to MPI_COMM_WORLD,
         * - the dof_handler object in charge of managing the interplay
         *     between geometry and degrees of freedom.
         */
        BaseAlgebra( const Multilayer<dim, 2>& bilayer);

        /* Create a basic Vector with the right data structure */
            Vector
        create_vector( bool ZeroOut = true ) const;
        /* Create a basic MultiVector with the right data structure */
            MultiVector
        create_multivector( size_t numVecs, 
                            bool ZeroOut = true ) const;

        /* Assemble identity observable */
            void
        set_to_identity( Vector& Id ) const;

        /* Diagonal of an observable, available on all processes */
            std::array<std::vector<Scalar>,2>
        diagonal( const Vector& A ) const;

        /* Trace of an observable, available on all processes */
            Scalar
        trace(  const Vector& A ) const;

        /* Scalar product of two observable as Tr(A^H * B), 
         * available on all processes 
         */
            Scalar
        dot(    const Vector& A, 
                const Vector& B ) const;
            void
        dot(    const MultiVector& A, 
                const MultiVector& B, 
                Teuchos::ArrayView<Scalar> dots) const;


        /* Operator representing the hamiltonian action, 
         * representing the right-product in the C* algebra 
         */
            RCP<const RangeBlockOp>    
        HamiltonianAction;

        /* Operator representing the transpose operation on an observable (*/
            RCP<const TransposeOp>     
        Transpose;

            RCP<const DerivationOp>
        Derivation;

        /* Scaled and shifted Liouvillian operator s ( A^* conj(H) A - H) + z I: */
            RCP<const LiouvillianOp>   
        make_liouvillian_operator(
                const Scalar z = Teuchos::ScalarTraits<Scalar>::zero (), 
                const Scalar s = Teuchos::ScalarTraits<Scalar>::one ())
            { return Teuchos::rcp(new LiouvillianOp(Transpose, HamiltonianAction, z, s)); };

        /* MPI utilities */
            RCP<const Teuchos::Comm<int> >
        getComm() const         { return mpi_communicator; };

            RCP<const typename Matrix::map_type>
        getMap() const          { return dof_handler.locally_owned_dofs(); };
        
            size_t
        getNumOrbitals() const  { return dof_handler.n_orbitals(0) 
                                        + dof_handler.n_orbitals(1); };
    protected:

        /* Initialization routines */
            void
        assemble_base_matrices();
            void
        assemble_hamiltonian_action();
            void
        assemble_transpose_interpolant();


        /* MPI communication environment and utilities */
            RCP<const Teuchos::Comm<int> >
        mpi_communicator;

        /* DoF Handler object */
            DoFHandler<dim,degree,Node>
        dof_handler;

        /* Matrices representing the sparse linear action 
         * of the two main operations, acting on blocks 
         */
            std::array<std::array<RCP<Matrix>, 2>, 2>
        transpose_interpolant;
            std::array<RCP<Matrix>, 2>
        hamiltonian_action;

        /* Convenience functions */
            const LayerData<dim>&
        layer(const int & idx)      const { return dof_handler.layer(idx); };
            const Lattice<dim>&
        lattice(const int & idx)    const { return dof_handler.lattice(idx); };
            const UnitCell<dim,degree>&
        unit_cell(const int & idx)  const { return dof_handler.unit_cell(idx); };
    

        /* Classes implementing the Tpetra Operator structure 
         * for our particular data structures 
         */
        class RangeBlockOp: public Operator {
        public:
                typedef typename 
                Operator::scalar_type 
            scalar_type;
                typedef typename 
                Operator::local_ordinal_type 
            local_ordinal_type;
                typedef typename 
                Operator::global_ordinal_type 
            global_ordinal_type;
                typedef typename 
                Operator::node_type 
            node_type;

                std::array<RCP<Matrix>, 2>                 
            A;
                std::array<Teuchos::Range1D, 2>
            ColumnRange;
                RCP<const typename Matrix::map_type>
            DomainMap;
                RCP<const typename Matrix::map_type>
            RangeMap;

        
            // Constructor for the Block operator:
            // H: array of two matrices acting on each range block.

            RangeBlockOp() {}
            RangeBlockOp( 
                    std::array<RCP<Matrix>, 2>& A, 
                    std::array<const size_t,2> n_orbitals
                        );
                

            //
            // These functions are required since we inherit from Tpetra::Operator
            //
            // Destructor
            virtual ~RangeBlockOp () {}

            // Get the domain Map of this Operator subclass.
                RCP<const typename Matrix::map_type> 
            getDomainMap()  const { return DomainMap; }

            // Get the range Map of this Operator subclass.
                RCP<const typename Matrix::map_type> 
            getRangeMap()   const { return RangeMap; }

            bool 
            hasTransposeApply  ()   const { return true; }

            // Compute Y := alpha mode(Op) X + beta Y.
                void
            apply ( const MultiVector&  X,
                    MultiVector&        Y, 
                    Teuchos::ETransp    mode = Teuchos::NO_TRANS,
                    scalar_type         alpha = Teuchos::ScalarTraits<scalar_type>::one (),
                    scalar_type         beta = Teuchos::ScalarTraits<scalar_type>::zero ()
                    ) const;
        };

        class TransposeOp: public Operator{
            public:
                typedef typename 
                Operator::scalar_type 
            scalar_type;
                typedef typename 
                Operator::local_ordinal_type 
            local_ordinal_type;
                typedef typename 
                Operator::global_ordinal_type 
            global_ordinal_type;
                typedef typename 
                Operator::node_type 
            node_type;

                std::array<std::array<RCP<Matrix>, 2>, 2>
            A;
                std::array<size_t,2>
            nOrbitals;
                std::array<double,2>
            unitCellAreas;
                std::array<Teuchos::Range1D, 2>
            ColumnRange;
                RCP<const typename Matrix::map_type>
            DomainMap;
                RCP<const typename Matrix::map_type>
            RangeMap;

            /* Data structures allocated for additional local 
             * computations in the adjoint operation 
             */
                std::array<std::array<RCP<MultiVector>, 2>, 2>
            helper;
        
            /* Constructor for the Block operator:
             * H: array of two matrices acting on each range block.
             */

            TransposeOp() {}
            TransposeOp(   
                    std::array<std::array<RCP<Matrix>, 2>, 2>& A, 
                    std::array<size_t,2>                       n_orbitals,
                    std::array<double, 2>                      unit_cell_areas,
                    RCP<const typename Matrix::map_type>       domain_map, 
                    RCP<const typename Matrix::map_type>       range_map
                        );

            //
            // These functions are required since we inherit from Tpetra::Operator
            //
            // Destructor
            virtual ~TransposeOp() {}

            // Get the domain Map of this Operator subclass.
                RCP<const typename
            Matrix::map_type> getDomainMap()            const { return DomainMap; }

            // Get the range Map of this Operator subclass.
                RCP<const typename Matrix::map_type>
            getRangeMap()                               const { return RangeMap; }

                bool 
            hasTransposeApply()                         const { return true; }

                std::array<std::array<RCP<const MultiVector>, 2>, 2>
            block_view_const(const MultiVector& X, const size_t j = 0) const;

                std::array<std::array<RCP<MultiVector>, 2>, 2>
            block_view(MultiVector& X, const size_t j = 0) const;

            // Compute Y := alpha mode(Op) X + beta Y.
                void
            apply(
                const MultiVector&  X,
                MultiVector&        Y, 
                Teuchos::ETransp    mode    = Teuchos::NO_TRANS,
                scalar_type         alpha   = Teuchos::ScalarTraits<scalar_type>::one (),
                scalar_type         beta    = Teuchos::ScalarTraits<scalar_type>::zero ()
                    ) const;
        };

        class DerivationOp: public Operator{
        public:
                typedef typename 
                Operator::scalar_type
            scalar_type;
                typedef typename 
                Operator::local_ordinal_type
            local_ordinal_type;
                typedef typename 
                Operator::global_ordinal_type
            global_ordinal_type;
                typedef typename 
                Operator::node_type
            node_type;
            /* An array storing local node coordinates */
                Kokkos::View<double*[2][dim], Kokkos::LayoutLeft> 
            nodalPositions_;
                RCP<const typename Matrix::map_type> 
            map_;
                std::array<size_t,2>
            nOrbitals_;
                std::array<double,2>
            normalizationFactor_;

            /* Constructor for the Block operator:
             * H: array of two matrices acting on each range block.
             */

            DerivationOp() {}
            DerivationOp( RCP<const DoFHandler<dim,degree,Node>> dofHandler );

            //
            // These functions are required since we inherit from Tpetra::Operator
            //
            // Destructor
            virtual ~DerivationOp() {}

            // Get the domain Map of this Operator subclass.
                RCP<const typename Matrix::map_type> 
            getDomainMap()                              const { return map_; }

            // Get the range Map of this Operator subclass.
                RCP<const typename Matrix::map_type>
            getRangeMap()                               const { return map_; }

                bool 
            hasTransposeApply()                         const { return true; }

                void
            weightedDot(
                const MultiVector& X,
                const MultiVector& Y,
                Teuchos::ArrayView<Scalar>& dots
                        ) const;

            // Compute Y := alpha mode(Op) X + beta Y.
                void
            apply(
                const MultiVector&  X,
                MultiVector&        Y, 
                Teuchos::ETransp    mode = Teuchos::NO_TRANS,
                scalar_type         alpha = Teuchos::ScalarTraits<scalar_type>::one (),
                scalar_type         beta = Teuchos::ScalarTraits<scalar_type>::zero ()
                    ) const;
        };



        class LiouvillianOp: public Operator {
        public:
                typedef typename 
                Operator::scalar_type
            scalar_type;
                typedef typename 
                Operator::local_ordinal_type
            local_ordinal_type;
                typedef typename 
                Operator::global_ordinal_type
            global_ordinal_type;
                typedef typename 
                Operator::node_type
            node_type;

            /* Base operators */
                RCP<const TransposeOp>
            A;
                RCP<const RangeBlockOp>
            H;
                scalar_type 
            s, z;

                RCP<const typename Matrix::map_type>
            DomainMap;
                RCP<const typename Matrix::map_type>
            RangeMap;

            /* Internal storage for the computation of the Liouvillian */
                RCP<MultiVector> 
            T1, T2;
        
            // Constructor for the Liouvillian:
            LiouvillianOp (   
                    RCP<const TransposeOp>&     A, 
                    RCP<const RangeBlockOp>&    H,
                    const scalar_type           z = Teuchos::ScalarTraits<scalar_type>::zero (), 
                    const scalar_type           s = Teuchos::ScalarTraits<scalar_type>::one ());

            //
            // These functions are required since we inherit from Tpetra::Operator
            //
            // Destructor
            virtual ~LiouvillianOp () {}

            // Get the domain Map of this Operator subclass.
                RCP<const typename Matrix::map_type> 
                getDomainMap() const                    { return DomainMap; }

            // Get the range Map of this Operator subclass.
                RCP<const typename Matrix::map_type> 
            getRangeMap() const                         { return RangeMap; }

            bool 
            hasTransposeApply() const                   { return true; }

            // Compute Y := alpha mode(Op) X + beta Y.
            void
            apply(
                const MultiVector&  X,
                MultiVector&        Y, 
                Teuchos::ETransp    mode    = Teuchos::NO_TRANS,
                scalar_type         alpha   = Teuchos::ScalarTraits<scalar_type>::one (),
                scalar_type         beta    = Teuchos::ScalarTraits<scalar_type>::zero ()
                ) const;
        };
    };

} /* End namespace Bilayer */
#endif
