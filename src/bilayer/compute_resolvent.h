/* 
* File:   bilayer/compute_dos.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/



#ifndef moire__bilayer_compute_resolvent_h
#define moire__bilayer_compute_resolvent_h

#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <BelosSolverFactory.hpp>
#include "tools/types.h"
#include "bilayer/base_algebra.h"
#include <BelosTpetraAdapter.hpp>


namespace Bilayer {

    /**
    * A class which encapsulates the resolvent computation of a discretized 
    * Tight-Binding bilayer Hamiltonian encoded by a C* algebra.
    *
    * Its three template parameters are respectively 
    * - dim: the dimension of the problem at hand 
    *       (required to be 1 or 2 at the moment),
    * - degree: the degree of the finite elements used to discretize
    *       and interpolate the functions over the unit cell,
    *       (which can take the values 1, 2 or 3),
    * - Scalar: the main number type used in the computation.
    *       This should be a double when no magnetic field is involved
    *       and a complex<double> otherwise.
    */
    template <int dim, int degree, typename Scalar, class Node>
    class ComputeResolvent : private BaseAlgebra<dim,degree,Scalar,Node>
    {
    public:
        /**
         *  Public typedefs.
         * These types correspond to the main Trilinos containers
         * used in the discretization.
         */
        typedef Scalar                          scalar_type;
        typedef BaseAlgebra<dim,degree,Scalar,Node>  LA;
        typedef typename LA::MultiVector        MultiVector;
        typedef typename LA::Matrix             Matrix;
        typedef typename LA::Operator           Operator;

        /**
         *  Default constructor. 
         * Initializes 
         * - the underlying BaseAlgebra object,
         * - the conditional output stream on the root node 
         *      (rank 0 on MPI_COMM_WORLD).
         */
        ComputeResolvent(const Multilayer<dim, 2>& bilayer);

        /**
         *  Default destructor. 
         * Outputs timing data.
         */
        ~ComputeResolvent();

        /**
         *  Set up all the necessary objects, vectors, matrices,
         * then runs the computation of the Density of States 
         * of the Tight-Binding bilayer Hamiltonian using the
         * Kernel Polynomial Method.
         */
        void run();

        /**
         *  Write to file ?? TBD ??
         * of the Tight-Binding bilayer Hamiltonian after it is
         * computed by the above run() call.
         */
        void write_to_file();

        /**
         *  Determine an estimate of the current memory usage
         * by this class, including the BaseAlgebra matrices
         * and the dof_handler object data fields, on the
         * local node. 
         * The global usage should then be reduced across 
         * MPI nodes if necessary.
         */
        types::MemUsage memory_consumption() const;

    private:
        /**
         *  Set up the dof_handler object, in particular
         * computes the geometry of the system,
         * then initializes the three vectors used for the
         * Chebyshev recurrence.
         */
        void setup();

        /**
         *  Assembles the matrices used to represent the product
         * by the Hamiltonian operator and the Transpose interpolation
         * within the BaseAlgebra base class object.
         */
        void assemble_matrices();

        /**
         *  Initialize and run the Chebyshev recurrence of the
         * Kernel Polynomial method, and stores the relevant
         * traces as the Chebyshev moments of the Density of
         * States.
         */
        void solve();

        /**
         *  Blackhole stream that does nothing but discard output.
         */
        Teuchos::oblackholestream blackHole;

        /**
         *  Conditional output stream that only outputs stuff
         * from the root node, i.e. with rank (or pid) zero.
         */
        std::ostream & pcout;

        /**
         * Two Tpetra MultiVectors each which hold
         * the data of the initial identity array and the computed
         * resolvent
         */
        MultiVector I, R;
    };

}/* End namespace Bilayer */
#endif
