/* 
* File:   bilayer/compute_conductivity.h
* Author: Paul Cazeaux
*
* Created on Feb 24, 2019, 4:00 PM
*/



#ifndef moire__bilayer_computeconductivity_h
#define moire__bilayer_computeconductivity_h

#include <Teuchos_VerboseObject.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include "tools/types.h"
#include "bilayer/base_algebra.h"
#include "bilayer/operator.h"
#include <Thyra_BelosLinearOpWithSolveFactory_decl.hpp>
#include <Thyra_BelosLinearOpWithSolve_decl.hpp>
#include <Thyra_LinearOpWithSolveFactoryHelpers.hpp>


namespace Bilayer {
using Teuchos::RCP;

    /**
    * A class which encapsulates the Conductivity computation of a discretized 
    * Tight-Binding bilayer Hamiltonian encoded by a C* algebra.
    *
    * Its two template parameters are respectively 
    * - dim: the dimension of the problem at hand 
    *       (required to be 1 or 2 at the moment),
    * - degree: the degree of the finite elements used to discretize
    *       and interpolate the functions over the unit cell,
    *       (which can take the values 1, 2 or 3),
    */
    template <int dim, int degree, class Node>
    class ComputeConductivity : private BaseAlgebra<dim, degree, std::complex<double>, Node>
    {
    public:
        /**
         *  Public typedefs.
         * These types correspond to the main Trilinos containers
         * used in the discretization.
         */
        typedef std::complex<double>                                  scalar_type;
        typedef std::complex<double>                                  Scalar;
        typedef typename Bilayer::BaseAlgebra<dim,degree,Scalar,Node> LA;
        typedef typename Bilayer::VectorSpace<dim,degree,Scalar,Node> VS;
        typedef typename Bilayer::Vector<dim,degree,Scalar,Node>      Vec;
        typedef typename Bilayer::MultiVector<dim,degree,Scalar,Node> MVec;
        typedef typename Bilayer::Operator<dim,degree,Scalar,Node>    Op;

        /**
         *  Default constructor. 
         * Initializes 
         * - the underlying BaseAlgebra object,
         * - the conditional output stream on the root node 
         *      (rank 0 on MPI_COMM_WORLD),
         * - the computing timer.
         */
        ComputeConductivity(const Multilayer<dim, 2>& bilayer, 
                            const Scalar tau,
                            const Scalar beta);

        /**
         *  Default destructor. 
         * Outputs timing data.
         */
        ~ComputeConductivity();

        /**
         *  Set up all the necessary objects, vectors, matrices,
         * then runs the computation of the Density of States 
         * of the Tight-Binding bilayer Hamiltonian using the
         * Kernel Polynomial Method.
         */
        void run();

        /**
         *  Write to file the Chebyshev moments of the Density of States
         * of the Tight-Binding bilayer Hamiltonian after it is
         * computed by the above run() call.
         */
        void write_to_file();

        /**
         *  Output the Chebyshev moments of the Density of States
         * of the Tight-Binding bilayer Hamiltonian after it is
         * computed by the above run() call.
         */
        std::vector<std::array<Scalar,dim*dim>> output();

    private:
        /**
         *  Set up the base_algebra object, in particular
         * computes the geometry of the system, computes the
         * matrices for all the operators involved,
         * then initializes the vectors used for the
         * Chebyshev recurrence and solution of the linear
         * system.
         */
        void setup();

        /**
         *  Initialize and run the Chebyshev recurrence of the
         * Kernel Polynomial method, and stores the relevant
         * traces as the Chebyshev moments of the Density of
         * States.
         */
        void solve();


        /**
         *  A utility function: compute the moments of a vector of the Chebyshev 
         *  recurrence and append the result to the chebyshev_moments vector.
         */
         void storeMoments(RCP<Vec> A, int i);

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
         * Three arrays of two Tpetra MultiVectors each which hold
         * the data of three successive steps during the Chebyshev
         * recursion, corresponding to Chebyshev polynomials
         * applied to the Hamiltonian operator.
         *
         */
        const Scalar tau, beta;
        RCP<const VS> vectorSpace;
        RCP<Vec> Tp, T, Tn;
        RCP<MVec> dH, LinvdH;
        RCP<Op> hamiltonianOp, liouvillianOp, derivationOp;

        /**
         * Array of Scalars holding the Chebyshev moments of the 
         * conductivity of the Hamiltonian, computed 
         * using the Kernel Polynomial Method in the solve() method above.
         */
        std::vector<std::array<Scalar,dim*dim>> chebyshev_moments;
    };

}/* End namespace Bilayer */
#endif
