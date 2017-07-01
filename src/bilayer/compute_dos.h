/* 
* File:   bilayer/compute_dos.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/



#ifndef moire__bilayer_computedos_h
#define moire__bilayer_computedos_h

#include "tools/types.h"
#include "bilayer/base_algebra.h"


namespace Bilayer {

    /**
    * A class which encapsulates the DoS computation of a discretized 
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
    *       and a complex<double> otherwise, since there is no need
    *       for adjoint calculations (see BaseAlgebra documentation).
    */
    template <int dim, int degree, typename Scalar = double >
    class ComputeDoS : private BaseAlgebra<dim, degree, Scalar>
    {
    public:
        /**
         *  Public typedefs.
         * These types correspond to the main Trilinos containers
         * used in the discretization.
         */
        typedef Scalar                          scalar_type;
        typedef BaseAlgebra<dim,degree,Scalar>  LA;
        typedef typename LA::MultiVector        MultiVector;
        typedef typename LA::Matrix             Matrix;

        /**
         *  Default constructor. 
         * Initializes 
         * - the underlying BaseAlgebra object,
         * - the conditional output stream on the root node 
         *      (rank 0 on MPI_COMM_WORLD),
         * - the computing timer.
         */
        ComputeDoS(const Multilayer<dim, 2>& bilayer);

        /**
         *  Set up all the necessary objects, vectors, matrices,
         * then runs the computation of the Density of States 
         * of the Tight-Binding bilayer Hamiltonian using the
         * Kernel Polynomial Method.
         */
        void run();

        /**
         *  Output the Chebyshev moments of the diagonal elements
         * (Local Density of States) of the Tight-Binding 
         * bilayer Hamiltonian after it is computed by 
         * the above run() call.
         */
        std::vector<std::array<std::vector<Scalar>,2>> output_LDoS();

        /**
         *  Output the Chebyshev moments of the Density of States
         * of the Tight-Binding bilayer Hamiltonian after it is
         * computed by the above run() call.
         */
        std::vector<Scalar> output_DoS();

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
         * by the Hamiltonian operator within the BaseAlgebra
         * base class object.
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
         * Conditional output stream that only outputs stuff
         * from the root node, i.e. with rank (or pid) zero.
         */
        dealii::ConditionalOStream pcout;

        /**
         * Computing timer. Currently tracks the time spent 
         * in three phases of the computation:
         * - Setup,
         * - Assembly,
         * - Solve (KPM).
         */
        dealii::TimerOutput computing_timer;

        /**
         * Three arrays of two Tpetra MultiVectors each which hold
         * the data of three successive steps during the Chebyshev
         * recursion, corresponding to Chebyshev polynomials
         * applied to the Hamiltonian operator.
         *
         */
        std::array<MultiVector, 2> Tp, T, Tn;

        /**
         * Array of Scalars holding the Chebyshev moments of the 
         * local Density of States of the Hamiltonian, computed 
         * using the Kernel Polynomial Method in the solve() method above.
         */
        std::vector<std::array<std::vector<Scalar>,2>> chebyshev_moments;
    };

}/* End namespace Bilayer */
#endif
