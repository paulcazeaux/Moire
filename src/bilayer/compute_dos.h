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



    template<int dim, int degree, typename Scalar>
    ComputeDoS<dim,degree,Scalar>::ComputeDoS(const Multilayer<dim, 2>& bilayer)
        :
        BaseAlgebra<dim,degree,Scalar>(bilayer),
        pcout (std::cout, (this->mpi_communicator->getRank () == 0)),
        computing_timer (MPI_COMM_WORLD,
                       pcout,
                       dealii::TimerOutput::summary,
                       dealii::TimerOutput::wall_times)
    {}


    template<int dim, int degree, typename Scalar>
    void
    ComputeDoS<dim,degree,Scalar>::run()
    {
        pcout   << "Starting setup...\n";
        setup();
        pcout   << "\tComplete!\n";

        pcout   << "Starting assembly...\n";
        assemble_matrices();
        pcout   << "\tComplete!\n";


        pcout   << "Starting solve...\n";
        solve();
        pcout   << "\tComplete!\n";
    }


    template<int dim, int degree, typename Scalar>
    std::vector<std::array<std::vector<Scalar>,2>>
    ComputeDoS<dim,degree,Scalar>::output_LDoS()
    {
        return chebyshev_moments;
    }

     template<int dim, int degree, typename Scalar>
    std::vector<Scalar>
    ComputeDoS<dim,degree,Scalar>::output_DoS()
    {
        std::vector<Scalar> DoS;
        for ( const auto diag_moments : chebyshev_moments)
        {
            Scalar T = std::accumulate(diag_moments.at(0).begin(), diag_moments.at(0).end(), 0.0)
                                    * this->unit_cell(1).area / (this->unit_cell(0).area + this->unit_cell(1).area)
                                    / static_cast<double>( this->dof_handler.n_domain_orbitals(0,0) * this->dof_handler.n_cell_nodes(0,0) )
                        + std::accumulate(diag_moments.at(1).begin(), diag_moments.at(1).end(), 0.0)
                                    * this->unit_cell(0).area / (this->unit_cell(0).area + this->unit_cell(1).area)
                                    / static_cast<double>( this->dof_handler.n_domain_orbitals(1,1) * this->dof_handler.n_cell_nodes(1,1) );
            DoS.push_back(T);
        }
        return DoS;
    }

    template<int dim, int degree, typename Scalar>
    types::MemUsage
    ComputeDoS<dim,degree,Scalar>::memory_consumption() const
    {
        types::MemUsage memory = this->dof_handler.memory_consumption();
        memory.Static += sizeof(this);
        memory.Static += sizeof(T) + sizeof(Tp) + sizeof(Tn);
        memory.Static += sizeof( this->hamiltonian_action ) + sizeof( this->adjoint_interpolant );

        memory.Vectors += 3 * sizeof(Scalar) * ( T.at(0).getNumVectors() * T.at(0).getGlobalLength()
                                               + T.at(1).getNumVectors() * T.at(1).getGlobalLength() );
        memory.Matrices += (sizeof(Scalar) + sizeof(Matrix::global_ordinal_type) ) 
                            * ( this->hamiltonian_action.at(0)->getNodeNumEntries() + this->hamiltonian_action.at(1)->getNodeNumEntries() );

        return memory;
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeDoS<dim,degree,Scalar>::setup()
    {
        dealii::TimerOutput::Scope t(computing_timer, "Setup");
        LA::base_setup();
        
        Tp = {{ MultiVector( this->dof_handler.locally_owned_dofs(0), this->dof_handler.n_range_orbitals(0,0) ), 
                MultiVector( this->dof_handler.locally_owned_dofs(1), this->dof_handler.n_range_orbitals(1,1) ) }};
        T  = {{ Tpetra::createCopy(Tp[0]),  Tpetra::createCopy(Tp[1]) }};
        Tn = {{ Tpetra::createCopy(Tp[0]),  Tpetra::createCopy(Tp[1]) }};
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeDoS<dim,degree,Scalar>::assemble_matrices()
    {
        dealii::TimerOutput::Scope t(computing_timer, "Assembly");

        LA::assemble_hamiltonian_action();
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeDoS<dim,degree,Scalar>::solve()
    {
        dealii::TimerOutput::Scope t(computing_timer, "Solve");

        /* Initialization of the vector to the identity */

        LA::create_identity(Tp);
        LA::hamiltonian_rproduct(Tp, T);

        std::array<std::vector<Scalar>,2> 
        m0 = LA::diagonal(Tp), 
        m = LA::diagonal(T);

        if (this->dof_handler.my_pid == 0)
        {
          chebyshev_moments.push_back(m0);
          chebyshev_moments.push_back(m);
        }

        for (int i=2; i <= this->dof_handler.poly_degree; ++i)
        {
            LA::hamiltonian_rproduct(T, Tn);
            LA::linear_combination(-1, Tp, 2., Tn);

            std::swap(Tn, Tp);
            std::swap(T, Tp);

            m = LA::diagonal(T);
            if (this->dof_handler.my_pid == 0)
                chebyshev_moments.push_back(m);
        }
    }

}/* End namespace Bilayer */
#endif
