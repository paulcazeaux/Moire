/* 
* File:   bilayer/compute_resolvent.h
* Author: Paul Cazeaux
*
* Created on June 30, 2017, 9:00 PM
*/

#include "bilayer/compute_resolvent.h"

namespace Bilayer {

    template<int dim, int degree, typename Scalar>
    ComputeResolvent<dim,degree,Scalar>::ComputeResolvent(const Multilayer<dim, 2>& bilayer)
        :
        BaseAlgebra<dim,degree,Scalar>(bilayer),
        pcout(( this->mpi_communicator->getRank () == 0) ? std::cout : blackHole)
    {
        pcout << bilayer;
    }
    
    template<int dim, int degree, typename Scalar>
    ComputeResolvent<dim,degree,Scalar>::~ComputeResolvent()
    {
        Teuchos::TimeMonitor::summarize (this->mpi_communicator(), pcout);
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeResolvent<dim,degree,Scalar>::run()
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
    types::MemUsage
    ComputeResolvent<dim,degree,Scalar>::memory_consumption() const
    {
        types::MemUsage memory = this->dof_handler.memory_consumption();
        memory.Static += sizeof(this);
        memory.Static += sizeof(I) + sizeof(R);
        memory.Static += sizeof( this->hamiltonian_action ) + sizeof( this->adjoint_interpolant );

        memory.Vectors += 3 * sizeof(Scalar) * ( I.at(0).getNumVectors() * I.at(0).getGlobalLength()
                                               + I.at(1).getNumVectors() * I.at(1).getGlobalLength() );
        memory.Matrices += (sizeof(Scalar) + sizeof(typename Matrix::global_ordinal_type) ) 
                            * ( this->hamiltonian_action.at(0)->getNodeNumEntries() + this->hamiltonian_action.at(1)->getNodeNumEntries() );
        memory.Matrices += (sizeof(Scalar) + sizeof(typename Matrix::global_ordinal_type) ) 
                            * ( this->adjoint_interpolant.at(0).at(0)->getNodeNumEntries() + this->adjoint_interpolant.at(1).at(0)->getNodeNumEntries() 
                               +this->adjoint_interpolant.at(0).at(1)->getNodeNumEntries() + this->adjoint_interpolant.at(1).at(1)->getNodeNumEntries());

        return memory;
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeResolvent<dim,degree,Scalar>::setup()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Setup<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );
        LA::base_setup();
        
        I = {{ MultiVector( this->dof_handler.locally_owned_dofs(0), this->dof_handler.n_range_orbitals(0,0) ), 
                MultiVector( this->dof_handler.locally_owned_dofs(1), this->dof_handler.n_range_orbitals(1,1) ) }};
        R  = {{ Tpetra::createCopy(I[0]),  Tpetra::createCopy(I[1]) }};
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeResolvent<dim,degree,Scalar>::assemble_matrices()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Assembly<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );

        LA::assemble_hamiltonian_action();
        LA::assemble_adjoint_interpolant();
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeResolvent<dim,degree,Scalar>::solve()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
        "Solve<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
        );

        /* Initialization of the vector to the identity */
        LA::create_identity(I);


        // Make an empty new parameter list.
        Teuchos::RCP<Teuchos::ParameterList> 
        solverParams = Teuchos::parameterList();
         // Set some GMRES parameters.
        //
        // "Num Blocks" = Maximum number of Krylov vectors to store.  This
        // is also the restart length.  "Block" here refers to the ability
        // of this particular solver (and many other Belos solvers) to solve
        // multiple linear systems at a time, even though we may only be
        // solving one linear system in this example.
        //
        // "Maximum Iterations": Maximum total number of iterations,
        // including restarts.
        //
        // "Convergence Tolerance": By default, this is the relative
        // residual 2-norm, although you can change the meaning of the
        // convergence tolerance using other parameters.
        solverParams->set ("Num Blocks", 40);
        solverParams->set ("Maximum Iterations", 400);
        solverParams->set ("Convergence Tolerance", 1.0e-6);

        Belos::SolverFactory<Scalar,MultiVector,Operator> factory;
        Teuchos::RCP<Belos::SolverManager<Scalar, MultiVector, Operator> > 
        solver = factory.create ("GMRES", solverParams);

        typedef Belos::LinearProblem<Scalar, MultiVector, Operator> problem_type;

        for (int b = 0; b < 2; ++b)
        {
            Teuchos::RCP<problem_type> 
            problem = Teuchos::rcp (new problem_type (this->hamiltonian_action.at(b), Teuchos::rcpFromRef (R.at(b)), Teuchos::rcpFromRef (I.at(b))));
            // You don't have to call this if you don't have a preconditioner.
            // If M is null, then Belos won't use a (right) preconditioner.
            // problem->setRightPrec (M);
            // Tell the LinearProblem to make itself ready to solve.
            problem->setProblem ();

            // Tell the solver what problem you want to solve.
            solver->setProblem (problem);

            // Attempt to solve the linear system.  result == Belos::Converged 
            // means that it was solved to the desired tolerance.  This call 
            // overwrites X with the computed approximate solution.
            Belos::ReturnType result = solver->solve();
        }
        
    }

    template<int dim, int degree, typename Scalar>
    void
    ComputeResolvent<dim,degree,Scalar>::write_to_file()
    {
        for (int b = 0; b<2; ++b)
        {
            std::ofstream output_file(this->dof_handler.output_file + "." + std::to_string(b) + "." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
            int 
            n_vectors = 2,
            M = H.at(b).getLocalLength(), 
            N = H.at(b).getNumVectors();
            output_file.write((char*) &n_vectors, sizeof(int));
            output_file.write((char*) &M, sizeof(int));
            output_file.write((char*) &N, sizeof(int));

            auto H_View = H.at(b).template getLocalView<Kokkos::Serial>();

            for (int j = 0; j<N; ++j)
                for (int i = 0; i < M; ++i)
                    output_file.write((char*) &H_View(i,j) , sizeof(Scalar));             

            auto R_View = R.at(b).template getLocalView<Kokkos::Serial>();

            for (int j = 0; j<N; ++j)
                for (int i = 0; i < M; ++i)
                    output_file.write((char*) &R_View(i,j) , sizeof(Scalar));    
        
            output_file.close();           
        }


        LA::linear_combination(-1, H, 1, R);
        for (int b = 0; b<2; ++b)
        {
            Teuchos::Array<double> norms (R.at(b) .getNumVectors());
            R .at(b).normInf(norms);
            pcout << std::accumulate(norms.begin(), norms.end(), 0.) << std::endl;
        }
    }


    /**
     * Explicit instantiations
     */
     template class ComputeResolvent<1,1,double>;
     template class ComputeResolvent<1,2,double>;
     template class ComputeResolvent<1,3,double>;
     template class ComputeResolvent<2,1,double>;
     template class ComputeResolvent<2,2,double>;
     template class ComputeResolvent<2,3,double>;

     template class ComputeResolvent<1,1,std::complex<double> >;
     template class ComputeResolvent<1,2,std::complex<double> >;
     template class ComputeResolvent<1,3,std::complex<double> >;
     template class ComputeResolvent<2,1,std::complex<double> >;
     template class ComputeResolvent<2,2,std::complex<double> >;
     template class ComputeResolvent<2,3,std::complex<double> >;
}/* End namespace Bilayer */
