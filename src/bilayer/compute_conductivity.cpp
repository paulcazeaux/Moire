/* 
* File:   bilayer/compute_conductivity.h
* Author: Paul Cazeaux
*
* Created on Feb 24, 2019, 4:00 PM
*/


#include "bilayer/compute_conductivity.h"

namespace Bilayer {

    template<int dim, int degree, typename Scalar, class Node>
    ComputeConductivity<dim,degree,Scalar,Node>::ComputeConductivity(
            const Multilayer<dim, 2>& bilayer,
            const Scalar tau,
            const Scalar beta)
        :
        BaseAlgebra<dim,degree,Scalar,Node>(bilayer),
        pcout(( this->mpi_communicator->getRank () == 0) ? std::cout : blackHole),
        tau(tau), beta(beta)
    {
        pcout << bilayer;
    }

    template<int dim, int degree, typename Scalar, class Node>
    ComputeConductivity<dim,degree,Scalar,Node>::~ComputeConductivity()
    {
        Teuchos::TimeMonitor::summarize (this->mpi_communicator(), pcout);
    }


    template<int dim, int degree, typename Scalar, class Node>
    void
    ComputeConductivity<dim,degree,Scalar,Node>::run()
    {
        pcout   << "Starting setup...\n";
        setup();
        pcout   << "\tComplete!\n";


        pcout   << "Starting solve...\n";
        solve();
        pcout   << "\tComplete!\n";
    }

    template<int dim, int degree, typename Scalar, class Node>
    void 
    ComputeConductivity<dim,degree,Scalar,Node>::write_Conductivity_to_file()
    {
        if (this->mpi_communicator->getRank () == 0)
        {
            std::ofstream output_file(this->dof_handler.output_file, std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
            int M = chebyshev_moments.size();

            output_file.write((char*) &M, sizeof(int));
            for (const auto & moment : chebyshev_moments)
                for (const Scalar m : moment)
                    output_file.write((char*) & m, sizeof(Scalar));
            output_file.close();
        }
    }

    
    template<int dim, int degree, typename Scalar, class Node>
    std::vector<std::array<Scalar,dim>>
    ComputeConductivity<dim,degree,Scalar,Node>::output_Conductivity()
    {
        return chebyshev_moments;
    }



    template<int dim, int degree, typename Scalar, class Node>
    void
    ComputeConductivity<dim,degree,Scalar,Node>::setup()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Setup<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );

        this->assemble_base_matrices();

        Teuchos::RCP<VS> vs = VectorSpace<dim,degree,Scalar,Node>::create();
        vs->initialize(Teuchos::rcpFromRef<BaseAlgebra<dim,degree,Scalar,Node>>(* this));
        vectorSpace = vs;


        /* 
         * Initialization of the vector I to the identity, 
         * H to the Hamiltonian observable,
         * dH to the derivations of the Hamiltonian observable
         */
        Teuchos::RCP<Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> >
            I_tpetra = Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getOrbVecMap(), vectorSpace->getNumOrbitals() );
        LA::set_to_identity(* I_tpetra);
        
        I = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
            I_tpetra);
        H = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
        Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
        Tp = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
            Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
        T  = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
            Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
        Tn = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
            Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));

        dH = Bilayer::createMultiVector<dim,degree,Scalar,Node>
        ( vectorSpace, 
        Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim) );
        LinvdH = Bilayer::createMultiVector<dim,degree,Scalar,Node>
        ( vectorSpace, 
        Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim) );


        std::complex<double> z (0, 1.);
        hamiltonianOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
            ( vectorSpace, this->HamiltonianAction);
        transposeOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
            ( vectorSpace, this->Transpose);
        liouvillianOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
            (vectorSpace, this->make_liouvillian_operator(tau, z));
        derivationOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
            (vectorSpace, this->Derivation);


        hamiltonianOp->apply(Thyra::NOTRANS, *I, H .ptr(), 1.0, 0.0);
        derivationOp ->apply(Thyra::NOTRANS, *H, dH.ptr(), 1.0, 0.0);
    }


    template<int dim, int degree, typename Scalar, class Node>
    void
    ComputeConductivity<dim,degree,Scalar,Node>::solve()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Solve<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );


        /* Solve the superoperator linear system */

        Teuchos::RCP<Teuchos::ParameterList> 
            solverParams = Teuchos::parameterList();
            solverParams->set ("Solver Type","GMRES");

        Teuchos::ParameterList& 
            solverParams_solver = solverParams->sublist("Solver Types");

        Teuchos::ParameterList& 
            solverParams_gmres = solverParams_solver.sublist("GMRES");

        solverParams_gmres.set("Maximum Iterations", 400);
        solverParams_gmres.set("Convergence Tolerance", 1.0e-6);
        solverParams_gmres.set("Maximum Restarts", 15);
        solverParams_gmres.set("Num Blocks", 40);
        solverParams_gmres.set("Output Frequency", 100);
        solverParams_gmres.set("Show Maximum Residual Norm Only", true);

        Teuchos::RCP<Thyra::LinearOpWithSolveFactoryBase<Scalar> >
            belosFactory = Teuchos::rcp(new Thyra::BelosLinearOpWithSolveFactory<Scalar>());
        belosFactory->setParameterList( solverParams );
        belosFactory->setVerbLevel(Teuchos::VERB_LOW);

        Teuchos::RCP<Thyra::LinearOpWithSolveBase<Scalar> >
            bA = belosFactory->createOp();

        Thyra::initializeOp<Scalar>( * belosFactory, liouvillianOp, bA.ptr() );
        Thyra::SolveStatus<Scalar> solveStatus;
        solveStatus = Thyra::solve( *bA, Thyra::NOTRANS, *dH, LinvdH.ptr() );

        pcout << "\nBelos Solve Status: "<< solveStatus << std::endl;


        // Scalar 
        // alpha = this->dof_handler.energy_shift / this->dof_handler.energy_rescale,
        // beta = 1. / this->dof_handler.energy_rescale;

        // T.update(alpha, Tp, beta); // T := alpha * Tp + beta * T

        // std::array<std::vector<Scalar>,2> 
        // m0 = LA::diagonal(Tp), 
        // m = LA::diagonal(T);

        // if (this->dof_handler.my_pid == 0)
        // {
        //   chebyshev_moments.push_back(m0);
        //   chebyshev_moments.push_back(m);
        // }

        // for (int i=2; i <= this->dof_handler.poly_degree; ++i)
        // {
        //     LA::HamiltonianAction->apply(T, Tn);
        //     Tn.update(-1, Tp, 2.*alpha, T, 2.*beta); // Tn := 2*alpha*T - Tp + 2*beta*Tn

        //     std::swap(Tn, Tp);
        //     std::swap(T, Tp);

        //     m = LA::diagonal(T);
        //     if (this->dof_handler.my_pid == 0)
        //         chebyshev_moments.push_back(m);
        // }
    }

    /**
     * Explicit instantiations
     */

     template class ComputeConductivity<1,1,std::complex<double> >;
     template class ComputeConductivity<1,2,std::complex<double> >;
     template class ComputeConductivity<1,3,std::complex<double> >;
     template class ComputeConductivity<2,1,std::complex<double> >;
     template class ComputeConductivity<2,2,std::complex<double> >;
     template class ComputeConductivity<2,3,std::complex<double> >;
}/* End namespace Bilayer */