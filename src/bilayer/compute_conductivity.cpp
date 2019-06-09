/* 
* File:   bilayer/compute_conductivity.h
* Author: Paul Cazeaux
*
* Created on Feb 24, 2019, 4:00 PM
*/


#include "bilayer/compute_conductivity.h"

namespace Bilayer {

    template<int dim, int degree, class Node>
    ComputeConductivity<dim,degree,Node>::ComputeConductivity(
            const Multilayer<dim, 2>& bilayer,
            const Scalar tau,
            const Scalar beta)
        :
        BaseAlgebra<dim,degree,Scalar, Node>(bilayer),
        pcout(( this->mpi_communicator->getRank () == 0) ? std::cout : blackHole),
        tau(tau), beta(beta)
    {
        pcout << bilayer;
    }

    template<int dim, int degree, class Node>
    ComputeConductivity<dim,degree,Node>::~ComputeConductivity()
    {
        Teuchos::TimeMonitor::summarize (this->mpi_communicator(), pcout);
    }


    template<int dim, int degree, class Node>
    void
    ComputeConductivity<dim,degree,Node>::run()
    {
        pcout   << "Starting setup...\n";
        setup();
        pcout   << "\tComplete!\n";


        pcout   << "Starting solve...\n";
        solve();
        pcout   << "\tComplete!\n";
    }

    template<int dim, int degree, class Node>
    void 
    ComputeConductivity<dim,degree,Node>::write_to_file()
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

    
    template<int dim, int degree, class Node>
    std::vector<std::array<typename ComputeConductivity<dim,degree,Node>::Scalar,dim*dim>>
    ComputeConductivity<dim,degree,Node>::output()
    {
        return chebyshev_moments;
    }



    template<int dim, int degree, class Node>
    void
    ComputeConductivity<dim,degree,Node>::setup()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Setup<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );

        chebyshev_moments.resize(this->dof_handler.poly_degree);

        this->assemble_base_matrices();
        Tp = Teuchos::rcp(new Vec());
        T = Teuchos::rcp(new Vec());
        Tn = Teuchos::rcp(new Vec());
        dH = Teuchos::rcp(new MVec());
        LinvdH = Teuchos::rcp(new MVec());
        hamiltonianOp = Teuchos::rcp(new Op());
        liouvillianOp = Teuchos::rcp(new Op());
        derivationOp = Teuchos::rcp(new Op());

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

        
        Tp->initialize( vectorSpace, I_tpetra);
        
        
        T->initialize( vectorSpace, 
            Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                                            ( vectorSpace->getVecMap() ));
        
        Tn->initialize( vectorSpace, 
            Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                                            ( vectorSpace->getVecMap() ));
        
        Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar>> 
        domainSpace = vectorSpace->createDomainVectorSpace(dim) ;
        dH->initialize( vectorSpace, domainSpace,
            Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim));
        
        LinvdH->initialize( vectorSpace, domainSpace,
            Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim));


        Scalar z (0, 1.);
        hamiltonianOp->constInitialize( vectorSpace, this->HamiltonianAction);
        liouvillianOp->constInitialize(vectorSpace, this->make_liouvillian_operator(tau, z));
        derivationOp->constInitialize(vectorSpace, this->Derivation);

        hamiltonianOp->apply(Thyra::NOTRANS, *Tp, T .ptr(), 1.0, 0.0);
        derivationOp->apply(Thyra::NOTRANS, *T, dH.ptr(), 1.0, 0.0);
    }


    template<int dim, int degree, class Node>
    void
    ComputeConductivity<dim,degree,Node>::solve()
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
        solveStatus = Thyra::solve( *bA, 
                                    Thyra::NOTRANS, 
                                    *dH, 
                                    Teuchos::ptr_implicit_cast<Thyra::MultiVectorBase<Scalar>>(LinvdH.ptr()) 
                                        );

        pcout << "\nBelos Solve Status: "<< solveStatus << std::endl;

        Scalar
        one = static_cast<Scalar>(1.0),
        alpha = this->dof_handler.energy_shift / this->dof_handler.energy_rescale,
        beta = one / this->dof_handler.energy_rescale;

        T->linear_combination(Teuchos::tuple(alpha), 
                             Teuchos::tuple(Teuchos::ptr_implicit_cast<const Thyra::VectorBase<Scalar>>(Tp.ptr())), 
                             beta); // T := alpha * I + beta * H

        this->storeMoments(Tp, 0);
        this->storeMoments(T, 1);

        for (int i=2; i <= this->dof_handler.poly_degree; ++i)
        {
            hamiltonianOp->apply(Thyra::NOTRANS, *T, Tn.ptr(), 1.0, 0.0 );
            Tn->linear_combination(  Teuchos::tuple(-one, 2.*alpha), 
                Teuchos::tuple( Teuchos::ptr_implicit_cast<const Thyra::VectorBase<Scalar>>(Tp.ptr()), 
                                Teuchos::ptr_implicit_cast<const Thyra::VectorBase<Scalar>>(T.ptr()) ), 
                2.*beta); // Tn :=  - Tp + 2*alpha*T + 2*beta*Tn

            this->storeMoments(Tn, i);
            std::swap(Tn, Tp);
            std::swap(T, Tp);
        }
    }

    template<int dim, int degree, class Node>
    void
    ComputeConductivity<dim,degree,Node>::storeMoments(RCP<Vec> A, int i)
    {
        Teuchos::ArrayView<Scalar> m_view (chebyshev_moments[i].data(), dim*dim);
        LA::Derivation->weightedDot(  * LinvdH->getConstThyraOrbMultiVector()->getConstTpetraMultiVector(), 
                                      * A->getConstThyraOrbVector()->getConstTpetraMultiVector(), 
                                        m_view );
    }

    /**
     * Explicit instantiations
     */

     template class ComputeConductivity<1,1,types::DefaultNode>;
     template class ComputeConductivity<1,2,types::DefaultNode>;
     template class ComputeConductivity<1,3,types::DefaultNode>;
     template class ComputeConductivity<2,1,types::DefaultNode>;
     template class ComputeConductivity<2,2,types::DefaultNode>;
     template class ComputeConductivity<2,3,types::DefaultNode>;
}/* End namespace Bilayer */