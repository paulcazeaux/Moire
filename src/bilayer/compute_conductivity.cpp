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
            const double tau)
        :
        BaseAlgebra<dim,degree,Scalar, Node>(bilayer),
        pcout(( this->mpi_communicator->getRank () == 0) ? std::cout : blackHole),
        tau(tau)
    {
        pcout << bilayer;
    }

    template<int dim, int degree, class Node>
    ComputeConductivity<dim,degree,Node>::~ComputeConductivity()
    {}


    template<int dim, int degree, class Node>
    bool
    ComputeConductivity<dim,degree,Node>::run(bool verbose)
    {
        pcout   << "Starting setup...\n";
        setup();
        pcout   << "\tComplete!\n";


        pcout   << "Starting solve...\n";
        bool success = solve(verbose);
        pcout   << "\tComplete!\n";

        return success;
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
        liouvillianOp->constInitialize(vectorSpace, this->make_liouvillian_operator(static_cast<Scalar>(1./tau), z));
        derivationOp->constInitialize(vectorSpace, this->Derivation);

        hamiltonianOp->apply(Thyra::NOTRANS, *Tp, T .ptr(), 1.0, 0.0);
        derivationOp->apply(Thyra::NOTRANS, *T, dH.ptr(), 1.0, 0.0);
    }


    template<int dim, int degree, class Node>
    bool
    ComputeConductivity<dim,degree,Node>::solve(bool verbose)
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Solve<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );

        /* Solve the superoperator linear system */
        int frequency = 50;       // frequency of status test output.
        int blocksize = 1;         // blocksize
        int numrhs = 1;          // number of right-hand sides to solve for
        int maxiters = 10000;        // maximum number of iterations allowed per linear system
        int maxsubspace = 50;      // maximum number of blocks the solver can use for the subspace
        int maxrestarts = 200;      // number of restarts allowed
        Teuchos::ScalarTraits<Scalar>::magnitudeType
        tol = 1.0e-6;              // relative residual tolerance

        if (!verbose)
            frequency = -1;  // reset frequency if run is not verbose

        /********Other information used by block solver***********/
        const int NumGlobalElements = this->dof_handler.n_dofs() 
                * (this->dof_handler.n_orbitals(0) + this->dof_handler.n_orbitals(1));
        if (maxiters == -1)
            maxiters = NumGlobalElements - 1; // maximum number of iterations to run

        Teuchos::ParameterList belosList;
        belosList.set( "Num Blocks", maxsubspace);             // Maximum number of blocks in Krylov factorization
        belosList.set( "Block Size", blocksize );              // Blocksize to be used by iterative solver
        belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
        belosList.set( "Maximum Restarts", maxrestarts );      // Maximum number of restarts allowed
        belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
        if (verbose) 
        {
            belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
            Belos::TimingDetails + Belos::StatusTestDetails );
            if (frequency > 0)
                belosList.set( "Output Frequency", frequency );
        }
        else
            belosList.set( "Verbosity", Belos::Errors + Belos::Warnings );
        /* Construct an unpreconditioned linear problem instance. */
        Belos::LinearProblem<Scalar,Thyra::MultiVectorBase<Scalar>,Thyra::LinearOpBase<Scalar>> 
        problem( liouvillianOp, LinvdH, dH );
        bool set = problem.setProblem();
        if (set == false)
            throw std::logic_error("ERROR:  Belos::LinearProblem failed to set up correctly!");
        /************* Create an iterative solver manager. *********************/ 
        Teuchos::RCP< Belos::SolverManager<Scalar,Thyra::MultiVectorBase<Scalar>,Thyra::LinearOpBase<Scalar>> > newSolver
        = Teuchos::rcp( new Belos::PseudoBlockGmresSolMgr<Scalar,Thyra::MultiVectorBase<Scalar>,Thyra::LinearOpBase<Scalar>>(
                                Teuchos::rcp(&problem,false), 
                                Teuchos::rcp(&belosList,false)) );

        /************ Print out information about problem. **********************/
        if (verbose) 
        {
            pcout << std::endl << std::endl;
            pcout << "Dimension of matrix: " << NumGlobalElements << std::endl;
            pcout << "Number of right-hand sides: " << numrhs << std::endl;
            pcout << "Block size used by solver: " << blocksize << std::endl;
            pcout << "Max number of restarts allowed: " << maxrestarts << std::endl;
            pcout << "Max number of Gmres iterations per restart cycle: " << maxiters << std::endl;
            pcout << "Relative residual tolerance: " << tol << std::endl;
            pcout << std::endl;
        }
        /*************************** Perform solve. *****************************/
        Belos::ReturnType ret = newSolver->solve();

        /************** Output information about solve status. ******************/
        bool success;
        if (ret!=Belos::Converged) 
        {
            success = false;
            if (verbose)
                pcout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
        }
        else 
        {
            success = true;
            if (verbose)
                pcout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
        }

        /* Perform Chebyshev iteration and compute moments of the conductivity. */
        Scalar
        one = static_cast<Scalar>(1.0),
        alpha = this->dof_handler.energy_shift / this->dof_handler.energy_rescale,
        beta = one / this->dof_handler.energy_rescale;

        T->linear_combination(Teuchos::tuple(alpha),  
                              Teuchos::tuple<Teuchos::Ptr<const Thyra::VectorBase<Scalar>>>(Tp.ptr()), beta); // T := alpha * I + beta * H

        this->storeMoments(Tp, 0);
        this->storeMoments(T, 1);

        for (int i=2; i <= this->dof_handler.poly_degree; ++i)
        {
            hamiltonianOp->apply(Thyra::NOTRANS, *T, Tn.ptr(), 1.0, 0.0 );
            Tn->linear_combination(  Teuchos::tuple(-one, 2.*alpha), 
                Teuchos::tuple<Teuchos::Ptr<const Thyra::VectorBase<Scalar>>>(Tp.ptr(), T.ptr()), 
                2.*beta); // Tn :=  - Tp + 2*alpha*T + 2*beta*Tn

            this->storeMoments(Tn, i);
            std::swap(Tn, Tp);
            std::swap(T, Tp);
        }


        return success;
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