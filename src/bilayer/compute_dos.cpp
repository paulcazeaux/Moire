/* 
* File:   bilayer/compute_dos.h
* Author: Paul Cazeaux
*
* Created on June 30, 2017, 9:00 PM
*/


#include "bilayer/compute_dos.h"

namespace Bilayer {

    template<int dim, int degree, typename Scalar>
    ComputeDoS<dim,degree,Scalar>::ComputeDoS(const Multilayer<dim, 2>& bilayer)
        :
        BaseAlgebra<dim,degree,Scalar>(bilayer),
        pcout(( this->mpi_communicator->getRank () == 0) ? std::cout : blackHole)
    {
        pcout << bilayer;
    }

    template<int dim, int degree, typename Scalar>
    ComputeDoS<dim,degree,Scalar>::~ComputeDoS()
    {
        Teuchos::TimeMonitor::summarize (this->mpi_communicator(), pcout);
    }


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
    void 
    ComputeDoS<dim,degree,Scalar>::write_LDoS_to_file()
    {
        if (this->mpi_communicator->getRank () == 0)
        {
            std::ofstream output_file(this->dof_handler.output_file, std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
            int M = chebyshev_moments.size();
            int N = chebyshev_moments.at(0).at(0).size() + chebyshev_moments.at(0).at(1).size();

            output_file.write((char*) &M, sizeof(int));
            output_file.write((char*) &N, sizeof(int));
            for (const auto & moment : chebyshev_moments)
            {
                for (const Scalar m : moment.at(0))
                    output_file.write((char*) & m, sizeof(Scalar));
                for (const Scalar m : moment.at(1))
                    output_file.write((char*) & m, sizeof(Scalar));
            }
            output_file.close();
        }
    }

    template<int dim, int degree, typename Scalar>
    void 
    ComputeDoS<dim,degree,Scalar>::write_DoS_to_file()
    {
        if (this->mpi_communicator->getRank () == 0)
        {
            std::ofstream output_file(this->dof_handler.output_file, std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
            int M = chebyshev_moments.size();
            output_file.write((char*) &M, sizeof(int));

            for ( const auto diag_moments : chebyshev_moments)
            {
                Scalar m = std::accumulate(diag_moments.at(0).begin(), diag_moments.at(0).end(), static_cast<Scalar>(0.0))
                                        * this->unit_cell(1).area / (this->unit_cell(0).area + this->unit_cell(1).area)
                                        / static_cast<double>( this->dof_handler.n_domain_orbitals(0,0) * this->dof_handler.n_cell_nodes(0,0) )
                            + std::accumulate(diag_moments.at(1).begin(), diag_moments.at(1).end(), static_cast<Scalar>(0.0))
                                        * this->unit_cell(0).area / (this->unit_cell(0).area + this->unit_cell(1).area)
                                        / static_cast<double>( this->dof_handler.n_domain_orbitals(1,1) * this->dof_handler.n_cell_nodes(1,1) );
                output_file.write((char*) & m, sizeof(Scalar));
            }
        }
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
            Scalar T = std::accumulate(diag_moments.at(0).begin(), diag_moments.at(0).end(), static_cast<Scalar>(0.0))
                                    * this->unit_cell(1).area / (this->unit_cell(0).area + this->unit_cell(1).area)
                                    / static_cast<double>( this->dof_handler.n_domain_orbitals(0,0) * this->dof_handler.n_cell_nodes(0,0) )
                        + std::accumulate(diag_moments.at(1).begin(), diag_moments.at(1).end(), static_cast<Scalar>(0.0))
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
        memory.Matrices += (sizeof(Scalar) + sizeof(typename Matrix::global_ordinal_type) ) 
                            * ( this->hamiltonian_action.at(0)->getNodeNumEntries() + this->hamiltonian_action.at(1)->getNodeNumEntries() );

        return memory;
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeDoS<dim,degree,Scalar>::setup()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Setup<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );
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
        TEUCHOS_FUNC_TIME_MONITOR(
            "Assembly<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );

        LA::assemble_hamiltonian_action();
    }


    template<int dim, int degree, typename Scalar>
    void
    ComputeDoS<dim,degree,Scalar>::solve()
    {
        TEUCHOS_FUNC_TIME_MONITOR(
            "Solve<" << Teuchos::ScalarTraits<Scalar>::name () << ">()"
            );

        /* Initialization of the vector to the identity */

        LA::create_identity(Tp);
        LA::hamiltonian_rproduct(Tp, T);
        Scalar 
        alpha = this->dof_handler.energy_shift / this->dof_handler.energy_rescale,
        beta = 1. / this->dof_handler.energy_rescale;
        LA::linear_combination(alpha, Tp, beta, T);

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
            LA::linear_combination(-1, Tp, 2.*alpha, T, 2.*beta, Tn);

            std::swap(Tn, Tp);
            std::swap(T, Tp);

            m = LA::diagonal(T);
            if (this->dof_handler.my_pid == 0)
                chebyshev_moments.push_back(m);
        }
    }

    /**
     * Explicit instantiations
     */
     template class ComputeDoS<1,1,double>;
     template class ComputeDoS<1,2,double>;
     template class ComputeDoS<1,3,double>;
     template class ComputeDoS<2,1,double>;
     template class ComputeDoS<2,2,double>;
     template class ComputeDoS<2,3,double>;

     template class ComputeDoS<1,1,std::complex<double> >;
     template class ComputeDoS<1,2,std::complex<double> >;
     template class ComputeDoS<1,3,std::complex<double> >;
     template class ComputeDoS<2,1,std::complex<double> >;
     template class ComputeDoS<2,2,std::complex<double> >;
     template class ComputeDoS<2,3,std::complex<double> >;
}/* End namespace Bilayer */