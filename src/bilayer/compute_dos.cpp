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