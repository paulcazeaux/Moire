/* 
* File:   bilayer/compute_dos.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/



#ifndef moire__bilayer_computedos_h
#define moire__bilayer_computedos_h

#include "bilayer/base_algebra.h"

/**
* This class encapsulates the DoS computation of a discretized Hamiltonian encoded by a C* algebra.
*/



namespace Bilayer {

template <int dim, int degree>
class ComputeDoS : private BaseAlgebra<dim, degree>
{
public:
    ComputeDoS(const Multilayer<dim, 2>& bilayer);
    ~ComputeDoS();
    void                        run();
    std::vector<PetscScalar>    output_results();

private:
    void                setup();
    void                assemble_matrices();
    void                solve();

    /* MPI utilities */
    dealii::ConditionalOStream  pcout;
    dealii::TimerOutput         computing_timer;

    /* PETSc vectors to hold information about three observables in successive Chebyshev recursion steps */
    Vec                         Tp, T, Tn;

    /* Chebyshev DoS moments to be computed */
    std::vector<PetscScalar>    chebyshev_moments;
};


template<int dim, int degree>
void
ComputeDoS<dim,degree>::solve()
{
    dealii::TimerOutput::Scope t(computing_timer, "Solve");

    /* Initialization of the vector to the identity */

    BaseAlgebra<dim,degree>::create_identity(Tp);
    MatMult(BaseAlgebra<dim,degree>::hamiltonian_action, Tp, T);
    PetscScalar 
    m0 = BaseAlgebra<dim,degree>::trace(Tp), 
    m = BaseAlgebra<dim,degree>::trace(T);

    if (this->dof_handler.my_pid == 0)
    {
      chebyshev_moments.reserve(this->dof_handler.poly_degree+1);
      chebyshev_moments.push_back(m0);
      chebyshev_moments.push_back(m);
    }

    for (unsigned int i=2; i <= this->dof_handler.poly_degree; ++i)
    {
        MatMult(BaseAlgebra<dim,degree>::hamiltonian_action, T, Tn);
        VecAXPBY(Tn, -1., 2., Tp);
        VecSwap(Tn, Tp);
        VecSwap(T, Tp);
        m = BaseAlgebra<dim,degree>::trace(T);
        if (this->dof_handler.my_pid == 0)
            chebyshev_moments.push_back(m);
    }
}

template<int dim, int degree>
std::vector<PetscScalar>
ComputeDoS<dim,degree>::output_results()
{
    return chebyshev_moments;
}


template<int dim, int degree>
ComputeDoS<dim,degree>::ComputeDoS(const Multilayer<dim, 2>& bilayer)
    :
    BaseAlgebra<dim, degree>(bilayer),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)),
    computing_timer (this->mpi_communicator,
                   pcout,
                   dealii::TimerOutput::summary,
                   dealii::TimerOutput::wall_times)
    
{
    dealii::TimerOutput::Scope t(computing_timer, "Initialization");

    VecCreate(this->mpi_communicator, &T);
    VecCreate(this->mpi_communicator, &Tp);
    VecCreate(this->mpi_communicator, &Tn);
}

template<int dim, int degree>
ComputeDoS<dim,degree>::~ComputeDoS()
{
    VecDestroy(&T);
    VecDestroy(&Tp);
    VecDestroy(&Tn);
}


template<int dim, int degree>
void
ComputeDoS<dim,degree>::run()
{
    setup();
    assemble_matrices();
    solve();
}


template<int dim, int degree>
void
ComputeDoS<dim,degree>::setup()
{
    dealii::TimerOutput::Scope t(computing_timer, "Setup");

    VecDestroy(&T);
    VecDestroy(&Tp);
    VecDestroy(&Tn);

    BaseAlgebra<dim, degree>::base_setup();

    PetscErrorCode ierr = VecCreateMPI (this->mpi_communicator, this->locally_owned_dofs.n_elements(), PETSC_DETERMINE, &T);
    AssertThrow (ierr == 0, dealii::ExcPETScError(ierr));
    PetscInt sz;
    VecGetSize (T, &sz);
    Assert (static_cast<unsigned int>(sz) == this->dof_handler.n_dofs(), dealii::ExcDimensionMismatch (sz,  this->dof_handler.n_dofs()));

    ierr = VecDuplicate(T, &Tp);
    ierr = VecDuplicate(T, &Tn);
    AssertThrow (ierr == 0, dealii::ExcPETScError(ierr));
}




template<int dim, int degree>
void
ComputeDoS<dim,degree>::assemble_matrices()
{
    dealii::TimerOutput::Scope t(computing_timer, "Assembly");

    BaseAlgebra<dim,degree>::assemble_base_matrices();

    MatShift(this->hamiltonian_action, this->dof_handler.energy_shift);
    MatScale(this->hamiltonian_action, 1./this->dof_handler.energy_rescale);
}

}/* End namespace Bilayer */
#endif
