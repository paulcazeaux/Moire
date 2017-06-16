/* 
* File:   bilayer/base_algebra.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/



#ifndef moire__bilayer_base_algebra_h
#define moire__bilayer_base_algebra_h

#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <fstream>

#include <complex>
#include "fftw3.h"

#include "petscvec.h" 

#include "deal.II/base/mpi.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/index_set.h"
#include "deal.II/base/utilities.h"
#include "deal.II/base/conditional_ostream.h"
#include "deal.II/base/timer.h"
#include "deal.II/lac/generic_linear_algebra.h"

namespace LA
{
#if defined(DEAL_II_WITH_PETSC)
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#else
#  error DEAL_II_WITH_PETSC required
#endif
}


#include "deal.II/lac/dynamic_sparsity_pattern.h"
#include "deal.II/lac/petsc_parallel_sparse_matrix.h"
#include "deal.II/lac/petsc_parallel_vector.h"

#include "tools/numbers.h"
#include "bilayer/dof_handler.h"

/**
* This class encapsulates the basic operations in a C* algebra equipped with a discretized Hamiltonian.
* It is intended as a base class creating the necessary operations for use in further computations 
* (density of states, conductivity, etc.)
*/

namespace Bilayer {

template <int dim, int degree>
class BaseAlgebra
{
public:
    BaseAlgebra(const Multilayer<dim, 2>& bilayer);
    ~BaseAlgebra();

protected:
    void                base_setup();
    void                assemble_base_matrices();

    /* Assemble identity observable */
    void                create_identity(Vec& Result);
    /* Adjoint operation on an observable */
    void                adjoint(const Vec& A, Vec& Result);
    /* Trace of an observable, returned on root process (pid = 0, returns 0 on other processes) */
    PetscScalar         trace(const Vec& A);

    /* MPI communication environment and utilities */
    MPI_Comm                    mpi_communicator;

    /* DoF Handler object and local indices range */
    DoFHandler<dim,degree>      dof_handler;
    dealii::IndexSet            locally_owned_dofs;
    dealii::IndexSet            locally_relevant_dofs;

    /* Matrices representing the sparse linear action of the two main operations */
    LA::MPI::SparseMatrix       adjoint_action;
    LA::MPI::SparseMatrix       hamiltonian_action;

    /* Data structures allocated for additional local computations in the adjoint operation */
    fftw_plan   fplan_0, bplan_0, fplan_1, bplan_1;
    std::complex<double> * data_in_0, * data_out_0, * data_in_1, * data_out_1;
    std::array<unsigned int, 2>     point_size;

    /* Convenience functions */
    const LayerData<dim>&       layer(const int & idx)      const { return dof_handler.layer(idx); }
    const Lattice<dim>&         lattice(const int & idx)    const { return dof_handler.lattice(idx); };
    const UnitCell<dim,degree>& unit_cell(const int & idx)  const { return dof_handler.unit_cell(idx); };
};

template<int dim, int degree>
BaseAlgebra<dim,degree>::BaseAlgebra(const Multilayer<dim, 2>& bilayer)
    :
    mpi_communicator(MPI_COMM_WORLD),
    dof_handler(bilayer)
{
    point_size[0] = layer(0).n_orbitals * layer(0).n_orbitals * unit_cell(1).n_nodes;
    point_size[1] = layer(1).n_orbitals * layer(1).n_orbitals * unit_cell(0).n_nodes;
    /* Allocate auxiliary arrays for FFT computations */
    data_in_0   = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * point_size[0]);
    data_out_0  = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * point_size[0]);
    data_in_1   = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * point_size[1]);
    data_out_1  = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * point_size[1]);
    fplan_0 = fftw_plan_many_dft(dim, unit_cell(1).n_nodes_per_dim.data(), layer(0).n_orbitals * layer(0).n_orbitals, 
                                            reinterpret_cast<fftw_complex *>(data_in_0),  NULL, layer(0).n_orbitals * layer(0).n_orbitals, 1,
                                            reinterpret_cast<fftw_complex *>(data_out_0), NULL, layer(0).n_orbitals * layer(0).n_orbitals, 1, FFTW_FORWARD, FFTW_MEASURE);
    bplan_0 = fftw_plan_many_dft(dim, unit_cell(1).n_nodes_per_dim.data(), layer(0).n_orbitals * layer(0).n_orbitals, 
                                            reinterpret_cast<fftw_complex *>(data_out_0), NULL, layer(0).n_orbitals * layer(0).n_orbitals, 1,
                                            reinterpret_cast<fftw_complex *>(data_in_0),  NULL, layer(0).n_orbitals * layer(0).n_orbitals, 1, FFTW_BACKWARD, FFTW_MEASURE);
    fplan_1 = fftw_plan_many_dft(dim, unit_cell(0).n_nodes_per_dim.data(), layer(1).n_orbitals * layer(1).n_orbitals, 
                                            reinterpret_cast<fftw_complex *>(data_in_1),  NULL, layer(1).n_orbitals * layer(1).n_orbitals, 1,
                                            reinterpret_cast<fftw_complex *>(data_out_1), NULL, layer(1).n_orbitals * layer(1).n_orbitals, 1, FFTW_FORWARD, FFTW_MEASURE);
    bplan_1 = fftw_plan_many_dft(dim, unit_cell(0).n_nodes_per_dim.data(), layer(1).n_orbitals * layer(1).n_orbitals, 
                                            reinterpret_cast<fftw_complex *>(data_out_1), NULL, layer(1).n_orbitals * layer(1).n_orbitals, 1,
                                            reinterpret_cast<fftw_complex *>(data_in_1),  NULL, layer(1).n_orbitals * layer(1).n_orbitals, 1, FFTW_BACKWARD, FFTW_MEASURE);
}

template<int dim, int degree>
BaseAlgebra<dim,degree>::~BaseAlgebra()
{
    fftw_destroy_plan(fplan_0);
    fftw_destroy_plan(bplan_0);
    fftw_destroy_plan(fplan_1);
    fftw_destroy_plan(bplan_1);

    fftw_free(data_in_0);
    fftw_free(data_out_0);
    fftw_free(data_in_1);
    fftw_free(data_out_1);
}


template<int dim, int degree>
void
BaseAlgebra<dim,degree>::base_setup()
{
    dof_handler.initialize(mpi_communicator);
    locally_owned_dofs = dof_handler.locally_owned_dofs();
    locally_relevant_dofs = dof_handler.locally_relevant_dofs();

    Assert(locally_owned_dofs.is_contiguous(), dealii::ExcNotImplemented());

    dealii::DynamicSparsityPattern dsp (locally_relevant_dofs);
    dof_handler.make_sparsity_pattern_adjoint(dsp);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, 
                    dof_handler.n_locally_owned_dofs_per_processor(), 
                    mpi_communicator, locally_relevant_dofs);

    adjoint_action.reinit(  locally_owned_dofs, 
                                locally_owned_dofs, 
                                dsp, 
                                mpi_communicator);

    dsp.reinit(dof_handler.n_dofs(), dof_handler.n_dofs(),
                    locally_owned_dofs);
    dof_handler.make_sparsity_pattern_rmultiply(dsp);

    hamiltonian_action.reinit(  locally_owned_dofs, 
                                locally_owned_dofs, 
                                dsp, 
                                mpi_communicator);

}




template<int dim, int degree>
void
BaseAlgebra<dim,degree>::assemble_base_matrices()
{
  /* First we assemble the matrix needed for the adjoint operation */

    std::vector<double> interpolation_weights;
    for (unsigned int n=0; n<dof_handler.n_locally_owned_points(); ++n)
    {
        const PointData& this_point = dof_handler.locally_owned_point(n);
        switch (this_point.block_id) {
                            /***********************/
            case 0:         /* Row in block 0 -> 0 */
            {               /***********************/
                std::array<int, dim> on_grid = lattice(0).get_vertex_grid_indices(this_point.index_in_block);
                for (auto & c : on_grid)
                    c = -c;
                unsigned int neighbor_index_in_block = lattice(0).get_vertex_global_index(on_grid);

                for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
                    for (int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
                        for (int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
                            adjoint_action.set(dof_handler.get_dof_index(0, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                dof_handler.get_dof_index(0, neighbor_index_in_block, cell_index, orbital_column, orbital_row),
                                                1.);
                break;
            }               /***************************/
            case 1:         /* COLUMNS in block 1 -> 2 */
            {               /***************************/

                for (auto & interp_point : this_point.interpolated_nodes)
                {
                    unsigned int element_index = interp_point.first;
                    auto [ row_block_id, row_index_in_block, row_cell_index] = interp_point.second;
                    dealii::Point<dim> quadrature_point (- (lattice(1).get_vertex_position(this_point.index_in_block)
                                                            + lattice(0).get_vertex_position(row_index_in_block)
                                                            + unit_cell(0).get_node_position(row_cell_index)    ));

                    unit_cell(1).subcell_list[element_index].get_interpolation_weights(quadrature_point, interpolation_weights);
                    for (unsigned int j = 0; j < Element<dim,degree>::dofs_per_cell; ++j)
                    {
                        unsigned int cell_index = unit_cell(1).subcell_list[element_index].unit_cell_dof_index_map[j];
                        if (unit_cell(1).is_node_interior(cell_index))
                        {
                            for (int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
                                for (int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
                                    adjoint_action.set(dof_handler.get_dof_index(2, row_index_in_block, row_cell_index, orbital_column, orbital_row),
                                                        dof_handler.get_dof_index(1, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                            interpolation_weights.at(j));
                        }
                        else // Boundary point!
                        {
                            auto [column_block_id, column_index_in_block, column_cell_index] 
                                        = this_point.boundary_lattice_points[cell_index - unit_cell(1).n_nodes];
                            /* Last check to see if the boundary point actually exists on the grid. Maybe some extrapolation would be useful? */
                            if (column_index_in_block != types::invalid_lattice_index) 
                                for (int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
                                    for (int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
                                        adjoint_action.set(dof_handler.get_dof_index(2, row_index_in_block, row_cell_index, orbital_column, orbital_row),
                                                            dof_handler.get_dof_index(1, column_index_in_block, column_cell_index, orbital_row, orbital_column),
                                                                interpolation_weights.at(j));
                        }
                    }
                }

                break;
            }               /***************************/
            case 2:         /* COLUMNS in block 2 -> 1 */
            {               /***************************/
                for (auto & interp_point : this_point.interpolated_nodes)
                {
                    unsigned int element_index = interp_point.first;
                    auto [ row_block_id, row_index_in_block, row_cell_index] = interp_point.second;
                    dealii::Point<dim> quadrature_point (- (lattice(0).get_vertex_position(this_point.index_in_block)
                                                            + lattice(1).get_vertex_position(row_index_in_block)
                                                            + unit_cell(1).get_node_position(row_cell_index)    ));

                    unit_cell(0).subcell_list[element_index].get_interpolation_weights(quadrature_point, interpolation_weights);

                    for (unsigned int j = 0; j < Element<dim,degree>::dofs_per_cell; ++j)
                    {
                        unsigned int cell_index = unit_cell(0).subcell_list[element_index].unit_cell_dof_index_map[j];
                        if (unit_cell(0).is_node_interior(cell_index))
                            for (int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
                                for (int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
                                    adjoint_action.set(dof_handler.get_dof_index(1, row_index_in_block, row_cell_index, orbital_column, orbital_row),
                                                        dof_handler.get_dof_index(2, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                            interpolation_weights.at(j));
                        else // Boundary point!
                        {
                            auto [column_block_id, column_index_in_block, column_cell_index] = this_point.boundary_lattice_points[cell_index - unit_cell(0).n_nodes];
                            /* Last check to see if the boundary point exists on the grid. Maybe some extrapolation would be useful? */
                            if (column_index_in_block != types::invalid_lattice_index) 
                                for (int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
                                    for (int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
                                        adjoint_action.set(dof_handler.get_dof_index(1, row_index_in_block, row_cell_index, orbital_column, orbital_row),
                                                            dof_handler.get_dof_index(2, column_index_in_block, column_cell_index, orbital_row, orbital_column),
                                                                interpolation_weights.at(j));
                        }
                    }
                }
                break;
            }               /************************/
            case 3:         /* Rows in block 3 -> 3 */
            {               /************************/
                std::array<int, dim> on_grid = lattice(1).get_vertex_grid_indices(this_point.index_in_block);
                for (auto & c : on_grid)
                    c = -c;
                unsigned int opposite_index_in_block = lattice(1).get_vertex_global_index(on_grid);

                for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
                    for (int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
                        for (int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
                            adjoint_action.set(dof_handler.get_dof_index(3, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                dof_handler.get_dof_index(3, opposite_index_in_block, cell_index, orbital_column, orbital_row),
                                                1.);
                break;
            }
        }
    }

    adjoint_action.compress(LA::VectorOperation::insert);



    /* Finally the matrix needed for taking the product with the Hamiltonian */


    for (unsigned int n=0; n< dof_handler.n_locally_owned_points(); ++n)
    {
        const PointData& this_point = dof_handler.locally_owned_point(n);
        switch (this_point.block_id) 
        {                   /******************/
            case 0:         /* Row in block 0 */
            {               /******************/
                dealii::Point<dim> 
                this_point_position = lattice(0).get_vertex_position(this_point.index_in_block);
                std::array<int, dim> 
                this_point_grid_indices = lattice(0).get_vertex_grid_indices(this_point.index_in_block);

                    /* Block 0 <-> 0 */

                std::vector<unsigned int>   
                neighbors = lattice(0).list_neighborhood_indices(this_point_position, 
                                            layer(0).intra_search_radius);
                for (auto neighbor_index_in_block : neighbors)
                {
                    std::array<int,dim> 
                    grid_vector = lattice(0).get_vertex_grid_indices(neighbor_index_in_block);
                    for (int j=0; j<dim; ++j)
                        grid_vector[j] -= this_point_grid_indices[j];

                    for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
                        for (int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
                            for (int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
                                for (int orbital_middle = 0; orbital_middle < layer(0).n_orbitals; orbital_middle++)
                                    hamiltonian_action.set(
                                        dof_handler.get_dof_index(0, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                        dof_handler.get_dof_index(0, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                        dof_handler.intralayer_term(orbital_middle, orbital_column, grid_vector, 0 ));
                }
                    


                    /* Block 1 -> 0 */
                neighbors = lattice(1).list_neighborhood_indices( this_point_position, 
                                            dof_handler.inter_search_radius + unit_cell(1).bounding_radius);
                
                for (auto neighbor_index_in_block : neighbors)
                    for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
                    {
                        dealii::Tensor<1,dim> arrow_vector = lattice(1).get_vertex_position(neighbor_index_in_block)
                                                            + unit_cell(1).get_node_position(cell_index)
                                                             - this_point_position;

                        if ( arrow_vector.norm() < dof_handler.inter_search_radius)
                            for (int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
                                for (int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
                                        for (int orbital_middle = 0; orbital_middle < layer(1).n_orbitals; orbital_middle++)
                                            hamiltonian_action.set(
                                                dof_handler.get_dof_index(0, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                dof_handler.get_dof_index(1, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                                dof_handler.interlayer_term(orbital_middle, orbital_column, arrow_vector, 1, 0 ));
                    }
                break;
            }               /******************/
            case 1:         /* Row in block 1 */
            {               /******************/
                dealii::Point<dim> 
                this_point_position = lattice(1).get_vertex_position(this_point.index_in_block);
                std::array<int, dim> 
                this_point_grid_indices = lattice(1).get_vertex_grid_indices(this_point.index_in_block);

                    /* Block 0 -> 1 */
                std::vector<unsigned int>   
                neighbors = lattice(0).list_neighborhood_indices( this_point_position, 
                                            dof_handler.inter_search_radius + unit_cell(1).bounding_radius);

                for (auto neighbor_index_in_block : neighbors)
                    for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
                    {
                        dealii::Tensor<1,dim> arrow_vector = lattice(0).get_vertex_position(neighbor_index_in_block)
                                                            - unit_cell(1).get_node_position(cell_index)
                                                             - this_point_position;

                        if ( arrow_vector.norm() < dof_handler.inter_search_radius )
                            for (int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
                                for (int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
                                        for (int orbital_middle = 0; orbital_middle < layer(0).n_orbitals; orbital_middle++)
                                            hamiltonian_action.set(
                                                dof_handler.get_dof_index(1, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                dof_handler.get_dof_index(0, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                                dof_handler.interlayer_term(orbital_middle, orbital_column, arrow_vector, 0, 1));
                    }

                    /* Block 1 <-> 1 */
                neighbors = lattice(1).list_neighborhood_indices(this_point_position, layer(1).intra_search_radius);

                for (auto neighbor_index_in_block : neighbors)
                {
                    std::array<int,dim> 
                    grid_vector = lattice(1).get_vertex_grid_indices(neighbor_index_in_block);
                    for (int j=0; j<dim; ++j)
                        grid_vector[j] -= this_point_grid_indices[j];

                    for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
                        for (int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
                            for (int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
                                for (int orbital_middle = 0; orbital_middle < layer(1).n_orbitals; orbital_middle++)
                                    hamiltonian_action.set(
                                        dof_handler.get_dof_index(1, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                        dof_handler.get_dof_index(1, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                        dof_handler.intralayer_term(orbital_middle, orbital_column, grid_vector, 1));
                }

                break;
            }               /*******************/
            case 2:         /* Rows in block 2 */
            {               /*******************/
                dealii::Point<dim> 
                this_point_position = lattice(0).get_vertex_position(this_point.index_in_block);
                std::array<int, dim> 
                this_point_grid_indices = lattice(0).get_vertex_grid_indices(this_point.index_in_block);

                        /* Block 3 -> 2 */
                std::vector<unsigned int>   
                neighbors = lattice(1).list_neighborhood_indices( this_point_position, 
                                            dof_handler.inter_search_radius + unit_cell(0).bounding_radius);

                for (auto neighbor_index_in_block : neighbors)
                    for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
                    {
                        dealii::Tensor<1,dim> arrow_vector = lattice(1).get_vertex_position(neighbor_index_in_block)
                                                            - unit_cell(0).get_node_position(cell_index)
                                                             - this_point_position;
                        if ( arrow_vector.norm() < dof_handler.inter_search_radius )
                            for (int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
                                for (int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
                                        for (int orbital_middle = 0; orbital_middle < layer(1).n_orbitals; orbital_middle++)
                                            hamiltonian_action.set(
                                                dof_handler.get_dof_index(2, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                dof_handler.get_dof_index(3, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                                dof_handler.interlayer_term(orbital_middle, orbital_column, arrow_vector, 1, 0));
                    }

                    /* Block 2 <-> 2 */
                neighbors = lattice(0).list_neighborhood_indices(this_point_position, layer(0).intra_search_radius);

                for (auto neighbor_index_in_block : neighbors)
                {
                    std::array<int,dim> 
                    grid_vector = lattice(0).get_vertex_grid_indices(neighbor_index_in_block);
                    for (int j=0; j<dim; ++j)
                        grid_vector[j] -= this_point_grid_indices[j];

                    for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
                        for (int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
                            for (int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
                                for (int orbital_middle = 0; orbital_middle < layer(0).n_orbitals; orbital_middle++)
                                    hamiltonian_action.set(
                                        dof_handler.get_dof_index(2, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                        dof_handler.get_dof_index(2, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                        dof_handler.intralayer_term(orbital_middle, orbital_column, grid_vector, 0 ));
                }

                break;
            }               /*******************/
            case 3:         /* Rows in block 3 */
            {               /*******************/
                dealii::Point<dim> 
                this_point_position = lattice(1).get_vertex_position(this_point.index_in_block);
                std::array<int, dim> 
                this_point_grid_indices = lattice(1).get_vertex_grid_indices(this_point.index_in_block);


                    /* Block 3 <-> 3 */
                std::vector<unsigned int>   
                neighbors = lattice(1).list_neighborhood_indices(this_point_position, 
                                            layer(1).intra_search_radius);
                for (auto neighbor_index_in_block : neighbors)
                {
                    std::array<int,dim> 
                    grid_vector = lattice(1).get_vertex_grid_indices(neighbor_index_in_block);
                    for (int j=0; j<dim; ++j)
                        grid_vector[j] -= this_point_grid_indices[j];

                    for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
                        for (int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
                            for (int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
                                for (int orbital_middle = 0; orbital_middle < layer(1).n_orbitals; orbital_middle++)
                                    hamiltonian_action.set(
                                        dof_handler.get_dof_index(3, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                        dof_handler.get_dof_index(3, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                        dof_handler.intralayer_term(orbital_middle,   orbital_column, grid_vector, 1 ));
                }

                    /* Block 2 -> 3 */
                neighbors = lattice(0).list_neighborhood_indices( this_point_position, 
                                            dof_handler.inter_search_radius + unit_cell(0).bounding_radius);
                
                for (auto neighbor_index_in_block : neighbors)
                    for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
                    {
                        dealii::Tensor<1,dim> arrow_vector = lattice(0).get_vertex_position(neighbor_index_in_block)
                                                            + unit_cell(0).get_node_position(cell_index)
                                                             - this_point_position;
                        if  ( arrow_vector.norm() < dof_handler.inter_search_radius )
                            for (int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
                                for (int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
                                        for (int orbital_middle = 0; orbital_middle < layer(0).n_orbitals; orbital_middle++)
                                            hamiltonian_action.set(
                                                dof_handler.get_dof_index(3, this_point.index_in_block, cell_index, orbital_row, orbital_column),
                                                dof_handler.get_dof_index(2, neighbor_index_in_block, cell_index, orbital_row, orbital_middle),
                                                dof_handler.interlayer_term(orbital_middle, orbital_column, arrow_vector, 0, 1));
                    }
                break;
            }
        }
    }

    hamiltonian_action.compress(LA::VectorOperation::insert);
}


template<int dim, int degree>
void
BaseAlgebra<dim,degree>::adjoint(const Vec&  A, Vec & Result)
{
    /* Check that the vectors have the same data distribution */
    const PetscInt * rangeA, * rangeResult;
    PetscMPIInt  nprocs;
    PetscErrorCode ierr = VecGetOwnershipRanges(A, & rangeA);
    ierr = VecGetOwnershipRanges(Result, & rangeResult);
    MPI_Comm_size(mpi_communicator, &nprocs);
    assert(nprocs == dof_handler.n_procs);
    for (int i=0; i<nprocs; ++i)
        Assert(rangeA[i] == rangeResult[i],
                    dealii::ExcDimensionMismatch (rangeA[i], rangeResult[i]));

    MatMult(adjoint_action, A, Result);

    PetscInt begin, end;
    ierr = VecGetOwnershipRange (Result, &begin, &end);
    AssertThrow (ierr == 0, dealii::ExcPETScError(ierr));

    PetscScalar * start_ptr;
    ierr = VecGetArray(Result, & start_ptr);
    AssertThrow (ierr == 0, dealii::ExcPETScError(ierr));
        

    /* We use the FFT now to translate in the unit cells! */
    for (unsigned int n=0; n < dof_handler.n_locally_owned_points(); ++n)
    {
        const PointData& this_point = dof_handler.locally_owned_point(n);

        auto dof_range = dof_handler.get_dof_range(this_point.block_id, this_point.index_in_block);
        assert(dof_range.first >= begin && dof_range.second <= end);

        switch (this_point.block_id) 
        {
            case 0:
            {
                dealii::Tensor<1,dim>
                this_point_position = unit_cell(1).inverse_basis * lattice(0).get_vertex_position(this_point.index_in_block);

                /* Copy into an array allocated with fftw alignment */
                std::copy(start_ptr + (dof_range.first - begin), start_ptr + (dof_range.second - begin), data_in_0);

                fftw_execute(fplan_0);

                /* Phase shift */
                dealii::Tensor<1,dim> indices;
                unsigned int stride = layer(0).n_orbitals * layer(0).n_orbitals;
                for (unsigned int unrolled_index = 0; unrolled_index < unit_cell(1).n_nodes; ++unrolled_index)
                {
                    indices[0] = unrolled_index % stride;
                    if (dim == 2)
                        indices[1] = unrolled_index / stride;

                    PetscScalar phase = std::polar(1./(double) point_size[0], 
                                                    -2 * numbers::PI * dealii::scalar_product(this_point_position,  indices));
                    for (unsigned int j=0; j<stride; ++j)
                        data_out_0[stride * unrolled_index + j] *= phase;
                }

                /* Backward FFT */
                fftw_execute(bplan_0);

                /* Copy back into MPI array */
                std::copy(data_in_0, data_in_0 + (dof_range.second - dof_range.first), start_ptr + (dof_range.first - begin));
                break;
            }
            case 3:
            {
                dealii::Tensor<1,dim>
                this_point_position = unit_cell(0).inverse_basis * lattice(1).get_vertex_position(this_point.index_in_block);

                /* Copy into an array allocated with fftw alignment */
                std::copy(start_ptr + (dof_range.first - begin), start_ptr + (dof_range.second - begin), data_in_1);

                /* Forward FFT */
                fftw_execute(fplan_1);

                /* Phase shift */
                dealii::Tensor<1,dim> indices;
                unsigned int stride = layer(1).n_orbitals * layer(1).n_orbitals;
                for (unsigned int unrolled_index = 0; unrolled_index < unit_cell(0).n_nodes; ++unrolled_index)
                {
                    indices[0] = unrolled_index % stride;
                    if (dim == 2)
                        indices[1] = unrolled_index / stride;

                    PetscScalar phase = std::polar(1./(double) point_size[1], 
                                                    -2 * numbers::PI * dealii::scalar_product( this_point_position, indices));
                    for (unsigned int j=0; j<stride; ++j)
                        data_out_1[stride * unrolled_index + j] *= phase;
                }

                /* Backward FFT */
                fftw_execute(bplan_1);

                /* Copy back into MPI array */
                std::copy(data_in_1, data_in_1 + (dof_range.second - dof_range.first), start_ptr + (dof_range.first - begin));
                break;
            }
        }   
    }
    /* We finally take the conjugate of everyone! */
    VecRestoreArray(Result, & start_ptr);
    ierr = VecConjugate(Result);
    AssertThrow (ierr == 0, dealii::ExcPETScError(ierr));
}


template<int dim, int degree>
void
BaseAlgebra<dim,degree>::create_identity(Vec& Result)
{
    VecSet(Result, 0.0);
    std::vector<int> indices;
    std::vector<PetscScalar> values;

    std::array<int, dim> lattice_indices_0;
    for (unsigned int i=0; i<dim; ++i)
        lattice_indices_0[i] = 0;
    /* Block 0 */
    unsigned int lattice_index_0 = lattice(0).get_vertex_global_index(lattice_indices_0);
    if (dof_handler.is_locally_owned_point(0,lattice_index_0))
        for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
            for (int orbital = 0; orbital < layer(0).n_orbitals; ++orbital)
            {
                indices.push_back(dof_handler.get_dof_index(0, lattice_index_0, cell_index, orbital, orbital));
                values.push_back(1.);
            }

    /* Block 3 */
    lattice_index_0 = lattice(1).get_vertex_global_index(lattice_indices_0);
    if (dof_handler.is_locally_owned_point(3,lattice_index_0))
        for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
            for (int orbital = 0; orbital < layer(1).n_orbitals; ++orbital)
            {
                indices.push_back(dof_handler.get_dof_index(3, lattice_index_0, cell_index, orbital, orbital));
                values.push_back(1.);
            }

    VecSetValues(Result, indices.size(), indices.data(), values.data(), INSERT_VALUES);
    VecAssemblyBegin(Result);
    VecAssemblyEnd(Result);
}


template<int dim, int degree>
PetscScalar
BaseAlgebra<dim,degree>::trace(const Vec& A)
{
    std::vector<int> indices;
    std::vector<PetscScalar> values;

    std::array<int, dim> lattice_indices_0;
    for (unsigned int i=0; i<dim; ++i)
        lattice_indices_0[i] = 0;
    /* Block 0 */
    unsigned int lattice_index_0 = lattice(0).get_vertex_global_index(lattice_indices_0);
    if (dof_handler.is_locally_owned_point(0,lattice_index_0))
        for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
            for (int orbital = 0; orbital < layer(0).n_orbitals; ++orbital)
                indices.push_back(dof_handler.get_dof_index(0, lattice_index_0, cell_index, orbital, orbital));

    values.resize(indices.size());
    VecGetValues(A, indices.size(), indices.data(), values.data());
    PetscScalar 
    loc_trace_0 = std::accumulate(values.begin(), values.end(), static_cast<PetscScalar>(0.0)) 
                        * unit_cell(1).area / (double) unit_cell(1).n_nodes;

    indices.clear();

    /* Block 3 */
    lattice_index_0 = lattice(1).get_vertex_global_index(lattice_indices_0);
    if (dof_handler.is_locally_owned_point(3,lattice_index_0))
        for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
            for (int orbital = 0; orbital < layer(1).n_orbitals; ++orbital)
                indices.push_back(dof_handler.get_dof_index(3, lattice_index_0, cell_index, orbital, orbital));

    values.resize(indices.size());
    VecGetValues(A, indices.size(), indices.data(), values.data());
    PetscScalar 
    loc_trace_1 = std::accumulate(values.begin(), values.end(), static_cast<PetscScalar>(0.0)) 
                        * unit_cell(0).area / static_cast<double>( unit_cell(0).n_nodes );

    PetscScalar 
    loc_trace = (loc_trace_0 + loc_trace_1) / (unit_cell(0).area + unit_cell(1).area),
    result = 0.0;
    // Weird bug does not allow for direct MPI_Reduce of PETSC_SCALAR / MPIU_SCALAR type */
    const double Loc_Trace [2] = {loc_trace.real(), loc_trace.imag()};
    double Result [2];
    int ierr = MPI_Reduce(&Loc_Trace, &Result, 2, MPI_DOUBLE, MPI_SUM, 0, mpi_communicator);
    CHKERRQ(ierr);
    result = Result[0] + Result[1] * PETSC_i;
    return result;
}   

}/* End namespace Bilayer */
#endif
