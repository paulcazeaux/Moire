/* 
* File:   dofhandler.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/



#ifndef BILAYER_DOFHANDLER_H
#define BILAYER_DOFHANDLER_H

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <utility>
#include <metis.h>

#include "deal.II/base/mpi.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/index_set.h"
#include "deal.II/base/utilities.h"

#include "deal.II/lac/dynamic_sparsity_pattern.h"
#include "deal.II/lac/sparsity_pattern.h"
#include "deal.II/lac/sparsity_tools.h"

#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unitcell.h"
#include "tools/transformation.h"
#include "bilayer/pointdata.h"


/**
* This class encapsulates the discretized underlying bilayer structure:
* assembly of distributed meshes and lattice points for a bilayer groupoid.
* It is in particular responsible for building a sparsity pattern 
* and handling partitioning of the degrees of freedom by metis.
*/

namespace Bilayer {

template <int dim, int degree>
class DoFHandler : public Multilayer<dim, 2>
{


static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!\n");

public:
	DoFHandler(const Multilayer<dim, 2>& bilayer);
	~DoFHandler() {};

	const Lattice<dim>&					lattice(const int & idx) 	const { return *lattices_[idx]; };
	const UnitCell<dim,degree>&			unit_cell(const int & idx) 	const { return *unit_cells_[idx]; };

	void 								initialize(MPI_Comm& mpi_communicator);
	unsigned int 						my_pid;

	/**
	 * Construction of the sparsity patterns for the two main operations: right-multiply by the Hamiltonian, and transpose.
	 * We assume that the dynamic pattern has been initialized to the right size and row index set, obtained from this class
	 * either as locally_owned_dofs() which is enough for the right multiply operator, or the larger locally_relevant_dofs()
	 * for the transpose operator, which needs a further communication step to make sure all processors have all the relevant 
	 * entries of the sparsity pattern.
	 */
	void 								make_sparsity_pattern_rmultiply(dealii::DynamicSparsityPattern& dynamic_pattern) const;
	void 								make_sparsity_pattern_transpose(dealii::DynamicSparsityPattern& dynamic_pattern, MPI_Comm& mpi_communicator) const;

	void 								coarse_setup(MPI_Comm& mpi_communicator);
	void 								distribute_dofs();

	/* Some basic information */
	types::global_index 				n_dofs() const;
	types::global_index 				n_locally_owned_dofs() 	const;
	const dealii::IndexSet & 			locally_owned_dofs() const;
	const dealii::IndexSet & 			locally_relevant_dofs() const;
	const std::vector<types::global_index> & n_locally_owned_dofs_per_processor() const;


	/* Accessors to go from geometric indices to global degrees of freedom */
	types::global_index 				get_dof_index(const unsigned char block_id, const unsigned int index_in_block,
														const unsigned int cell_index, 
														const unsigned int orbital_row, const unsigned int orbital_column) const;
	std::pair<types::global_index,types::global_index>
										get_dof_range(const unsigned char block_id, const unsigned int index_in_block,
														const unsigned int cell_index) const;
	std::pair<types::global_index,types::global_index>
										get_dof_range(const unsigned char block_id, const unsigned int index_in_block) const;

private:
	/* The base geometry, available on every processor */
	std::array<std::unique_ptr<Lattice<dim>>, 2> 			lattices_;
	std::array<std::unique_ptr<UnitCell<dim,degree>>, 2> 	unit_cells_;

	/** 
	 * First, we organize our coarse degrees of freedom, i.e. lattice points for all 4 blocks.
	 * The lattice points for each term are organized according to the following structure.
	 *
	 *	[	Lattice 1   (intralayer terms of layer 1)	][	Lattice 2   (interlayer terms 2->1)	] ...
	 *			...	[	Lattice 1   (interlayer terms 1->2)	][	Lattice 2   (intralayer terms of layer 2)	]
	 *
	 * The starting index for each block is stored in the following array:
	 */
	std::array<unsigned int, 4> 							start_block_indices_;

	/* Total number of lattice points: */
	unsigned int 											num_lattice_points_;

	/* The subdomain ID for each lattice point, determined by the setup() function and known to all processors, 
	 * in the original numbering */
	unsigned int 											num_partitions_;
	std::vector<types::subdomain_id> 						partition_indices_;

	/* Re-ordering of corresponding lattice point indices, known to all processors */
	std::vector<unsigned int> 								reordered_indices_;
	std::vector<unsigned int> 								original_indices_;


	/* Corresponding slice owned by each processor (contiguous in the reordered set of indices) */
	std::vector<unsigned int> 								locally_owned_points_partition_;
	unsigned int 											n_locally_owned_points_;

	/* Tool to construct the above global partition and reordering of lattice points */
	dealii::DynamicSparsityPattern		make_coarse_sparsity_pattern();
	void 								make_coarse_partition(std::vector<unsigned int>& partition_indices);

	/**
	 *	Now we turn to the matter of enumerating local, fine degrees of freedom, including grid cell and orbital indices.
	 *  Total number of degrees of freedom for each point is 
	 *  								num_orbitals_[0]^2 * num_cell_points_[0] 		[block 0],
	 *  				num_orbitals_[0] * num_orbitals_[1] * num_cell_points_[1] 		[block 1],
	 *  				num_orbitals_[1] * num_orbitals_[0] * num_cell_points_[0] 		[block 2],
	 *  								num_orbitals_[1]^2 * num_cell_points_[1] 		[block 3],
	 */

	std::array<unsigned int, 2>								num_cell_points_;
	std::array<unsigned int, 2> 							num_orbitals_;

	std::array<unsigned int, 4> 							point_size_per_block_;
	std::array<std::array<unsigned int,3>, 4> 				strides_per_block_;

	std::vector<types::global_index> 						lattice_point_dof_range_;
	std::vector<PointData> 									lattice_points_;

	dealii::IndexSet 										locally_owned_dofs_;
	dealii::IndexSet 										locally_relevant_dofs_;
	std::vector<types::global_index>						n_locally_owned_dofs_per_processor_;
};


template<int dim, int degree>
DoFHandler<dim,degree>::DoFHandler(const Multilayer<dim, 2>& bilayer)
	:
	Multilayer<dim, 2>(bilayer)
{
	auto layers = bilayer.layer_data;
	for (unsigned int i = 0; i<2; ++i)
	{
		dealii::Tensor<2,dim>
		rotated_basis = Transformation<dim>::matrix(layers[i].get_dilation(), layers[i].get_angle()) * layers[i].get_lattice();
			
		lattices_[i]   = std::make_unique<Lattice<dim>>(rotated_basis, bilayer.cutoff_radius);
		unit_cells_[i] = std::make_unique<UnitCell<dim,degree>>(rotated_basis, bilayer.refinement_level);
		num_orbitals_[i] = layers[i].get_num_orbitals();
	}

	num_lattice_points_ = 2*(lattices_[0]->num_vertices + lattices_[1]->num_vertices);

	start_block_indices_ = {{	0, 
						lattices_[0]->num_vertices, 
						lattices_[0]->num_vertices + lattices_[1]->num_vertices,
						2*lattices_[0]->num_vertices + lattices_[1]->num_vertices	}};
}


template<int dim, int degree>
void
DoFHandler<dim,degree>::initialize(MPI_Comm& mpi_communicator)
{
	this->coarse_setup(mpi_communicator);
	this->distribute_dofs();
};


template<int dim, int degree>
void
DoFHandler<dim,degree>::coarse_setup(MPI_Comm& mpi_communicator)
{
	reordered_indices_.resize(num_lattice_points_);
	original_indices_.resize(num_lattice_points_);


	my_pid = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
	num_partitions_ = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
	partition_indices_.resize(num_lattice_points_);

	if (!my_pid) // Use Parmetis in the future?
	{
		/* the reordering array needs to be initialized to form a sparsity pattern */
		for (unsigned int i=0; i<num_lattice_points_; ++i)
			reordered_indices_[i] = i; 
		this->make_coarse_partition(partition_indices_);
	}
	/* Broadcast the result of this operation */
	int ierr = MPI_Bcast(partition_indices_.data(), num_lattice_points_, MPI_UNSIGNED,
                           0, mpi_communicator);
	AssertThrowMPI(ierr);
	/* Compute the reordering of the indices into contiguous range for each processor */

	locally_owned_points_partition_.resize(num_partitions_+1);
	unsigned int next_free_index = 0;
	for (types::subdomain_id m = 0; m<num_partitions_; ++m)
	{
		locally_owned_points_partition_[m] = next_free_index;
		for (unsigned int i = 0; i<num_lattice_points_; ++i)
			if (partition_indices_[i] == m)
			{
				reordered_indices_[i] = next_free_index;
				original_indices_[next_free_index] = i;
				++next_free_index;
			}
	}

	locally_owned_points_partition_[num_partitions_] = next_free_index;
	n_locally_owned_points_ = locally_owned_points_partition_[my_pid+1]
								- locally_owned_points_partition_[my_pid];
};

template<int dim, int degree>
dealii::DynamicSparsityPattern
DoFHandler<dim,degree>::make_coarse_sparsity_pattern()
{
	/**	We create first a dynamic (i.e. non compressed) sparsity coarse pattern. Coarse degrees 
	 *  of freedom correspond to these lattice points (condensing each a whole unit cell's worth of dofs)
	 */
	dealii::DynamicSparsityPattern dynamic_pattern(num_lattice_points_);

	/**
	 * Start with entries corresponding to the right-product by the Hamiltonian.
	 * For this operation,  the first two blocks are actually independent of the last two.
	 */

	/*******************/
	/* Rows in block 0 */
	/*******************/
	for (unsigned int m = 0; m<lattices_[0]->num_vertices; ++m)
	{
			/* Block 0 <-> 0 */
		std::vector<unsigned int> 	
		neighbors =	lattices_[0]->list_neighborhood_indices(
									lattices_[0]->get_vertex_position(m), 
									this->intra_search_radius);
		for (auto & mm: neighbors)
			mm = reordered_indices_[mm+start_block_indices_[0]];
		dynamic_pattern.add_entries(reordered_indices_[m+start_block_indices_[0]], neighbors.begin(), neighbors.end());
			/* Block 1 -> 0 */
		neighbors = lattices_[1]->list_neighborhood_indices(
									lattices_[0]->get_vertex_position(m), 
									this->inter_search_radius + unit_cells_[1]->bounding_radius);
		for (auto & pp: neighbors)
			pp = reordered_indices_[pp+start_block_indices_[1]];
		dynamic_pattern.add_entries(reordered_indices_[m+start_block_indices_[0]], neighbors.begin(), neighbors.end());
	}

	/*******************/
	/* Rows in block 1 */
	/*******************/
	for (unsigned int p = 0; p<lattices_[1]->num_vertices; ++p)
	{
			/* Block 0 -> 1 */
		std::vector<unsigned int> 	
		neighbors =	lattices_[0]->list_neighborhood_indices(
									lattices_[1]->get_vertex_position(p), 
									this->inter_search_radius + unit_cells_[1]->bounding_radius);
		for (auto & mm: neighbors) 
			mm = reordered_indices_[mm+start_block_indices_[0]];
		dynamic_pattern.add_entries(reordered_indices_[p+start_block_indices_[1]], neighbors.begin(), neighbors.end());
			/* Block 1 <-> 1 */
		neighbors = lattices_[1]->list_neighborhood_indices(
									lattices_[1]->get_vertex_position(p), this->intra_search_radius );
		for (auto & pp: neighbors) 
			pp = reordered_indices_[pp+start_block_indices_[1]];
		dynamic_pattern.add_entries(reordered_indices_[p+start_block_indices_[1]], neighbors.begin(), neighbors.end());
	}

	/*******************/
	/* Rows in block 2 */
	/*******************/
	for (unsigned int q = 0; q<lattices_[0]->num_vertices; ++q)
	{
			/* Block 3 -> 2 */
		std::vector<unsigned int> 	
		neighbors =	lattices_[1]->list_neighborhood_indices(
									lattices_[0]->get_vertex_position(q), 
									this->inter_search_radius + unit_cells_[0]->bounding_radius);
		for (auto & nn: neighbors) 
			nn = reordered_indices_[nn+start_block_indices_[3]];
		dynamic_pattern.add_entries(reordered_indices_[q+start_block_indices_[2]], neighbors.begin(), neighbors.end());
			/* Block 2 <-> 2 */
		neighbors = lattices_[0]->list_neighborhood_indices(
									lattices_[0]->get_vertex_position(q), this->intra_search_radius);
		for (auto & qq: neighbors) 
			qq = reordered_indices_[qq+start_block_indices_[2]];
		dynamic_pattern.add_entries(reordered_indices_[q+start_block_indices_[2]], neighbors.begin(), neighbors.end());
	}

	/*******************/
	/* Rows in block 3 */
	/*******************/
	for (unsigned int n = 0; n<lattices_[1]->num_vertices; ++n)
	{
			/* Block 3 <-> 3 */
		std::vector<unsigned int> 	
		neighbors =	lattices_[1]->list_neighborhood_indices(
									lattices_[1]->get_vertex_position(n), 
									this->intra_search_radius);
		for (auto & nn: neighbors) 
			nn = reordered_indices_[nn+start_block_indices_[3]];
		dynamic_pattern.add_entries(reordered_indices_[n+start_block_indices_[3]], neighbors.begin(), neighbors.end());
			/* Block 2 -> 3 */
		neighbors = lattices_[0]->list_neighborhood_indices(
									lattices_[1]->get_vertex_position(n), 
									this->inter_search_radius + unit_cells_[0]->bounding_radius);
		for (auto & pp: neighbors) 
			pp = reordered_indices_[pp+start_block_indices_[2]];
		dynamic_pattern.add_entries(reordered_indices_[n+start_block_indices_[3]], neighbors.begin(), neighbors.end());
	}

	/**
	 * Next, we add the terms corresponding to the adjoint operation.
	 * For this operation, the first and last blocks are independent of the rest, 
	 * while the two central blocks exchange terms.
	 * Within this coarsened framework we neglect boundary terms linking two neighboring cells.
	 */

	/*******************/
	/* Rows in block 0 */
	/*******************/
	for (unsigned int m = 0; m<lattices_[0]->num_vertices; ++m)
	{
			/* Block 0 <-> 0 */
		std::array<int, dim> on_grid = lattices_[0]->get_vertex_grid_indices(m);
		for (auto & c : on_grid)
			c = -c;
		unsigned int mm = lattices_[0]->get_vertex_global_index(on_grid);
		dynamic_pattern.add(reordered_indices_[m+start_block_indices_[0]], reordered_indices_[mm+start_block_indices_[0]]);

	}

	/*******************/
	/* Rows in block 1 */
	/*******************/
	for (unsigned int p = 0; p<lattices_[1]->num_vertices; ++p)
	{
			/* Block 2 -> 1 */
		std::vector<unsigned int> 	
		neighbors =	lattices_[0]->list_neighborhood_indices(
									-lattices_[1]->get_vertex_position(p), 
									unit_cells_[0]->bounding_radius+unit_cells_[1]->bounding_radius);
		for (auto & pp: neighbors) 
			pp = reordered_indices_[pp+start_block_indices_[2]];
		dynamic_pattern.add_entries(reordered_indices_[p+start_block_indices_[1]], neighbors.begin(), neighbors.end());

	}

	/*******************/
	/* Rows in block 2 */
	/*******************/
	for (unsigned int q = 0; q<lattices_[0]->num_vertices; ++q)
	{
			/* Block 1 -> 2 */
		std::vector<unsigned int> 	
		neighbors =	lattices_[1]->list_neighborhood_indices(
									-lattices_[0]->get_vertex_position(q), 
									unit_cells_[0]->bounding_radius+unit_cells_[1]->bounding_radius);
		for (auto & qq: neighbors) 
			qq = reordered_indices_[qq+start_block_indices_[1]];
		dynamic_pattern.add_entries(reordered_indices_[q+start_block_indices_[2]], neighbors.begin(), neighbors.end());
	}

	/*******************/
	/* Rows in block 3 */
	/*******************/
	for (unsigned int n = 0; n<lattices_[1]->num_vertices; ++n)
	{
			/* Block 3 <-> 3 */
		std::array<int, dim> on_grid = lattices_[1]->get_vertex_grid_indices(n);
		for (auto & c : on_grid)
			c = -c;
		unsigned int nn = lattices_[1]->get_vertex_global_index(on_grid);
		dynamic_pattern.add(reordered_indices_[n+start_block_indices_[3]], reordered_indices_[nn+start_block_indices_[3]]);
	}

	return dynamic_pattern;
};


template<int dim, int degree>
void
DoFHandler<dim,degree>::make_coarse_partition(std::vector<unsigned int>& partition_indices)
{
	// /* First round of reordering to facilitate partitioning? */
	// dealii::DynamicSparsityPattern dynamic_pattern = this->make_coarse_sparsity_pattern();
	// dealii::SparsityTools::reorder_Cuthill_McKee(dynamic_pattern, reordered_indices_);
	// std::vector<unsigned int> partition_indices_reordered(num_lattice_points_);

	/* Produce a sparsity pattern used for the Metis partitioner */
	dealii::SparsityPattern sparsity_pattern;
	sparsity_pattern.copy_from(this->make_coarse_sparsity_pattern());


	idx_t
    n       = static_cast<signed int>(num_lattice_points_),
    ncon    = 1,                              
    nparts  = static_cast<int>(num_partitions_), 
    dummy;                                    

    /* Set Metis options */
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions (options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options[METIS_OPTION_MINCONN] = 1;

    /* Transfer arrays to signed integer type */
    std::vector<idx_t> int_rowstart(1);
    int_rowstart.reserve(num_lattice_points_+1);
    std::vector<idx_t> int_colnums;
    int_colnums.reserve(sparsity_pattern.n_nonzero_elements());
    for (dealii::SparsityPattern::size_type row=0; row<num_lattice_points_; ++row)
      {
        for (dealii::SparsityPattern::iterator col=sparsity_pattern.begin(row);
             col < sparsity_pattern.end(row); ++col)
          int_colnums.push_back(col->column());
        int_rowstart.push_back(int_colnums.size());
      }

      /* Call Metis */
    std::vector<idx_t> int_partition_indices (num_lattice_points_);
    int
	ierr = METIS_PartGraphKway(&n, &ncon, &int_rowstart[0], &int_colnums[0],
                                 nullptr, nullptr, nullptr,
                                 &nparts,nullptr,nullptr,&options[0],
                                 &dummy,&int_partition_indices[0]);

    std::copy (int_partition_indices.begin(),
               int_partition_indices.end(),
               partition_indices.begin());

	/* revert Cuthill-McKee ordering */
	// for (unsigned int i=0; i<num_lattice_points_; ++i)
	// 	partition_indices[i] = partition_indices_reordered[reordered_indices_[i]];
}


template<int dim, int degree>
void
DoFHandler<dim,degree>::distribute_dofs()
{
	/* Initialize basic #dofs information for each lattice point */

	num_cell_points_[0] = unit_cells_[0]->number_of_interior_grid_points;
	num_cell_points_[1] = unit_cells_[1]->number_of_interior_grid_points;

	point_size_per_block_ =
				{{ 	num_cell_points_[0] * num_orbitals_[0] * num_orbitals_[0],
					num_cell_points_[1] * num_orbitals_[0] * num_orbitals_[1],
					num_cell_points_[0] * num_orbitals_[1] * num_orbitals_[0],
					num_cell_points_[1] * num_orbitals_[1] * num_orbitals_[1]	}};
	strides_per_block_ =
				{{	{{1, num_orbitals_[0], num_orbitals_[0] * num_orbitals_[0]}},
				 	{{1, num_orbitals_[1], num_orbitals_[1] * num_orbitals_[0]}},
				 	{{1, num_orbitals_[0], num_orbitals_[0] * num_orbitals_[1]}},
				 	{{1, num_orbitals_[1], num_orbitals_[1] * num_orbitals_[1]}} 	}};

	types::global_index
	total_num_dofs = point_size_per_block_[0] * lattices_[0]->num_vertices
					 + point_size_per_block_[1] * lattices_[1]->num_vertices
					 + point_size_per_block_[2] * lattices_[0]->num_vertices
					 + point_size_per_block_[3] * lattices_[1]->num_vertices;

	/**
	 * First, each processor assigns degree of freedom ranges for all lattice points
	 * (even those it doesn't own)
	 */
	lattice_point_dof_range_.resize(num_lattice_points_+1);

	types::global_index next_free_dof = 0;
	for (unsigned int n = 0; n<num_lattice_points_; ++n)
	{
		lattice_point_dof_range_[n] = next_free_dof;
		/* Find the block id to determine the size */
		unsigned char block_id = 3;
		while (original_indices_[n] < start_block_indices_[block_id]) // note: start_indices[0] = 0, terminates
			block_id--;
		next_free_dof += point_size_per_block_[block_id];
	}
	lattice_point_dof_range_[num_lattice_points_] = next_free_dof;

	if (total_num_dofs != next_free_dof)
		throw std::runtime_error("Error in the distribution of degrees of freedom to the lattice points, which do not sum up to the right value: " 
			+ std::to_string(next_free_dof)+ " out of expected " + std::to_string(total_num_dofs) + ".\n");

	n_locally_owned_dofs_per_processor_.resize(num_partitions_);
	for (types::subdomain_id m = 0; m<num_partitions_; ++m)
		n_locally_owned_dofs_per_processor_[m] =  lattice_point_dof_range_[locally_owned_points_partition_[m+1]] 
								- lattice_point_dof_range_[locally_owned_points_partition_[m]];

	locally_owned_dofs_.set_size(total_num_dofs);
	locally_owned_dofs_.add_range(lattice_point_dof_range_[locally_owned_points_partition_[my_pid]], 
								lattice_point_dof_range_[locally_owned_points_partition_[my_pid+1]]);
	locally_relevant_dofs_.set_size(total_num_dofs);

	/**
	 * Now we know which global dof range is owned by each lattice point system-wide. 
	 * We explore our local lattice points and fill in the PointData structure
	 */
	for (unsigned int n = locally_owned_points_partition_[my_pid]; n < locally_owned_points_partition_[my_pid+1]; ++n)
	{
		/* Find the block id to determine the size */
		unsigned int 	this_point_index = original_indices_[n];
		unsigned char 	block_id = 3;
		while (this_point_index < start_block_indices_[block_id]) // note: start_indices[0] = 0, terminates
			block_id--;
		this_point_index -= start_block_indices_[block_id];

		/* Basic info goes into PointData structure */
		lattice_points_.emplace_back(block_id, this_point_index, total_num_dofs);
		PointData& this_point = lattice_points_.back();
		this_point.owned_dofs.add_range(lattice_point_dof_range_[n], 
														lattice_point_dof_range_[n+1]);
		this_point.relevant_dofs.add_range(lattice_point_dof_range_[n], 
														lattice_point_dof_range_[n+1]);

		/* Case of inter-layer blocks: boundary points and interpolation points require more data */
		if (block_id == 1 || block_id == 2)
		{
			const auto& cell 	= this->unit_cell(block_id == 1? 1: 0);
			const auto& lattice = this->lattice(block_id == 1? 1: 0);
			std::array<int,dim> 
			this_vertex_indices = lattice.get_vertex_grid_indices(this_point_index);

		/* We now deal with the technicalities of boundary points */
			this_point.boundary_lattice_points.reserve(cell.number_of_boundary_grid_points);

			for (unsigned int p = 0; p < cell.number_of_boundary_grid_points; ++p)
			{
				/* Which cell does this boundary point belong to? */
				auto [cell_index, lattice_indices] = cell.map_boundary_point_interior(p);
				for (unsigned int i=0; i<dim; ++i)
					lattice_indices[i] += this_vertex_indices[i];

				/* Find out what is the corresponding lattice point index */
				unsigned int index_in_block = lattice.get_vertex_global_index(lattice_indices);
				/* Check that this point exists in our cutout */
				if (index_in_block != types::invalid_lattice_index)
				{
					/* Identify block of global degrees of freedom */
					auto [dof_range_start, dof_range_end] = get_dof_range(block_id, index_in_block, cell_index);
					/* Add the point to the vector of identified boundary points */
					this_point.boundary_lattice_points.push_back(
						std::make_tuple(block_id, index_in_block, cell_index));
					/* Add its block of degrees of freedom to the relevant range */
					this_point.relevant_dofs.add_range(dof_range_start, dof_range_end);
				}
				else // The point is out of bounds
					this_point.boundary_lattice_points.push_back(std::make_tuple(
						block_id,
						types::invalid_lattice_index, types::invalid_lattice_index));
			}

		/* We now find and add the interpolation points from the other grid */

			const auto this_point_position 	= lattice.get_vertex_position(this_point_index);
			const auto& other_cell 			= this->unit_cell(block_id == 1? 0: 1);
			const auto& other_lattice 		= this->lattice(block_id == 1? 0: 1);

			unsigned int estimate_size = other_cell.number_of_interior_grid_points 
						* std::ceil(dealii::determinant(cell.basis)/dealii::determinant(other_cell.basis));
			this_point.interpolated_grid_points.reserve(estimate_size);

			/* Select and iterate through the relevant neighbors of our current point in the other lattice */
			std::vector<unsigned int> 	
			neighbors = other_lattice.list_neighborhood_indices( -this_point_position, 
								other_cell.bounding_radius+cell.bounding_radius);
			unsigned char other_block_id = (block_id == 1? 2 : 1);
			for (unsigned int index_in_block : neighbors)
			{
				const dealii::Point<dim> relative_point_position = - (other_lattice.get_vertex_position(index_in_block) + this_point_position);
				
				/* Iterate through the grid points in the corresponding unit cell and test invidually if they are relevant */
				for (unsigned int cell_index = 0; cell_index < other_cell.number_of_interior_grid_points; ++cell_index)
				{
					unsigned int element_index = cell.find_element( relative_point_position - other_cell.get_grid_point_position(cell_index));
					/* Test wheter the point is in the cell */
					if (element_index != types::invalid_lattice_index) 
					{
						/* Add the point to the vector of identified interpolated points */
						this_point.interpolated_grid_points.push_back(std::make_pair(element_index,
										std::make_tuple(other_block_id, index_in_block, cell_index)));
					}
				}
			}
		}
		this_point.owned_dofs.compress();
		this_point.relevant_dofs.compress();
		locally_relevant_dofs_.add_indices(this_point.relevant_dofs);
	}
	// locally_owned_dofs_.compress();
	locally_relevant_dofs_.compress();
};


template<int dim, int degree>
void
DoFHandler<dim,degree>::make_sparsity_pattern_rmultiply(dealii::DynamicSparsityPattern& dynamic_pattern) const
{
	
	std::vector<types::global_index> columns;

	for (unsigned int n=0; n<n_locally_owned_points_; ++n)
	{
		const PointData& this_point = lattice_points_[n];

		auto [row_start, row_stop] = get_dof_range(this_point.block_id, this_point.index_in_block);
		unsigned int row_stride = strides_per_block_[this_point.block_id][2];
		assert(locally_owned_dofs_.is_element(row_start));
		switch (this_point.block_id) 
		{					/******************/
			case 0:			/* Row in block 0 */
			{				/******************/
				dealii::Point<dim> this_point_position = lattices_[0]->get_vertex_position(this_point.index_in_block);


					/* Block 0 <-> 0 */
				unsigned int col_stride = strides_per_block_[0][2];
				std::vector<unsigned int> 	
				neighbors =	lattices_[0]->list_neighborhood_indices(this_point_position, 
											this->intra_search_radius);
				columns.clear();
				columns.reserve(neighbors.size() * col_stride);
				for (auto neighbor_index_in_block : neighbors)
				{
					auto [col_start, col_stop] = get_dof_range(0, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());

					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
				}
				for (unsigned int cell_index = 0; cell_index < num_cell_points_[1]; ++cell_index)
				{
					for (types::global_index row = row_start + cell_index * row_stride; 
											row < row_start + (cell_index+1) * row_stride; ++row)
						dynamic_pattern.add_entries(row, columns.begin(), columns.end());
					for (auto & col : columns)
						col += col_stride;
				}

					/* Block 1 -> 0 */
				col_stride = strides_per_block_[1][2];
				neighbors = lattices_[1]->list_neighborhood_indices( this_point_position, 
											this->inter_search_radius + unit_cells_[1]->bounding_radius);
				
				for (auto neighbor_index_in_block : neighbors)
				{
					columns.clear();
					columns.reserve(col_stride);
					auto [col_start, col_stop] = get_dof_range(1, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());

					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
					dealii::Point<dim> neighbor_position = lattices_[1]->get_vertex_position(neighbor_index_in_block);

					for (unsigned int cell_index = 0; cell_index < num_cell_points_[1]; ++cell_index)
					{
						if (this_point_position.distance(neighbor_position + unit_cells_[1]->get_grid_point_position(cell_index)) < this->inter_search_radius)
							for (types::global_index row = row_start + cell_index * row_stride; 
													row < row_start + (cell_index+1) * row_stride; ++row)
								dynamic_pattern.add_entries(row, columns.begin(), columns.end());
						for (auto & col : columns)
							col += col_stride;
					}
				}
				break;
			}				/******************/
			case 1:			/* Row in block 1 */
			{				/******************/
				dealii::Point<dim> this_point_position = lattices_[1]->get_vertex_position(this_point.index_in_block);

					/* Block 0 -> 1 */
				unsigned int col_stride = strides_per_block_[0][2];
				std::vector<unsigned int> 	
				neighbors =	lattices_[0]->list_neighborhood_indices( this_point_position, 
											this->inter_search_radius + unit_cells_[1]->bounding_radius);

				for (auto neighbor_index_in_block : neighbors)
				{
					columns.clear();
					columns.reserve(col_stride);
					auto [col_start, col_stop] = get_dof_range(0, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());

					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
					dealii::Point<dim> neighbor_position = lattices_[0]->get_vertex_position(neighbor_index_in_block);

					for (unsigned int cell_index = 0; cell_index < num_cell_points_[1]; ++cell_index)
					{
						if (neighbor_position.distance(this_point_position + unit_cells_[1]->get_grid_point_position(cell_index)) < this->inter_search_radius)
							for (types::global_index row = row_start + cell_index * row_stride; 
													row < row_start + (cell_index+1) * row_stride; ++row)
								dynamic_pattern.add_entries(row, columns.begin(), columns.end());
						for (auto & col : columns)
							col += col_stride;
					}
				}
					/* Block 1 <-> 1 */
				col_stride = strides_per_block_[1][2];
				neighbors = lattices_[1]->list_neighborhood_indices(this_point_position, this->intra_search_radius);
				columns.clear();
				columns.reserve(neighbors.size() * col_stride);
				for (auto neighbor_index_in_block : neighbors)
				{
					auto [col_start, col_stop] = get_dof_range(1, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());

					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
				}
				for (unsigned int cell_index = 0; cell_index < num_cell_points_[1]; ++cell_index)
				{
					for (types::global_index row = row_start + cell_index * row_stride; 
											row < row_start + (cell_index+1) * row_stride; ++row)
						dynamic_pattern.add_entries(row, columns.begin(), columns.end());
					for (auto & col : columns)
						col += col_stride;
				}

				break;
			}				/*******************/
			case 2:			/* Rows in block 2 */
			{				/*******************/
				dealii::Point<dim> this_point_position = lattices_[0]->get_vertex_position(this_point.index_in_block);

						/* Block 3 -> 2 */
				unsigned int col_stride = strides_per_block_[3][2];
				std::vector<unsigned int> 	
				neighbors =	lattices_[1]->list_neighborhood_indices( this_point_position, 
											this->inter_search_radius + unit_cells_[0]->bounding_radius);

				for (auto neighbor_index_in_block : neighbors)
				{
					columns.clear();
					columns.reserve(col_stride);
					auto [col_start, col_stop] = get_dof_range(3, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());

					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
					dealii::Point<dim> neighbor_position = lattices_[1]->get_vertex_position(neighbor_index_in_block);

					for (unsigned int cell_index = 0; cell_index < num_cell_points_[0]; ++cell_index)
					{
						if (neighbor_position.distance(this_point_position + unit_cells_[0]->get_grid_point_position(cell_index)) < this->inter_search_radius)
							for (types::global_index row = row_start + cell_index * row_stride; 
													row < row_start + (cell_index+1) * row_stride; ++row)
								dynamic_pattern.add_entries(row, columns.begin(), columns.end());
						for (auto & col : columns)
							col += col_stride;
					}
				}
					/* Block 2 <-> 2 */
				col_stride = strides_per_block_[2][2];
				neighbors = lattices_[0]->list_neighborhood_indices(this_point_position, this->intra_search_radius);
				columns.clear();
				columns.reserve(neighbors.size() * col_stride);
				for (auto neighbor_index_in_block : neighbors)
				{
					auto [col_start, col_stop] = get_dof_range(2, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());

					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
				}
				for (unsigned int cell_index = 0; cell_index < num_cell_points_[0]; ++cell_index)
				{
					for (types::global_index row = row_start + cell_index * row_stride; 
											row < row_start + (cell_index+1) * row_stride; ++row)
						dynamic_pattern.add_entries(row, columns.begin(), columns.end());
					for (auto & col : columns)
						col += col_stride;
				}

				break;
			}				/*******************/
			case 3:			/* Rows in block 3 */
			{				/*******************/
				dealii::Point<dim> this_point_position = lattices_[1]->get_vertex_position(this_point.index_in_block);


					/* Block 3 <-> 3 */
				unsigned int col_stride = strides_per_block_[3][2];
				std::vector<unsigned int> 	
				neighbors =	lattices_[1]->list_neighborhood_indices(this_point_position, 
											this->intra_search_radius);
				columns.clear();
				columns.reserve(neighbors.size() * col_stride);
				for (auto neighbor_index_in_block : neighbors)
				{
					auto [col_start, col_stop] = get_dof_range(3, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());
					
					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
				}
				for (unsigned int cell_index = 0; cell_index < num_cell_points_[0]; ++cell_index)
				{
					for (types::global_index row = row_start + cell_index * row_stride; 
											row < row_start + (cell_index+1) * row_stride; ++row)
						dynamic_pattern.add_entries(row, columns.begin(), columns.end());
					for (auto & col : columns)
						col += col_stride;
				}

					/* Block 2 -> 3 */
				col_stride = strides_per_block_[2][2];
				neighbors = lattices_[0]->list_neighborhood_indices( this_point_position, 
											this->inter_search_radius + unit_cells_[0]->bounding_radius);
				
				for (auto neighbor_index_in_block : neighbors)
				{
					columns.clear();
					columns.reserve(col_stride);
					auto [col_start, col_stop] = get_dof_range(2, neighbor_index_in_block);
					assert(col_stop <= locally_owned_dofs_.size());
					
					for (types::global_index col = col_start; col < col_start+col_stride; ++col)
						columns.push_back(col);
					dealii::Point<dim> neighbor_position = lattices_[1]->get_vertex_position(neighbor_index_in_block);

					for (unsigned int cell_index = 0; cell_index < num_cell_points_[0]; ++cell_index)
					{
						if (this_point_position.distance(neighbor_position + unit_cells_[0]->get_grid_point_position(cell_index)) < this->inter_search_radius)
							for (types::global_index row = row_start + cell_index * row_stride; 
													row < row_start + (cell_index+1) * row_stride; ++row)
								dynamic_pattern.add_entries(row, columns.begin(), columns.end());
						for (auto & col : columns)
							col += col_stride;
					}
				}
				break;
			}
		}
	}
};



template<int dim, int degree>
void
DoFHandler<dim,degree>::make_sparsity_pattern_transpose(dealii::DynamicSparsityPattern& dynamic_pattern, MPI_Comm& mpi_communicator) const
{
	/**
	 * Note : the second step (shift translation) is dealt with by FFT after the operation modeled here
	 */
	
	for (unsigned int n=0; n<n_locally_owned_points_; ++n)
	{
		const PointData& this_point = lattice_points_[n];
		
		switch (this_point.block_id) {
							/***********************/
			case 0:			/* Row in block 0 -> 0 */
			{				/***********************/
				std::array<int, dim> on_grid = lattices_[0]->get_vertex_grid_indices(this_point.index_in_block);
				for (auto & c : on_grid)
					c = -c;
				unsigned int neighbor_index_in_block = lattices_[0]->get_vertex_global_index(on_grid);

				for (unsigned int cell_index = 0; cell_index < num_cell_points_[1]; ++cell_index)
					for (unsigned int orbital_row = 0; orbital_row < num_orbitals_[0]; orbital_row++)
						for (unsigned int orbital_column = 0; orbital_column < num_orbitals_[0]; orbital_column++)
							dynamic_pattern.add(get_dof_index(0, this_point.index_in_block, cell_index, orbital_row, orbital_column),
												get_dof_index(0, neighbor_index_in_block, cell_index, orbital_column, orbital_row));
				break;
			}				/***************************/
			case 1:			/* COLUMNS in block 1 -> 2 */
			{				/***************************/

				for (auto & interp_point : this_point.interpolated_grid_points)
				{
					unsigned int element_index = interp_point.first;
					auto [ row_block_id, row_index_in_block, row_cell_index] = interp_point.second;
					for (unsigned int cell_index : unit_cells_[1]->subcell_list[element_index].unit_cell_dof_index_map)
					{
						if (unit_cells_[1]->is_grid_point_interior(cell_index))
						{
							for (unsigned int orbital_row = 0; orbital_row < num_orbitals_[0]; orbital_row++)
								for (unsigned int orbital_column = 0; orbital_column < num_orbitals_[1]; orbital_column++)
									dynamic_pattern.add(get_dof_index(2, row_index_in_block, row_cell_index, orbital_column, orbital_row),
														get_dof_index(1, this_point.index_in_block, cell_index, orbital_row, orbital_column));
						}
						else // Boundary point!
						{
							auto [column_block_id, column_index_in_block, column_cell_index] 
										= this_point.boundary_lattice_points[cell_index - num_cell_points_[1]];
							/* Last check to see if the boundary point actually exists on the grid. Maybe some extrapolation would be useful? */
							if (column_index_in_block != types::invalid_lattice_index) 
								for (unsigned int orbital_row = 0; orbital_row < num_orbitals_[0]; orbital_row++)
									for (unsigned int orbital_column = 0; orbital_column < num_orbitals_[1]; orbital_column++)
										dynamic_pattern.add(get_dof_index(2, row_index_in_block, row_cell_index, orbital_column, orbital_row),
															get_dof_index(1, column_index_in_block, column_cell_index, orbital_row, orbital_column));
						}
					}
				}

				break;
			}				/***************************/
			case 2:			/* COLUMNS in block 2 -> 1 */
			{				/***************************/
				for (auto & interp_point : this_point.interpolated_grid_points)
				{
					unsigned int element_index = interp_point.first;
					auto [ row_block_id, row_index_in_block, row_cell_index] = interp_point.second;

					for (unsigned int cell_index : unit_cells_[0]->subcell_list[element_index].unit_cell_dof_index_map)
					{
						if (unit_cells_[0]->is_grid_point_interior(cell_index))
							for (unsigned int orbital_row = 0; orbital_row < num_orbitals_[1]; orbital_row++)
								for (unsigned int orbital_column = 0; orbital_column < num_orbitals_[0]; orbital_column++)
									dynamic_pattern.add(get_dof_index(1, row_index_in_block, row_cell_index, orbital_column, orbital_row),
														get_dof_index(2, this_point.index_in_block, cell_index, orbital_row, orbital_column));
						else // Boundary point!
						{
							auto [column_block_id, column_index_in_block, column_cell_index] = this_point.boundary_lattice_points[cell_index - num_cell_points_[0]];
							/* Last check to see if the boundary point exists on the grid. Maybe some extrapolation would be useful? */
							if (column_index_in_block != types::invalid_lattice_index) 
								for (unsigned int orbital_row = 0; orbital_row < num_orbitals_[1]; orbital_row++)
									for (unsigned int orbital_column = 0; orbital_column < num_orbitals_[0]; orbital_column++)
										dynamic_pattern.add(get_dof_index(1, row_index_in_block, row_cell_index, orbital_column, orbital_row),
															get_dof_index(2, column_index_in_block, column_cell_index, orbital_row, orbital_column));
						}
					}
				}
				break;
			}				/************************/
			case 3:			/* Rows in block 3 -> 3 */
			{				/************************/
				std::array<int, dim> on_grid = lattices_[1]->get_vertex_grid_indices(this_point.index_in_block);
				for (auto & c : on_grid)
					c = -c;
				unsigned int opposite_index_in_block = lattices_[1]->get_vertex_global_index(on_grid);

				for (unsigned int cell_index = 0; cell_index < num_cell_points_[0]; ++cell_index)
					for (unsigned int orbital_row = 0; orbital_row < num_orbitals_[1]; orbital_row++)
						for (unsigned int orbital_column = 0; orbital_column < num_orbitals_[1]; orbital_column++)
							dynamic_pattern.add(get_dof_index(3, this_point.index_in_block, cell_index, orbital_row, orbital_column),
												get_dof_index(3, opposite_index_in_block, cell_index, orbital_column, orbital_row));
				break;
			}
		}
	}
	std::cout << std::endl;
	dealii::SparsityTools::distribute_sparsity_pattern(dynamic_pattern, n_locally_owned_dofs_per_processor_, mpi_communicator, locally_relevant_dofs_);
};



template<int dim, int degree>
types::global_index
DoFHandler<dim,degree>::n_dofs() const 
	{ return locally_owned_dofs_.size(); };


template<int dim, int degree>
types::global_index
DoFHandler<dim,degree>::n_locally_owned_dofs() const 
	{ return locally_owned_dofs_.n_elements(); };


template<int dim, int degree>
const dealii::IndexSet &
DoFHandler<dim,degree>::locally_owned_dofs() const 
	{ return locally_owned_dofs_; };


template<int dim, int degree>
const dealii::IndexSet &
DoFHandler<dim,degree>::locally_relevant_dofs() const 
	{ return locally_relevant_dofs_; };

template<int dim, int degree>
const std::vector<types::global_index> &
DoFHandler<dim,degree>::n_locally_owned_dofs_per_processor() const
	{ return n_locally_owned_dofs_per_processor_; }



template<int dim, int degree>
types::global_index
DoFHandler<dim,degree>::get_dof_index(const unsigned char block_id, const unsigned int index_in_block,
									const unsigned int cell_index, 
									const unsigned int orbital_row, const unsigned int orbital_column) const
{
	unsigned int index = reordered_indices_[start_block_indices_[block_id] + index_in_block];
	return lattice_point_dof_range_[index] + cell_index * strides_per_block_[block_id][2]
											+ orbital_row * strides_per_block_[block_id][1]
											+ orbital_column;
};

template<int dim, int degree>
std::pair<types::global_index,types::global_index>
DoFHandler<dim,degree>::get_dof_range(const unsigned char block_id, const unsigned int index_in_block,
									const unsigned int cell_index) const
{
	unsigned int index = reordered_indices_[start_block_indices_[block_id] + index_in_block];
	types::global_index
	dof_range_start = lattice_point_dof_range_[index] + cell_index * strides_per_block_[block_id][2];
	return std::make_pair(dof_range_start, dof_range_start + strides_per_block_[block_id][2]);
};

template<int dim, int degree>
std::pair<types::global_index,types::global_index>
DoFHandler<dim,degree>::get_dof_range(const unsigned char block_id, const unsigned int index_in_block) const
{
	unsigned int index = reordered_indices_[start_block_indices_[block_id] + index_in_block];
	return std::make_pair(lattice_point_dof_range_[index], lattice_point_dof_range_[index+1]);
};

}

#endif /* BILAYER_DOFHANDLER_H */
