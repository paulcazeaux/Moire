/* 
* File:   bilayer.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/



#ifndef BILAYER_H
#define BILAYER_H

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
class Groupoid : public Multilayer<dim, 2>
{


static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!\n");

public:
	Groupoid(const Multilayer<dim, 2>& parameters);
	~Groupoid() {};

	const Lattice<dim>&					lattice(const int & idx) 	const { return *lattices_[idx]; };
	const UnitCell<dim,degree>&			unit_cell(const int & idx) 	const { return *unit_cells_[idx]; };

	void 								coarse_setup(MPI_Comm mpi_communicator);
	void 								local_setup(MPI_Comm mpi_communicator);

private:
	/* The base geometry, available on every process */
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
	std::array<unsigned int, 4> 							start_indices_;

	/* Total number of lattice points: */
	unsigned int 											num_lattice_points_;

	/* The subdomain ID for each lattice point, determined by the setup() function and known to all processes, 
	 * in the original numbering */
	unsigned int 											num_partitions_;
	std::vector<types::subdomain_id> 						partition_indices_;

	/* Re-ordering of corresponding lattice point indices, known to all processes */
	std::vector<unsigned int> 								reordered_indices_;
	std::vector<unsigned int> 								original_indices_;

	/* Corresponding slice owned by / relevant to the current process (contiguous in the reordered set of indices) */
	std::array<types::global_index,2>						locally_owned_points_range_;

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
	types::global_index 									total_num_dofs_;

	unsigned int 											num_local_lattice_points_;
	std::vector<types::global_index> 						lattice_point_dof_range_;
	std::vector<PointData> 									lattice_points_;

	dealii::IndexSet 										owned_global_dofs;
	dealii::IndexSet 										relevant_global_dofs;

	void 								distribute_dofs();
	void								make_local_sparsity_pattern();
};


template<int dim, int degree>
Groupoid<dim,degree>::Groupoid(const Multilayer<dim, 2>& parameters)
	:
	Multilayer<dim, 2>(parameters)
{
	auto layers = parameters.layer_data;
	for (unsigned int i = 0; i<2; ++i)
	{
		dealii::Tensor<2,dim>
		rotated_basis = Transformation<dim>::matrix(layers[i].get_dilation(), layers[i].get_angle()) * layers[i].get_lattice();
			
		lattices_[i]   = std::make_unique<Lattice<dim>>(rotated_basis, parameters.cutoff_radius);
		unit_cells_[i] = std::make_unique<UnitCell<dim,degree>>(rotated_basis, parameters.refinement_level);
		num_orbitals_[i] = layers[i].get_num_orbitals();
	}

	num_lattice_points_ = 2*(lattices_[0]->num_vertices + lattices_[1]->num_vertices);

	start_indices_ = {{	0, 
						lattices_[0]->num_vertices, 
						lattices_[0]->num_vertices + lattices_[1]->num_vertices,
						2*lattices_[0]->num_vertices + lattices_[1]->num_vertices	}};
}


template<int dim, int degree>
void
Groupoid<dim,degree>::coarse_setup(MPI_Comm mpi_communicator)
{
	reordered_indices_.resize(num_lattice_points_);
	original_indices_.resize(num_lattice_points_);


	unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
	num_partitions_ = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
	partition_indices_.resize(num_lattice_points_);

	if (my_rank == 0) // Use Parmetis in the future?
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
	unsigned int next_free_index = 0, start, end;
	for (types::subdomain_id m = 0; m<num_partitions_; ++m)
	{
		if (m == my_rank)
			locally_owned_points_range_[0] = next_free_index;
		for (unsigned int i = 0; i<num_lattice_points_; ++i)
			if (partition_indices_[i] == m)
			{
				reordered_indices_[i] = next_free_index;
				original_indices_[next_free_index] = i;
				++next_free_index;
			}
		if (m == my_rank)
			locally_owned_points_range_[1] = next_free_index;
	}

	num_local_lattice_points_ = locally_owned_points_range_[1] - locally_owned_points_range_[0];
};

template<int dim, int degree>
void
Groupoid<dim,degree>::local_setup(MPI_Comm mpi_communicator)
{
	this->distribute_dofs();
	unsigned int my_rank = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

	if (my_rank == 0)
	{
		for (unsigned int n=0; n<num_local_lattice_points_; n++)
		{
			unsigned char block_id = lattice_points_[n].block_id;
			unsigned int idx = original_indices_[n + locally_owned_points_range_[0]] 
						- start_indices_[block_id];
			if (block_id == 0)
				std::cout << lattices_[0]->get_vertex_position(idx) << ";";
		}
		std::cout << std::endl<< std::endl<< std::endl;
		for (unsigned int n=0; n<num_local_lattice_points_; n++)
		{
			unsigned char block_id = lattice_points_[n].block_id;
			unsigned int idx = original_indices_[n + locally_owned_points_range_[0]] 
						- start_indices_[block_id];
			if (block_id == 1)
				std::cout << lattices_[1]->get_vertex_position(idx) << ";";
		}
		std::cout << std::endl<< std::endl<< std::endl;
		for (unsigned int n=0; n<num_local_lattice_points_; n++)
		{
			unsigned char block_id = lattice_points_[n].block_id;
			unsigned int idx = original_indices_[n + locally_owned_points_range_[0]] 
						- start_indices_[block_id];
			if (block_id == 2)
				std::cout << lattices_[0]->get_vertex_position(idx) << ";";
		}
		std::cout << std::endl<< std::endl<< std::endl;
		for (unsigned int n=0; n<num_local_lattice_points_; n++)
		{
			unsigned char block_id = lattice_points_[n].block_id;
			unsigned int idx = original_indices_[n + locally_owned_points_range_[0]] 
						- start_indices_[block_id];
			if (block_id == 3)
				std::cout << lattices_[1]->get_vertex_position(idx) << ";";
		}
		std::cout << std::endl<< std::endl<< std::endl;
		for (unsigned int n=0; n<num_local_lattice_points_; n++)
		{
			for (auto bp : lattice_points_[n].boundary_lattice_points)
			{
				auto [index,grid_index,dof_start,dof_end]  = bp;
				if (index != types::invalid_lattice_index && (index < locally_owned_points_range_[0] || index >= locally_owned_points_range_[1]))
				{
					unsigned char block_id = lattice_points_[n].block_id;
					unsigned int idx = original_indices_[index] 
								- start_indices_[block_id];
					if (block_id == 2)
						std::cout << lattices_[0]->get_vertex_position(idx) + unit_cells_[0]->get_grid_point_position(grid_index) << ";";
					else
						std::cout << lattices_[1]->get_vertex_position(idx) + unit_cells_[1]->get_grid_point_position(grid_index) << ";";
				}
			}
		}
	}
}

template<int dim, int degree>
dealii::DynamicSparsityPattern
Groupoid<dim,degree>::make_coarse_sparsity_pattern()
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
			mm = reordered_indices_[mm+start_indices_[0]];
		dynamic_pattern.add_entries(reordered_indices_[m+start_indices_[0]], neighbors.begin(), neighbors.end());
			/* Block 1 -> 0 */
		neighbors = lattices_[1]->list_neighborhood_indices(
									lattices_[0]->get_vertex_position(m), 
									this->inter_search_radius + unit_cells_[1]->bounding_radius);
		for (auto & pp: neighbors)
			pp = reordered_indices_[pp+start_indices_[1]];
		dynamic_pattern.add_entries(reordered_indices_[m+start_indices_[0]], neighbors.begin(), neighbors.end());
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
			mm = reordered_indices_[mm+start_indices_[0]];
		dynamic_pattern.add_entries(reordered_indices_[p+start_indices_[1]], neighbors.begin(), neighbors.end());
			/* Block 1 <-> 1 */
		neighbors = lattices_[1]->list_neighborhood_indices(
									lattices_[1]->get_vertex_position(p), 
									this->intra_search_radius + 2*unit_cells_[1]->bounding_radius);
		for (auto & pp: neighbors) 
			pp = reordered_indices_[pp+start_indices_[1]];
		dynamic_pattern.add_entries(reordered_indices_[p+start_indices_[1]], neighbors.begin(), neighbors.end());
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
			nn = reordered_indices_[nn+start_indices_[3]];
		dynamic_pattern.add_entries(reordered_indices_[q+start_indices_[2]], neighbors.begin(), neighbors.end());
			/* Block 2 <-> 2 */
		neighbors = lattices_[0]->list_neighborhood_indices(
									lattices_[0]->get_vertex_position(q), 
									this->intra_search_radius + 2*unit_cells_[0]->bounding_radius);
		for (auto & qq: neighbors) 
			qq = reordered_indices_[qq+start_indices_[2]];
		dynamic_pattern.add_entries(reordered_indices_[q+start_indices_[2]], neighbors.begin(), neighbors.end());
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
			nn = reordered_indices_[nn+start_indices_[3]];
		dynamic_pattern.add_entries(reordered_indices_[n+start_indices_[3]], neighbors.begin(), neighbors.end());
			/* Block 2 -> 3 */
		neighbors = lattices_[0]->list_neighborhood_indices(
									lattices_[1]->get_vertex_position(n), 
									this->inter_search_radius + unit_cells_[0]->bounding_radius);
		for (auto & pp: neighbors) 
			pp = reordered_indices_[pp+start_indices_[2]];
		dynamic_pattern.add_entries(reordered_indices_[n+start_indices_[3]], neighbors.begin(), neighbors.end());
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
		dynamic_pattern.add(reordered_indices_[m+start_indices_[0]], reordered_indices_[mm+start_indices_[0]]);

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
			pp = reordered_indices_[pp+start_indices_[2]];
		dynamic_pattern.add_entries(reordered_indices_[p+start_indices_[1]], neighbors.begin(), neighbors.end());

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
			qq = reordered_indices_[qq+start_indices_[1]];
		dynamic_pattern.add_entries(reordered_indices_[q+start_indices_[2]], neighbors.begin(), neighbors.end());
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
		dynamic_pattern.add(reordered_indices_[n+start_indices_[3]], reordered_indices_[nn+start_indices_[3]]);
	}

	return dynamic_pattern;
};


template<int dim, int degree>
void
Groupoid<dim,degree>::make_coarse_partition(std::vector<unsigned int>& partition_indices)
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
Groupoid<dim,degree>::distribute_dofs()
{
	/* Basic #dofs information for each lattice point */

	num_cell_points_[0] = unit_cells_[0]->number_of_interior_grid_points;
	num_cell_points_[1] = unit_cells_[1]->number_of_interior_grid_points;

	std::array<unsigned int, 4> 
	point_size_per_block 
				{{ 	num_cell_points_[0] * num_orbitals_[0] * num_orbitals_[0],
					num_cell_points_[1] * num_orbitals_[0] * num_orbitals_[1],
					num_cell_points_[0] * num_orbitals_[1] * num_orbitals_[0],
					num_cell_points_[1] * num_orbitals_[1] * num_orbitals_[1]	}};
	std::array<std::array<unsigned int,3>, 4> 
	strides_per_block 
				{{	{{1, num_orbitals_[0], num_orbitals_[0] * num_orbitals_[0]}},
				 	{{1, num_orbitals_[1], num_orbitals_[1] * num_orbitals_[0]}},
				 	{{1, num_orbitals_[0], num_orbitals_[0] * num_orbitals_[1]}},
				 	{{1, num_orbitals_[1], num_orbitals_[1] * num_orbitals_[1]}} 	}};

	total_num_dofs_ = point_size_per_block[0] * lattices_[0]->num_vertices
					 + point_size_per_block[1] * lattices_[1]->num_vertices
					 + point_size_per_block[2] * lattices_[0]->num_vertices
					 + point_size_per_block[3] * lattices_[1]->num_vertices;
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
		while (original_indices_[n] < start_indices_[block_id]) // note: start_indices[0] = 0, terminates
			block_id--;
		next_free_dof += point_size_per_block[block_id];
	}
	lattice_point_dof_range_[num_lattice_points_] = next_free_dof;

	if (total_num_dofs_ != next_free_dof)
		throw std::runtime_error("Error in the distribution of degrees of freedom to the lattice points, which do not sum up to the right value: " 
			+ std::to_string(next_free_dof)+ " out of expected " + std::to_string(total_num_dofs_) + ".\n");

	owned_global_dofs.set_size(total_num_dofs_);
	owned_global_dofs.add_range(lattice_point_dof_range_[locally_owned_points_range_[0]], 
								lattice_point_dof_range_[locally_owned_points_range_[1]]);
	relevant_global_dofs.set_size(total_num_dofs_);

	/**
	 * Now we know which global dof range is owned by each lattice point system-wide. 
	 * We explore our local lattice points and fill in the PointData structure
	 */
	for (unsigned int n = locally_owned_points_range_[0]; n < locally_owned_points_range_[1]; ++n)
	{
		/* Find the block id to determine the size */
		unsigned int 	this_point_index = original_indices_[n];
		unsigned char 	block_id = 3;
		while (this_point_index < start_indices_[block_id]) // note: start_indices[0] = 0, terminates
			block_id--;
		this_point_index -= start_indices_[block_id];

		/* Basic info goes into PointData structure */
		lattice_points_.emplace_back(block_id, point_size_per_block[block_id],
							total_num_dofs_, strides_per_block[block_id]);
		PointData& this_point = lattice_points_.back();
		this_point.owned_global_dofs.add_range(lattice_point_dof_range_[n], 
														lattice_point_dof_range_[n+1]);
		this_point.relevant_global_dofs.add_range(lattice_point_dof_range_[n], 
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
				unsigned int lattice_index = lattice.get_vertex_global_index(lattice_indices);
				/* Check that this point exists in our cutout */
				if (lattice_index != types::invalid_lattice_index)
				{
					lattice_index = reordered_indices_[	lattice_index + start_indices_[block_id]	];
					/* Identify block of global degrees of freedom */
					types::global_index 
					dof_block_start = lattice_point_dof_range_[lattice_index] 
										+ cell_index * strides_per_block[block_id][2],
					dof_block_end = dof_block_start + strides_per_block[block_id][2];
					/* Add the point to the vector of identified boundary points */
					this_point.boundary_lattice_points.push_back(
						std::make_tuple(lattice_index, cell_index, dof_block_start, dof_block_end ));
					/* Add its block of degrees of freedom to the relevant range */
					this_point.relevant_global_dofs.add_range(dof_block_start, dof_block_end);
				}
				else // The point is out of bounds
					this_point.boundary_lattice_points.push_back(std::make_tuple(
						types::invalid_lattice_index, types::invalid_lattice_index, 
						types::invalid_global_index, types::invalid_global_index	));
			}

		/* We now find and add the interpolation points from the other grid */

			const auto this_point_position 	= lattice.get_vertex_position(this_point_index);
			const auto& other_cell 			= this->unit_cell(block_id == 1? 0: 1);
			const auto& other_lattice 		= this->lattice(block_id == 1? 0: 1);

			unsigned int estimate_size = other_cell.number_of_interior_grid_points 
						* std::ceil(dealii::determinant(cell.basis)/dealii::determinant(other_cell.basis));
			this_point.interpolated_grid_points.reserve(estimate_size);
			this_point.interpolating_elements.reserve(estimate_size);

			/* Select and iterate through the relevant neighbors of our current point in the other lattice */
			std::vector<unsigned int> 	
			neighbors = other_lattice.list_neighborhood_indices( -this_point_position, 
								other_cell.bounding_radius+cell.bounding_radius);
			for (unsigned int lattice_index : neighbors)
			{
				const dealii::Point<dim> relative_point_position = - (other_lattice.get_vertex_position(lattice_index) + this_point_position);
				const unsigned int reordered_lattice_index = reordered_indices_[lattice_index + start_indices_[block_id == 1? 2 : 1]];
				
				/* Iterate through the grid points in the corresponding unit cell and test invidually if they are relevant */
				for (unsigned int cell_index = 0; cell_index < other_cell.number_of_interior_grid_points; ++cell_index)
				{
					unsigned int element_index = cell.find_element( relative_point_position - other_cell.get_grid_point_position(cell_index));
					/* Test wheter the point is in the cell */
					if (element_index != types::invalid_lattice_index) 
					{
						/* Identify block of global degrees of freedom */
						types::global_index 
						dof_block_start = lattice_point_dof_range_[reordered_lattice_index] 
											+ cell_index * strides_per_block[block_id == 1? 2 : 1][2],
						dof_block_end = dof_block_start + strides_per_block[block_id == 1? 2 : 1][2];
						this_point.interpolated_grid_points.push_back(
							std::make_tuple( reordered_lattice_index, cell_index, dof_block_start, dof_block_end ));
						this_point.interpolating_elements.push_back(element_index);
					}
				}
			}
		}
		this_point.owned_global_dofs.compress();
		this_point.relevant_global_dofs.compress();
		relevant_global_dofs.add_indices(this_point.relevant_global_dofs);
	}
	owned_global_dofs.compress();
	relevant_global_dofs.compress();
};


template<int dim, int degree>
void
Groupoid<dim,degree>::make_local_sparsity_pattern()
{

};


}

#endif /* BILAYER_H */
