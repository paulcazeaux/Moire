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

#include <Teuchos_GlobalMPISession.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_CrsGraph_decl.hpp>

#include "deal.II/base/exceptions.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/index_set.h"

#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unitcell.h"
#include "bilayer/pointdata.h"
#include "bilayer/blockdata.h"
#include "tools/transformation.h"


/**
 This class encapsulates the discretized underlying bilayer structure:
* assembly of distributed meshes and lattice points for a bilayer groupoid.
* It is in particular responsible for building a sparsity pattern 
* and handling partitioning of the degrees of freedom by metis.
*/

namespace Bilayer {

template <int dim, int degree>
class DoFHandler : public Multilayer<dim, 2>
{
	typedef Tpetra::Map<int, types::global_index> 		Map;
	typedef Tpetra::CrsGraph<int, types::global_index> 	SparsityPattern;

static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!\n");

public:
	DoFHandler(const Multilayer<dim, 2>& bilayer);
	~DoFHandler() {};

	const LayerData<dim>& 				layer(const int & idx) 		const { return this->layer_data.at(idx); }
	const Lattice<dim>&					lattice(const int & idx) 	const { return *lattices_.at(idx); };
	const UnitCell<dim,degree>&			unit_cell(const int & idx) 	const { return *unit_cells_.at(idx); };
	const BlockData& 					block(const int& idx) 		const { return this->block_.at(idx); }

	void 								initialize(Teuchos::RCP<const Teuchos::Comm<int> > comm);
	int 								my_pid;
	int 								n_procs;

	/**
	 * Construction of the sparsity patterns for the two main operations: right-multiply by the Hamiltonian, and adjoint.
	 */
	Teuchos::RCP<const SparsityPattern> make_sparsity_pattern_rmultiply(const unsigned char block_col) const;
	Teuchos::RCP<const SparsityPattern> make_sparsity_pattern_adjoint(const unsigned char block_col, const unsigned char block_row) const;

	/* Coarse level of discretization: basic information about lattice points */
	unsigned int 						n_points() const;
	unsigned int 						n_locally_owned_points() const;

	/* Accessors to traverse the lattice points level of discretization */
	const PointData & 					locally_owned_point(	const unsigned char block_col, 	const unsigned int local_index) const;
	const PointData & 					locally_owned_point(	const unsigned char block_col, 	const unsigned char block_row, 
																			const unsigned int index_in_block) const;
	bool 								is_locally_owned_point(const unsigned char block_row, const unsigned int index_in_block) const;

	/* Accessors to go from geometric indices to global degrees of freedom */
	types::global_index 				get_dof_index(	const unsigned char block_col, 		const unsigned char block_row,
														const unsigned int index_in_block,	const unsigned int cell_index, 
														const unsigned int orbital 	) const;
	std::pair<types::global_index,types::global_index>
										get_dof_range(	const unsigned char block_col, 		const unsigned char block_row,
														const unsigned int index_in_block,	const unsigned int cell_index) const;
	std::pair<types::global_index,types::global_index>
										get_dof_range(	const unsigned char block_col, 		const unsigned char block_row,
														const unsigned int index_in_block) const;

private:
	/* The base geometry, available on every processor */
	std::array<std::unique_ptr<Lattice<dim>>, 2> 			lattices_;
	std::array<std::unique_ptr<UnitCell<dim,degree>>, 2> 	unit_cells_;

	/** 
	 * First, we organize our coarse degrees of freedom, i.e. lattice points for both lattices.
	 * The dofs corresponding to each lattice point for each term are then
	 * organized as two MultiVectors according to the following structure.
	 *
	 * 			< row orbitals of layer 1 > 		
	 *	[	Lattice 1   (intralayer terms of layer 1)	]   -> block (0,0)
	 *	[	Lattice 2   (interlayer terms 2->1)			]	-> block (1,0)
	 *
	 *			< row orbitals of layer 2 >
	 *	[	Lattice 1   (interlayer terms 1->2)			]	-> block (0,1)
	 *	[	Lattice 2   (intralayer terms of layer 2)	]	-> block (1,1)
	 *
	 */

	/* Total number of lattice points: */
	unsigned int 											n_lattice_points_;

	/* The subdomain ID for each lattice point, determined by the setup() function and known to all processors, 
	 * in the original numbering */
	std::vector<types::subdomain_id> 						partition_indices_;

	/* Re-ordering of corresponding lattice point indices, known to all processors */
	std::vector<unsigned int> 								reordered_indices_;
	std::vector<unsigned int> 								original_indices_;


	/* Corresponding slice owned by each processor (contiguous in the reordered set of indices) */
	std::vector<unsigned int> 								locally_owned_points_partition_;
	unsigned int 											n_locally_owned_points_;

	/* Tool to construct the above global partition and reordering of lattice points */
	void 								make_coarse_partition(std::vector<unsigned int>& partition_indices);
	void 								coarse_setup(Teuchos::RCP<const Teuchos::Comm<int> > comm);
	void 								distribute_dofs(Teuchos::RCP<const Teuchos::Comm<int> > comm);

	/**
	 *	Now we turn to the matter of enumerating local, fine degrees of freedom, including cell node and orbital indices,
	 *  for our two MultiVectors.
	 *  Total number of degrees of freedom for each point is 
	 *  							layer(0).n_orbitals^2 				* unit_cell(0).n_nodes 		[block 0,0],
	 *  				layer(0).n_orbitals 	* layer(1).n_orbitals  	* unit_cell(1).n_nodes 		[block 1,0];
	 *
	 *  				layer(1).n_orbitals 	* layer(0).n_orbitals  	* unit_cell(0).n_nodes 		[block 0,1],
	 *  							layer(1).n_orbitals^2 				* unit_cell(1).n_nodes 		[block 1,1].
	 */

	std::array<BlockData, 2> 									block_;

};


template<int dim, int degree>
DoFHandler<dim,degree>::DoFHandler(const Multilayer<dim, 2>& bilayer)
	:
	Multilayer<dim, 2>(bilayer)
{
	for (unsigned int i = 0; i<2; ++i)
	{
		dealii::Tensor<2,dim>
		rotated_basis = Transformation<dim>::matrix(layer(i).dilation, layer(i).angle) * layer(i).lattice_basis;
			
		lattices_[i]   = std::make_unique<Lattice<dim>>(rotated_basis, bilayer.cutoff_radius);
		unit_cells_[i] = std::make_unique<UnitCell<dim,degree>>(rotated_basis, bilayer.refinement_level);
	}

	n_lattice_points_ = lattice(0).n_vertices + lattice(1).n_vertices;
}


template<int dim, int degree>
void
DoFHandler<dim,degree>::initialize(Teuchos::RCP<const Teuchos::Comm<int> > comm)
{
	this->coarse_setup(comm);
	this->distribute_dofs(comm);
};


template<int dim, int degree>
void
DoFHandler<dim,degree>::coarse_setup(Teuchos::RCP<const Teuchos::Comm<int> > comm)
{
	reordered_indices_.resize(n_points());
	original_indices_.resize(n_points());

	my_pid = comm->getRank ();
	n_procs = comm->getSize ();
	partition_indices_.resize(n_points());
	if (n_procs > 1)
	{
		if (!my_pid) // Use Parmetis in the future?
			this->make_coarse_partition(partition_indices_);
		/* Broadcast the result of this operation */
		Teuchos::broadcast<int, unsigned int>(* comm, 0, partition_indices_.size(), partition_indices_.data());
	}
	else
		std::fill(partition_indices_.begin(), partition_indices_.end(), 0);
	
	/* Compute the reordering of the indices into contiguous range for each processor */
	locally_owned_points_partition_.resize(n_procs+1);
	unsigned int next_free_index = 0;
	for (types::subdomain_id m = 0; m<n_procs; ++m)
	{
		locally_owned_points_partition_[m] = next_free_index;
		for (unsigned int i = 0; i<n_points(); ++i)
			if (partition_indices_[i] == m)
			{
				reordered_indices_[i] = next_free_index;
				original_indices_[next_free_index] = i;
				++next_free_index;
			}
	}

	locally_owned_points_partition_[n_procs] = next_free_index;
	n_locally_owned_points_ = locally_owned_points_partition_[my_pid+1]
								- locally_owned_points_partition_[my_pid];
};

template<int dim, int degree>
void
DoFHandler<dim,degree>::make_coarse_partition(std::vector<unsigned int>& partition_indices)
{
	/* Produce a sparsity pattern in CSR format and pass it to the METIS partitioner */
    std::vector<idx_t> int_rowstart(1);
    	int_rowstart.reserve(coarse_map->getNodeNumElements()+1);
    std::vector<idx_t> int_colnums;
    	int_colnums.reserve(coarse_map->getNodeNumElements());


    	/*******************/
    	/* Rows in block 0 */
    	/*******************/

	Teuchos::Array<types::global_index> col_indices;
	for (int m = 0; m < lattice(0).n_vertices; ++m)
	{
		/**
		 * Start with entries corresponding to the right-product by the Hamiltonian.
		 * For this operation,  the first two blocks are actually independent of the last two.
		 */
				/* Block 0 <-> 0 */
		std::vector<unsigned int> 	
		neighbors =	lattice(0).list_neighborhood_indices(
									lattice(0).get_vertex_position(m), 
									this->intra_search_radius);
		for (auto & mm: neighbors)
			col_indices.push_back(mm);
		
				/* Block 1 -> 0 */
		neighbors = lattice(1).list_neighborhood_indices(
									lattice(0).get_vertex_position(m), 
									this->inter_search_radius + std::max(unit_cell(0).bounding_radius,
																			unit_cell(1).bounding_radius));
		for (auto & pp: neighbors)
			col_indices.push_back(pp+lattice(0).n_vertices);


		/**
		 * Next, we add the terms corresponding to the adjoint operation.
		 * Within this coarsened framework we neglect boundary terms linking two neighboring cells.
		 */

				/* Block 0 <-> 0 */
		std::array<int, dim> on_grid = lattice(0).get_vertex_grid_indices(m);
		for (auto & c : on_grid)
			c = -c;
		col_indices.push_back(lattice(0).get_vertex_global_index(on_grid));

				/* Block 1 -> 0 */
		neighbors =	lattice(1).list_neighborhood_indices(
									-lattice(0).get_vertex_position(m), 
									unit_cell(0).bounding_radius+unit_cell(1).bounding_radius);
		for (auto & qq: neighbors) 
			col_indices.push_back(qq + lattice(0).n_vertices);

		/* Fill the sparsity pattern row */
	 	std::sort(col_indices.begin(), col_indices.end());
	 	for (const auto & col: col_indices)
	 		int_colnums.push_back(col);
	 	int_rowstart.push_back(int_colnums.size());
	 }


    	/*******************/
    	/* Rows in block 1 */
    	/*******************/
	for (int n = 0; n < lattice(1).n_vertices; ++n)
	{
			/* Block 0 -> 1 */
		std::vector<unsigned int> 	
		neighbors =	lattice(0).list_neighborhood_indices(
									lattice(1).get_vertex_position(n), 
									this->inter_search_radius + std::max(unit_cell(0).bounding_radius,
																			unit_cell(1).bounding_radius));
		for (auto & qq: neighbors) 
			col_indices.push_back(qq);
			/* Block 1 <-> 1 */
		neighbors = lattice(1).list_neighborhood_indices(
									lattice(1).get_vertex_position(n), this->intra_search_radius );
		for (auto & nn: neighbors) 
			col_indices.push_back(nn+lattice(0).n_vertices);

		/**
		 * Next, we add the terms corresponding to the adjoint operation.
		 * Within this coarsened framework we neglect boundary terms linking two neighboring cells.
		 */

				/* Block 0 -> 1 */	
		neighbors =	lattice(0).list_neighborhood_indices(
									-lattice(1).get_vertex_position(n), 
									unit_cell(0).bounding_radius+unit_cell(1).bounding_radius);
		for (auto & qq: neighbors) 
			col_indices.push_back(qq);

				/* Block 1 <-> 1 */
		std::array<int, dim> on_grid = lattice(1).get_vertex_grid_indices(n);
		for (auto & c : on_grid)
			c = -c;
		col_indices.push_back(lattice(1).get_vertex_global_index(on_grid) + lattice(0).n_vertices);

		/* Fill the sparsity pattern row */
		std::sort(col_indices.begin(), col_indices.end());
	 	for (const auto & col: col_indices)
	 		int_colnums.push_back(col);
	 	int_rowstart.push_back(int_colnums.size());
	}

	idx_t
    n = static_cast<int>(n_points()),
    ncon = 1,
    nparts  = static_cast<int>(n_procs), 
    dummy;                                    

    /* Set Metis options */
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions (options);
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    options[METIS_OPTION_MINCONN] = 1;

      /* Call Metis */
    std::vector<idx_t> int_partition_indices (n_points());
    int ierr = 
    METIS_PartGraphKway(&n, &ncon, &int_rowstart[0], &int_colnums[0],
                                 nullptr, nullptr, nullptr,
                                 &nparts,nullptr,nullptr,&options[0],
                                 &dummy,&int_partition_indices[0]);

    std::copy (int_partition_indices.begin(),
               int_partition_indices.end(),
               partition_indices.begin());
}

template<int dim, int degree>
void
DoFHandler<dim,degree>::distribute_dofs(Teuchos::RCP<const Teuchos::Comm<int> > comm)
{
	/* Initialize basic #dofs information for each lattice point */
	block_[0] = BlockData(	{ lattice(0).n_vertices, 	lattice(1).n_vertices 	},
							{ unit_cell(1).n_nodes, 	unit_cell(1).n_nodes 	},
							{ layer(0).n_orbitals, 		layer(1).n_orbitals 	},
							 original_indices_, locally_owned_points_partition_, comm);

	block_[1] = BlockData(	{ lattice(0).n_vertices, 	lattice(1).n_vertices 	},
							{ unit_cell(0).n_nodes, 	unit_cell(0).n_nodes 	},
							{ layer(0).n_orbitals, 		layer(1).n_orbitals 	},
							 original_indices_, locally_owned_points_partition_, comm);

	/**
	 * Now we know which global dof range is owned by each lattice point system-wide. 
	 * We explore our local lattice points and fill in the PointData structures in each BlockData.
	 */
	for (unsigned int n = locally_owned_points_partition_[my_pid]; n < locally_owned_points_partition_[my_pid+1]; ++n)
	{
		/* Find the block id to determine the size */
		unsigned int 	this_point_index = original_indices_[n];
		unsigned char block_row = 0;
		if (this_point_index >= lattice(0).n_vertices)
		{
			block_row = 1;
			this_point_index -= lattice(0).n_vertices;
		}

		/* Basic info goes into PointData structure */
		for (int block_col = 0; block_col<2; ++block_col)
		{
			block_[block_col].lattice_points.emplace_back(block_col, block_row, this_point_index, n_dofs_);
			PointData& this_point = block_[block_col].lattice_points .back();
			this_point.owned_dofs .add_range(block_[block_col].lattice_point_dof_range [n], 
												block_[block_col].lattice_point_dof_range [n+1]);
			this_point.owned_dofs .compress();

		/* Case of inter-layer blocks: boundary points and interpolation points require more data */
			if (block_col != block_row)
			{
				const auto& this_cell 	= unit_cell(block_row);
				const auto& this_lattice = lattice(block_row);
				std::array<int,dim> 
				this_vertex_indices = this_lattice.get_vertex_grid_indices(this_point_index);

			/* We now deal with the technicalities of boundary points */
				this_point.boundary_lattice_points .reserve(this_cell.n_boundary_nodes);

				for (unsigned int p = 0; p < this_cell.n_boundary_nodes; ++p)
				{
					/* Which cell does this boundary point belong to? */
					auto [cell_index, lattice_indices] = this_cell.map_boundary_point_interior(p);
					for (unsigned int i=0; i<dim; ++i)
						lattice_indices[i] += this_vertex_indices[i];

					/* Find out what is the corresponding lattice point index */
					unsigned int index_in_block = this_lattice.get_vertex_global_index(lattice_indices);
					/* Check that this point exists in our cutout */
					if (index_in_block != types::invalid_lattice_index)
					{
						/* Add the point to the vector of identified boundary points */
						this_point.boundary_lattice_points .push_back(
							std::make_tuple(block_row, index_in_block, cell_index));
					}
					else // The point is out of bounds
						this_point.boundary_lattice_points .push_back(
							std::make_tuple(block_row,
							types::invalid_lattice_index, types::invalid_lattice_index));
				}

			/* We now find and add the interpolation points from the other grid */

				const auto this_point_position 	= this_lattice.get_vertex_position(this_point_index);
				const auto& other_cell 			= unit_cell(block_col);
				const auto& other_lattice 		= lattice(block_col);

				unsigned int estimate_size = other_cell.n_nodes 
							* std::ceil(dealii::determinant(this_cell.basis)/dealii::determinant(other_cell.basis));
				this_point.interpolated_nodes .reserve(estimate_size);

				/* Select and iterate through the relevant neighbors of our current point in the other lattice */
				std::vector<unsigned int> 	
				neighbors = other_lattice.list_neighborhood_indices( -this_point_position, 
									other_cell.bounding_radius+this_cell.bounding_radius);
				for (unsigned int index_in_block : neighbors)
				{
					const dealii::Point<dim> relative_points_position = - (other_lattice.get_vertex_position(index_in_block) + this_point_position);
					
					/* Iterate through the grid points in the corresponding unit cell and test invidually if they are relevant */
					for (unsigned int cell_index = 0; cell_index < other_cell.n_nodes; ++cell_index)
					{
						unsigned int element_index = this_cell.find_element( relative_points_position - other_cell.get_node_position(cell_index));
						/* Test wheter the point is in the cell */
						if (element_index != types::invalid_lattice_index) 
						{
							/* Add the point to the vector of identified interpolated points */
							this_point.interpolated_nodes.push_back(std::make_pair(element_index,
											std::make_tuple(block_col, index_in_block, cell_index)));
						/* Identify block of global degrees of freedom */
							auto [dof_range_start, dof_range_end] = get_dof_range(block_row, block_col, index_in_block, cell_index);
						}
					}
				}
			}
		}
	}
};


template<int dim, int degree>
Teuchos::RCP<const SparsityPattern>
DoFHandler<dim,degree>::make_sparsity_pattern_rmultiply(const unsigned char block_col) const
{
	assert(block_col == 0 || block_col == 1);

	Teuchos::RCP<SparsityPattern> sparsity_pattern = Tpetra::createCrsGraph(block(block_col).owned_dofs);
	Teuchos::Array<types::global_index> ColIndices;

	/* Each column block has a specific unit cell attached */
	const auto& cell = unit_cell(block_col == 0 ? 1 : 0);

	for (unsigned int n=0; n < n_locally_owned_points(); ++n)
	{
		const PointData& this_point = locally_owned_point(block_col, n);
		dealii::Point<dim> this_point_position = lattice(this_point.block_row).get_vertex_position(this_point.index_in_block);

				/* Block b <-> b */
		std::vector<unsigned int> 	
		neighbors =	lattice(this_point.block_row).list_neighborhood_indices(this_point_position, 
																			this->intra_search_radius);
		for (auto neighbor_index_in_block : neighbors)
			for (unsigned int cell_index = 0; cell_index < cell.n_nodes [this_point.block_row]; ++cell_index)
			{
				ColIndices.clear();
				for (unsigned int orbital_middle = 0; orbital_middle <  block(block_col).n_orbitals [this_point.block_row]; orbital_middle++)
					ColIndices.push_back(
						get_dof_index(block_col, this_point.block_row, neighbor_index_in_block, cell_index, orbital_middle));
				for (unsigned int orbital_column = 0; orbital_column <  block(block_col).n_orbitals [this_point.block_row]; orbital_column++)
					sparsity_pattern.insertGlobalIndices(
						get_dof_index(block_col, this_point.block_row, this_point.index_in_block, cell_index, orbital_column), ColIndices);
			}

		unsigned char other_block_row = (this_point.block_row == 0 ? 1 : 0);
				/* Block a -> b, a != b */
		neighbors = lattice(other_block_row).list_neighborhood_indices( this_point_position, 
									this->inter_search_radius + cell.bounding_radius);
				
		for (auto neighbor_index_in_block : neighbors)
			for (unsigned int cell_index = 0; cell_index < cell.n_nodes; ++cell_index)
			{
				/* Note: the unit cell node displacement follows the point which is in the interlayer block */
				dealii::Tensor<1,dim> arrow_vector = lattice(other_block_row).get_vertex_position(neighbor_index_in_block)
													+ (other_block_row != block_col ? 1. : -1.) 
														* cell.get_node_position(cell_index)
													 		- this_point_position;
				
				if ( arrow_vector.norm() < this->inter_search_radius + 1e-10)
				{
					ColIndices.clear();
					for (unsigned int orbital_middle = 0; orbital_middle < block(block_col).n_orbitals [other_block_row]; orbital_middle++)
						ColIndices.push_back(
							get_dof_index(block_col, other_block_row, neighbor_index_in_block, cell_index, orbital_middle));
					for (unsigned int orbital_column = 0; orbital_column < block(block_col).n_orbitals [this_point.block_row]; orbital_column++)
						sparsity_pattern.insertGlobalIndices(
							get_dof_index(block_col, this_point.block_row, this_point.index_in_block, cell_index, orbital_column), ColIndices);
				}
			}
	}
	sparsity_pattern.fillComplete ();
	return sparsity_pattern.getConst ();
};



template<int dim, int degree>
Teuchos::RCP<const SparsityPattern>
DoFHandler<dim,degree>::make_sparsity_pattern_adjoint(const unsigned char block_col, const unsigned char block_row) const
{
	/**
	 * Note : the second step (shift translation, local transpose) is dealt with by FFT after the operation modeled here
	 */
	Teuchos::RCP<SparsityPattern> sparsity_pattern = Tpetra::createCrsGraph(locally_owned_dofs_);
	Teuchos::Array<types::global_index> ColIndices;
	
	for (unsigned int n=0; n<n_locally_owned_points(); ++n)
	{
		const PointData& this_point = locally_owned_point(n);
		
		switch (this_point.block_row) {
							/***********************/
			case 0:			/* Row in block 0 -> 0 */
			{				/***********************/
				std::array<int, dim> on_grid = lattice(0).get_vertex_grid_indices(this_point.index_in_block);
				for (auto & c : on_grid)
					c = -c;
				unsigned int opposite_index_in_block = lattice(0).get_vertex_global_index(on_grid);

				for (unsigned int cell_index = 0; cell_index < unit_cell(1).n_nodes; ++cell_index)
					for (unsigned int orbital = 0; orbital < layer(0).n_orbitals; orbital++)
					{
						types::global_index 
						row = get_dof_index(0, this_point.index_in_block, cell_index, orbital),
						col = get_dof_index(0, opposite_index_in_block, cell_index, orbital);
						sparsity_pattern.insertGlobalIndices(row, 1, &col);
					}
				break;
			}				/***************************/
			case 1:			/* COLUMNS in block 1 -> 2 */
			{				/***************************/

				for (auto & interp_point : this_point.interpolated_nodes)
				{
					unsigned int element_index = interp_point.first;
					auto [row_block_id, row_index_in_block, row_cell_index] = interp_point.second;
					for (unsigned int cell_index : unit_cell(1).subcell_list[element_index].unit_cell_dof_index_map)
					{
						if (unit_cell(1).is_node_interior(cell_index))
						{
							for (unsigned int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
								for (unsigned int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
								{
									dynamic_pattern.add(get_dof_index(2, row_index_in_block, row_cell_index, orbital_column, orbital_row),
														get_dof_index(1, this_point.index_in_block, cell_index, orbital_row, orbital_column));
								}
						}
						else // Boundary point!
						{
							auto [column_block_id, column_index_in_block, column_cell_index] 
										= this_point.boundary_lattice_points[cell_index - unit_cell(1).n_nodes];
							/* Last check to see if the boundary point actually exists on the grid. Maybe some extrapolation would be useful? */
							if (column_index_in_block != types::invalid_lattice_index) 
								for (unsigned int orbital_row = 0; orbital_row < layer(0).n_orbitals; orbital_row++)
									for (unsigned int orbital_column = 0; orbital_column < layer(1).n_orbitals; orbital_column++)
										dynamic_pattern.add(get_dof_index(2, row_index_in_block, row_cell_index, orbital_column, orbital_row),
															get_dof_index(1, column_index_in_block, column_cell_index, orbital_row, orbital_column));
						}
					}
				}

				break;
			}				/***************************/
			case 2:			/* COLUMNS in block 2 -> 1 */
			{				/***************************/
				for (auto & interp_point : this_point.interpolated_nodes)
				{
					unsigned int element_index = interp_point.first;
					auto [ row_block_id, row_index_in_block, row_cell_index] = interp_point.second;

					for (unsigned int cell_index : unit_cell(0).subcell_list[element_index].unit_cell_dof_index_map)
					{
						if (unit_cell(0).is_node_interior(cell_index))
							for (unsigned int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
								for (unsigned int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
									dynamic_pattern.add(get_dof_index(1, row_index_in_block, row_cell_index, orbital_column, orbital_row),
														get_dof_index(2, this_point.index_in_block, cell_index, orbital_row, orbital_column));
						else // Boundary point!
						{
							auto [column_block_id, column_index_in_block, column_cell_index] = this_point.boundary_lattice_points[cell_index - unit_cell(0).n_nodes];
							/* Last check to see if the boundary point exists on the grid. Maybe some extrapolation would be useful? */
							if (column_index_in_block != types::invalid_lattice_index) 
								for (unsigned int orbital_row = 0; orbital_row < layer(1).n_orbitals; orbital_row++)
									for (unsigned int orbital_column = 0; orbital_column < layer(0).n_orbitals; orbital_column++)
										dynamic_pattern.add(get_dof_index(1, row_index_in_block, row_cell_index, orbital_column, orbital_row),
															get_dof_index(2, column_index_in_block, column_cell_index, orbital_row, orbital_column));
						}
					}
				}
				break;
			}				/************************/
			case 3:			/* Rows in block 3 -> 3 */
			{				/************************/
				std::array<int, dim> on_grid = lattice(1).get_vertex_grid_indices(this_point.index_in_block);
				for (auto & c : on_grid)
					c = -c;
				unsigned int opposite_index_in_block = lattice(1).get_vertex_global_index(on_grid);

				for (unsigned int cell_index = 0; cell_index < unit_cell(0).n_nodes; ++cell_index)
					for (unsigned int orbital = 0; orbital < layer(1).n_orbitals; orbital++)
					{
						types::global_index 
						row = get_dof_index(3, this_point.index_in_block, cell_index, orbital),
						col = get_dof_index(3, opposite_index_in_block, cell_index, orbital);
						sparsity_pattern.insertGlobalIndices(row, 1, &col);
					}
				break;
			}
		}
	}
	sparsity_pattern.fillComplete ();
	return sparsity_pattern.getConst ();
};

/* Interface accessors */

template<int dim, int degree>
unsigned int 
DoFHandler<dim,degree>::n_points() const
	{ return n_lattice_points_; };


template<int dim, int degree>
unsigned int 
DoFHandler<dim,degree>::n_locally_owned_points() const
	{ return n_locally_owned_points_; };

template<int dim, int degree>
const PointData &
DoFHandler<dim,degree>::locally_owned_point(	const unsigned char block_col, 	const unsigned int local_index) const
{ 
	assert(local_index < n_locally_owned_points_);
	return block(block_col).lattice_points [local_index]; 
};


template<int dim, int degree>
const PointData &
DoFHandler<dim,degree>::locally_owned_point(	const unsigned char block_col, 	const unsigned char block_row, 
																const unsigned int index_in_block) const
{
	/* Bounds checking */
	assert(block_col < 2);
	assert(block_row < 2);
	assert(index_in_block < lattice(block_row).n_vertices );

	unsigned int local_index = reordered_indices_[start_block_rows_[block_row] + index_in_block]
										- locally_owned_points_partition_[my_pid];
	return block(block_col).lattice_points [local_index];
};


template<int dim, int degree>
bool
DoFHandler<dim,degree>::is_locally_owned_point(unsigned char block_row, const unsigned int index_in_block) const
{
	/* Bounds checking */
	assert(block_row < 2);
	assert(index_in_block < lattice(block_row).n_vertices );

	unsigned int idx;
	if (block_row == 0)
		idx = reordered_indices_[index_in_block];
	else
		idx = reordered_indices_[index_in_block + lattice(0).n_vertices];
	return !(idx < locally_owned_points_partition_[my_pid] || idx >= locally_owned_points_partition_[my_pid+1]);
};

template<int dim, int degree>
types::global_index
DoFHandler<dim,degree>::get_dof_index(const unsigned char block_col, const unsigned char block_row, 
									const unsigned int index_in_block, const unsigned int cell_index, 
									const unsigned int orbital) const
{
	/* Bounds checking */
	assert(	block_col < 2	);
	assert(	block_row < 2	);
	assert(	index_in_block 	< lattice(0).n_vertices	);
	assert(	cell_index 		< block(block_col).n_nodes	[block_row]		);
	assert(		orbital 	< block(block_col).n_orbitals [block_row]	);

	unsigned int idx;
	if (block_row == 0)
		idx = reordered_indices_[index_in_block];
	else
		idx = reordered_indices_[index_in_block + lattice(0).n_vertices];
	return block(block_col).lattice_point_dof_range [idx] 
				+ cell_index * block(block_col).n_orbitals [block_row]
				+ column;
};

template<int dim, int degree>
std::pair<types::global_index,types::global_index>
DoFHandler<dim,degree>::get_dof_range(const unsigned char block_col, const unsigned char block_row, 
									const unsigned int index_in_block, const unsigned int cell_index) const
{
	/* Bounds checking */
	assert(	block_col < 2	);
	assert(	block_row < 2	);
	assert(	index_in_block 	< lattice(0).n_vertices	);
	assert(	cell_index 		< block(block_col).n_nodes	[block_row]		);

	unsigned int idx;
	if (block_row == 0)
		idx = reordered_indices_[index_in_block];
	else
		idx = reordered_indices_[index_in_block + lattice(0).n_vertices];
	types::global_index
	dof_range_start = block(block_col).lattice_point_dof_range [idx] 
						+ cell_index * block(block_col).n_orbitals [block_row];
	return std::make_pair(dof_range_start, dof_range_start + block(block_col).n_orbitals [block_row]);
};

template<int dim, int degree>
std::pair<types::global_index,types::global_index>
DoFHandler<dim,degree>::get_dof_range(const unsigned char block_col, const unsigned char block_row, 
									const unsigned int index_in_block) const
{
	/* Bounds checking */
	assert(	block_col < 2	);
	assert(	block_row < 2	);
	assert(	index_in_block 	< lattice(0).n_vertices	);

	unsigned int idx;
	if (block_row == 0)
		idx = reordered_indices_[index_in_block];
	else
		idx = reordered_indices_[index_in_block + lattice(0).n_vertices];
	return std::make_pair(	block(block_col).lattice_point_dof_range [idx], 
							block(block_col).lattice_point_dof_range [idx+1]);
};

} /* Namespace Bilayer */

#endif /* BILAYER_DOFHANDLER_H */
