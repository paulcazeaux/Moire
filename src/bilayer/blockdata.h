/* 
* File:   blockdata.h
* Author: Paul Cazeaux
*
* Created on May 24, 2017, 19:35 PM
*/

#ifndef BILAYER_BLOCKDATA_H
#define BILAYER_BLOCKDATA_H

#include <vector>
#include <array>
#include "tools/types.h"

 /**
 * This class holds basic information about a column block DoF organization.
 * Each such block corresponds to a MultiVector, which holds two row-wise sub-blocks (see dofhandler.h):
 * The degrees of freedom are then organized as two MultiVectors according to the following structure.
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

namespace Bilayer {


struct BlockData {
public:

	std::array<unsigned int, 2> 							n_points;
	unsigned int							 				n_nodes;
	std::array<unsigned int, 2>				 				n_orbitals;

	std::vector<types::global_index> 						lattice_point_dof_range;
	std::vector<PointData> 									lattice_points;

	types::global_index 									n_dofs;
	Teuchos::RCP<const Map> 			 					owned_dofs;

	std::array<types::global_index>, 2> 					n_dofs_row_block;
	std::array<Teuchos::RCP<const Map>, 2>					owned_dofs_row_block;
	std::array<Teuchos::RCP<const Map>, 2>					transpose_row_block_range;

	BlockData(const std::array<unsigned int, 2>& 	n_points, 
				const unsigned int& 				n_nodes, 
				const std::array<unsigned int, 2>& 	n_orbitals, 
				const std::vector<unsigned int>& 	original_indices,
				const std::vector<unsigned int>& 	locally_owned_points_partition,
				Teuchos::RCP<const Teuchos::Comm<int> > comm);
	~BlockData() {};
};


BlockData::BlockData(const std::array<unsigned int, 2>& n_points, 
					const unsigned int& 				n_nodes, 
					const std::array<unsigned int, 2>& 	n_orbitals, 
					const std::vector<unsigned int>& 	original_indices,
					const std::vector<unsigned int>& 	locally_owned_points_partition,
					const unsigned char block_col,
					Teuchos::RCP<const Teuchos::Comm<int> > comm)
	:
	n_points(n_points),
	n_orbitals(n_orbitals),
	n_nodes(n_nodes)
{
	n_dofs_row_block = {n_points[0] * n_nodes * n_orbitals[0],  
						n_points[1] * n_nodes * n_orbitals[1] };
	n_dofs = n_dofs_row_block[0] + n_dofs_row_block[1];
	lattice_point_dof_range.resize(n_points[0]+n_points[1]+1);

	/**
	 * First, each processor assigns degree of freedom ranges for all lattice points
	 * (even those it doesn't own)
	 */
	lattice_point_dof_range[0] = 0;
	types::global_index next_free_dof = 0;
	for (unsigned int n = 0; n<n_points[0]+n_points[1]; n++)
	{
		/* Find the block id to determine the size */
		if (original_indices[n] < n_points[0])
			next_free_dof += n_nodes * n_orbitals[0];
		else
			next_free_dof += n_nodes * n_orbitals[1];
		lattice_point_dof_range[n+1] = next_free_dof;
	}

	if (n_dofs != next_free_dof)
		throw std::runtime_error("Error in the distribution of degrees of freedom to the lattice points, which do not sum up to the right value: " 
			+ std::to_string(next_free_dof)+ " out of expected " + std::to_string(n_dofs) + ".\n");

	/* Create distributed DoF maps */
	my_pid = comm->getRank ();

		/* Find the first local point index belonging to the second row block */
	const unsigned int local_begin = locally_owned_points_partition[my_pid];
	const unsigned int local_end  = locally_owned_points_partition[my_pid+1];
	unsigned int local_sep = local_begin;
	while (original_indices[local_sep] < n_points[0] && local_sep < local_end)
		++local_sep;

	const unsigned int n_local_dofs = lattice_point_dof_range[local_end] - lattice_point_dof_range[local_begin];
	const unsigned int n_local_dofs_row_block[2] = 
			{	lattice_point_dof_range[local_sep] - lattice_point_dof_range[local_begin],
				lattice_point_dof_range[local_end] - lattice_point_dof_range[local_sep]		};

		/* Create Tpetra Maps */
	owned_dofs = createContigMap(n_dofs, n_local_dofs, comm);
	owned_dofs_row_block[0] = createContigMap(n_dofs_row_block[0], n_local_dofs_row_block[0], comm);
	owned_dofs_row_block[1] = createContigMap(n_dofs_row_block[1], n_local_dofs_row_block[1], comm);

	if (block_col == 0)
	{
		transpose_row_block_range[0] = owned_dofs_row_block[0];
		transpose_row_block_range[1] = createContigMap(n_dofs_row_block[0] / n_orbitals[0] * n_orbitals[1] , 
														n_dofs_row_block[0] / n_orbitals[0] * n_orbitals[1], comm);
	}
	else if (block_col == 1)
	{
		transpose_row_block_range[0] = createContigMap(n_dofs_row_block[1] / n_orbitals[1] * n_orbitals[0] , 
														n_dofs_row_block[1] / n_orbitals[1] * n_orbitals[0], comm);
		transpose_row_block_range[1] = owned_dofs_row_block[1];
	}
	else
		throw std::runtime_error("Wrong block column argument in BlockData constructor, block_col = " 
					+ std::to_string(block_col)+ " instead of 0 or 1.\n");
};

} /* Namespace Bilayer */
#endif /* BILAYER_BLOCKDATA_H */