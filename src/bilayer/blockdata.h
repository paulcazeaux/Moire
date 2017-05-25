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
	std::array<unsigned int, 2>				 				n_nodes;
	std::array<unsigned int, 2>				 				n_orbitals;

	std::vector<types::global_index> 						lattice_point_dof_range;
	std::vector<PointData> 									lattice_points;

	types::global_index 									n_dofs;
	Teuchos::RCP<const Map> 			 					owned_dofs;

	std::array<types::global_index>, 2> 					n_dofs_block_row;
	std::array<Teuchos::RCP<const Map>, 2>					owned_dofs_block_row;

	BlockData(const std::array<unsigned int, 2>& 	n_points, 
				const std::array<unsigned int, 2>& 	n_nodes, 
				const std::array<unsigned int, 2>& 	n_orbitals, 
				const std::vector<unsigned int>& 	original_indices,
				const std::vector<unsigned int>& 	locally_owned_points_partition,
				Teuchos::RCP<const Teuchos::Comm<int> > comm);
	~BlockData() {};
};


BlockData::BlockData(const std::array<unsigned int, 2>& n_points, 
					const std::array<unsigned int, 2>& 	n_nodes, 
					const std::array<unsigned int, 2>& 	n_orbitals, 
					const std::vector<unsigned int>& 	original_indices,
					const std::vector<unsigned int>& 	locally_owned_points_partition,
					Teuchos::RCP<const Teuchos::Comm<int> > comm)
	:
	n_points(n_points),
	n_orbitals(n_orbitals),
	n_nodes(n_nodes)
{
	n_dofs_block_row = {n_points[0] * n_nodes[0] * n_orbitals[0],  
						n_points[1] * n_nodes[1] * n_orbitals[1] };
	n_dofs = n_dofs_block_row[0] + n_dofs_block_row[1];
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
			next_free_dof += n_nodes[0] * n_orbitals[0];
		else
			next_free_dof += n_nodes[1] * n_orbitals[1];
		lattice_point_dof_range[n+1] = next_free_dof;
	}

	if (n_dofs != next_free_dof)
		throw std::runtime_error("Error in the distribution of degrees of freedom to the lattice points, which do not sum up to the right value: " 
			+ std::to_string(next_free_dof)+ " out of expected " + std::to_string(n_dofs) + ".\n");

		/* Create a distributed DoF map */
	my_pid = comm->getRank ();
	n_procs = comm->getSize ();

	n_owned_dofs_per_proc .resize(n_procs);
	for (types::subdomain_id m = 0; m<n_procs; ++m)
		n_owned_dofs_per_proc [m] =  lattice_point_dof_range [locally_owned_points_partition[m+1]] 
							- lattice_point_dof_range [locally_owned_points_partition[m]];


		/* Find the first local point index belonging to the second row block */
	const unsigned int local_start = lattice_point_dof_range [locally_owned_points_partition[my_pid]];
	const unsigned int local_size = n_owned_dofs_per_proc[my_pid];
	unsigned int local_sep = start;
	while (original_indices[start + local_sep] < n_points[0] && local_sep < local_size)
		++local_sep;

	owned_dofs = createContigMap(n_dofs, local_size, comm);
	owned_dofs_block_row[0] = createContigMap(n_dofs_block_row[0], local_sep, comm);
	owned_dofs_block_row[1] = createContigMap(n_dofs_block_row[1], local_size - local_sep, comm);
};

} /* Namespace Bilayer */
#endif /* BILAYER_BLOCKDATA_H */