/* 
* File:   pointdata.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/



#ifndef BILAYER_POINTDATA_H
#define BILAYER_POINTDATA_H

#include <vector>
#include <array>
#include <utility>

#include "deal.II/base/index_set.h"
#include "tools/types.h"

/**
* This class holds basic information about a lattice point (coarse discretization)
*/

namespace Bilayer {


struct PointData 
{
public:
	/**
	 * Type used to fully identify a grid point geometrically.
	 * First index is the block index, then the lattice point (within its block), 
	 * and the unit cell grid index.
	 */
	typedef std::tuple<	unsigned char,
						unsigned int, 
						unsigned int> 			point_indices;

	/* To which block does the point belong? */
	unsigned char 								block_col;
	unsigned char 								block_row;

	/* Within this block, which is its index on the corresponding lattice */
	unsigned int 								index_in_block;

	/* Range of global dofs owned by the lattice point (interior dofs) */
	dealii::IndexSet							owned_dofs;

	/**
	 * Now we need to store details for the interpolation process.
	 * These vectors should remain empty if block_row and block_col are the same
	 * (where cells have periodic boundary conditions).
	 *
	 *
	 * First some information about the unit cell's boundary.
	 * These points belong to the same block
	 */
	std::vector<point_indices> 					boundary_lattice_points;

	/**
	 * Now some information about points we will interpolate to. 
	 * These points belong to the other middle block (2 if block_row = 1 and vice versa).
	 * First element of the pair is the element index, second the grid point information as above
	 */
	std::vector<std::pair<unsigned int, point_indices>	>	interpolated_nodes;

	PointData(const unsigned char, const unsigned char, const unsigned int, const types::global_index);

};

PointData::PointData(	const unsigned char block_col, const unsigned char block_row, 
						const unsigned int index_in_block, const types::global_index total_n_dofs)
	:
	block_col(block_col), 
	block_row(block_row), 
	index_in_block(index_in_block),
	owned_dofs(total_n_dofs)
{};

} /* Namespace Bilayer */
#endif /* BILAYER_POINTDATA_H */