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
	 * Type used to identify a grid point and its corresponding global dof block.
	 * First index is the lattice point, second is the unit cell grid index,
	 * then the global dof index block start and end.
	 */
	typedef std::tuple<unsigned int, 
						unsigned int, 
						types::global_index, 
						types::global_index> 	point_indices;

	/* To which block [0-3] does the point belong? */
	unsigned char 								block_id;

	/* What is its size? */
	unsigned int 								size;

	/* Range of global dofs owned by the lattice point (interior dofs) */
	dealii::IndexSet							owned_global_dofs;
	/* Range of global dofs relevant to the lattice point (interior + boundary dofs) */
	dealii::IndexSet							relevant_global_dofs;

	/* Stride associated with each index: orbital column, orbital row, grid point */
	std::array<unsigned int,3> 					strides;

	/**
	 * Now we need to store details for the interpolation process.
	 * These vectors should remain empty if block_id is 0 or 3 
	 * (where cells have periodic boundary conditions).
	 */

	/* First some information about the unit cell's boundary. */
	std::vector<point_indices> 					boundary_lattice_points;

	/* Now some information about points we will interpolate to. */
	std::vector<point_indices>					interpolated_grid_points;
	std::vector<unsigned int> 					interpolating_elements;

	PointData(unsigned char, unsigned int, types::global_index, 
				std::array<unsigned int, 3>);

};

PointData::PointData(unsigned char block_id, 
				unsigned int size, 
				types::global_index total_num_dofs, 
				std::array<unsigned int, 3> strides)
	:
	block_id(block_id), 
	size(size), 
	owned_global_dofs(total_num_dofs), relevant_global_dofs(total_num_dofs),
	strides(strides) {};
};


#endif /* BILAYER_POINTDATA_H */