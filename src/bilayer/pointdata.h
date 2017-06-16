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
    typedef std::tuple< unsigned char,
                        unsigned int, 
                        unsigned int>           point_indices;

    /* To which block [0-3] does the point belong? */
    unsigned char                               block_id;

    /* Within this block, which is its index on the corresponding lattice */
    unsigned int                                index_in_block;

    /* Range of global dofs owned by the lattice point (interior dofs) */
    dealii::IndexSet                            owned_dofs;
    /* Range of global dofs relevant to the lattice point (interior + boundary dofs) */
    dealii::IndexSet                            relevant_dofs;

    /**
     * Now we need to store details for the interpolation process.
     * These vectors should remain empty if block_id is 0 or 3 
     * (where cells have periodic boundary conditions).
     *
     *
     *
     * First some information about the unit cell's boundary.
     * These points belong to the same block
     */
    std::vector<point_indices>                  boundary_lattice_points;

    /**
     * Now some information about points we will interpolate to. 
     * These points belong to the other middle block (2 if block_id = 1 and vice versa).
     * First element of the pair is the element index, second the grid point information as above
     */
    std::vector<std::pair<unsigned int, point_indices>  >   interpolated_nodes;

    PointData(unsigned char, unsigned int, types::global_index);

};

PointData::PointData(unsigned char block_id, unsigned int index_in_block, types::global_index total_n_dofs)
    :
    block_id(block_id), 
    index_in_block(index_in_block),
    owned_dofs(total_n_dofs), 
    relevant_dofs(total_n_dofs) 
{}

} /* End namespace Bilayer */
#endif /* BILAYER_POINTDATA_H */