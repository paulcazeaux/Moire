/* 
* File:   point_data.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/



#ifndef moire__bilayer_point_data_h
#define moire__bilayer_point_data_h

#include <vector>
#include <array>
#include <utility>

#include "tools/types.h"


namespace Bilayer {
    using types::loc_t;
    using types::glob_t;
    using types::block_t;


    /**
    * A class holding basic information about a lattice point (coarse discretization)
    */
    struct PointData 
    {
    public:
        /**
         * To which block does the point belong? 
         */
        block_t range_block;
        block_t domain_block;

        /**
         * Within this block, which is its index on the corresponding lattice 
         */
        loc_t lattice_index;

        /**
         * Now we need to store details for the interpolation process.
         * These vectors should remain empty if range_block and domain_block are the same
         * (where cells have periodic boundary conditions).
         *
         *
         * First some information about the unit cell's boundary.
         * These points belong to the same block.
         *
         * The tuple fully identifies a point geometrically.
         * First and second index are the block identifiers, 
         * then the lattice point index (within its block), 
         * and the unit cell grid index.
         */
        std::vector<std::tuple<block_t, block_t, loc_t, loc_t>>
                                                    boundary_lattice_points;

        /**
         * Now some information about points we will interpolate to. 
         * These points belong to the other middle block (2 if range_block = 1 and vice versa).
         * First element of the tuple is the element index, and then the grid point information as above
         */
        std::vector<std::tuple<loc_t, block_t, block_t, loc_t, loc_t> >   
                                                    interpolated_nodes;

        PointData(const block_t, const block_t, const loc_t);

    };

    PointData::PointData(   const block_t range_block, const block_t domain_block, 
                            const loc_t lattice_index)
        :
        range_block(range_block), 
        domain_block(domain_block), 
        lattice_index(lattice_index)
    {}

} /* End namespace Bilayer */
#endif
