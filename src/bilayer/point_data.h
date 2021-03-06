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
#include <tuple>

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
        block_t domain_block;

        /**
         * Within this block, which is its index on the corresponding lattice 
         */
        loc_t lattice_index;

        /**
         * Now some information about points we will interpolate from. 
         * These points belong to the other block (1 if range_block = 0 and vice versa).
         * First element of the tuple is the current cell and block index, 
         * then the block, lattice index and element index of the interpolating element,
         * then the interpolating weights for this point.
         */
        std::vector<std::tuple<loc_t, block_t, block_t, loc_t, loc_t, std::vector<double> > > intra_interpolating_nodes;
        std::vector<std::tuple<loc_t, block_t, block_t, loc_t, loc_t, std::vector<double> > > inter_interpolating_nodes;

        PointData(const block_t, const loc_t);

    };

} /* End namespace Bilayer */
#endif
