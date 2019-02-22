/* 
* File:   bilayer/point_data.cpp
* Author: Paul Cazeaux
*
* Created on June 30, 2017, 9:00 PM
*/

#include "bilayer/point_data.h"

namespace Bilayer {

    PointData::PointData(   const block_t domain_block, 
                            const loc_t lattice_index)
        :
        domain_block(domain_block), 
        lattice_index(lattice_index)
    {}

} /* End namespace Bilayer */
