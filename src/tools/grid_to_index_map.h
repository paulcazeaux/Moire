/* 
* File:   grid_to_index_map.h
* Author: Paul Cazeaux
*
* Created on May 1, 2017, 9:00 PM
*/


#ifndef moire__tools_grid_to_index_map_h
#define moire__tools_grid_to_index_map_h

#include <vector>
#include <array>

#include "tools/types.h"

/* A simple object to relate grid points belonging to a subset of a rectangular regular grid to linear global indexing  */
template <int dim>
class GridToIndexMap {

public:

    /* Constructors and destructor */
    GridToIndexMap() {};
    GridToIndexMap(const std::vector<std::array<types::loc_t, dim>>& grid_vertices,
                                const std::array<types::loc_t, dim> range_min, 
                                const std::array<types::loc_t, dim> range_max);
    ~GridToIndexMap() {};

    /* Discard the present state of the object and reinitialize with new data */
    void                reinit(const std::vector<std::array<types::loc_t, dim>>& grid_vertices,
                                const std::array<types::loc_t, dim> range_min, 
                                const std::array<types::loc_t, dim> range_max);

    /* Find the index of a single vertex in the grid */
    types::loc_t                find(std::array<types::loc_t, dim> grid_vertex) const;
    /* Find the index of a rectangular range of vertices in the grid */
    std::vector<types::loc_t>   find(std::array<types::loc_t, dim> search_range_min, 
                                        std::array<types::loc_t, dim> search_range_max,
                                        bool include_invalid_vertices) const;

private:
    std::vector<types::loc_t>           global_indices_;
    std::array<types::loc_t, dim>       range_min_;
    std::array<types::loc_t, dim>       range_max_;
    std::array<types::loc_t, dim+1>     strides_;
};

#endif
