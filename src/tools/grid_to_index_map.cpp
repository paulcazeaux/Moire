/* 
 * File:   periodic_translation_unit.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 29, 2017, 15:02AM
 */

#include "grid_to_index_map.h"

template <int dim>
GridToIndexMap<dim>::GridToIndexMap(
            const std::vector<std::array<types::loc_t, dim>>& grid_vertices,
            const std::array<types::loc_t, dim> range_min, 
            const std::array<types::loc_t, dim> range_max    ) 
    :
    range_min_(range_min),
    range_max_(range_max)
{
    /* Recursively build the stride information */
    strides_[0] = 1;
    for (size_t i=0; i<dim; ++i)
        strides_[i+1] = strides_[i] * (range_max_[i] - range_min_[i] + 1);
    /* Initialize the indices array */
    global_indices_.resize(strides_[dim], types::invalid_local_index);


    /* Populate the indices array with the vector of grid vertices */
    types::loc_t index = 0;
    for (const auto & grid_vertex : grid_vertices)
    {
        types::loc_t unrolled_index = 0;
        for (types::loc_t i=0; i<dim; ++i)
        {
            /* Bounds checking */
            if (grid_vertex[i] < range_min[i] || grid_vertex[i] > range_max[i])
                throw std::runtime_error("Vertices are out of bounds!");
            unrolled_index = unrolled_index + static_cast<types::loc_t>(strides_[i] * (grid_vertex[i] - range_min[i]));
        }
        global_indices_[unrolled_index] = index;
        ++index;
    }
}


template <int dim>
void GridToIndexMap<dim>::reinit(
            const std::vector<std::array<types::loc_t, dim>>& grid_vertices,
            const std::array<types::loc_t, dim> range_min, 
            const std::array<types::loc_t, dim> range_max    )
{
    range_min_ = range_min;
    range_max_ = range_max;

    /* Recursively build the stride information */
    strides_[0] = 1;
    for (size_t i=0; i<dim; ++i)
        strides_[i+1] = strides_[i] * (range_max_[i] - range_min_[i] + 1);
    /* Initialize the indices array */
    global_indices_.resize(strides_[dim], types::invalid_local_index);


    /* Populate the indices array with the vector of grid vertices */
    types::loc_t index = 0;
    for (const auto & grid_vertex : grid_vertices)
    {
        types::loc_t unrolled_index = 0;
        for (size_t i=0; i<dim; ++i)
        {
            /* Bounds checking */
            if (grid_vertex[i] < range_min[i] || grid_vertex[i] > range_max[i])
                throw std::runtime_error("Vertices are out of bounds!");
            unrolled_index = unrolled_index + static_cast<types::loc_t>(strides_[i] * (grid_vertex[i] - range_min[i]));
        }
        global_indices_[unrolled_index] = index;
        ++index;
    }
}


template <int dim>
types::loc_t
GridToIndexMap<dim>::find(const std::array<types::loc_t,dim> grid_vertex ) const
{
    types::loc_t unrolled_index = 0;
    for (size_t i=0; i<dim; ++i)
    {
        /* Bounds checking */
        if (grid_vertex[i] < range_min_[i] || grid_vertex[i] > range_max_[i])
            return types::invalid_local_index;
        unrolled_index = unrolled_index + static_cast<types::loc_t>(strides_[i]*(grid_vertex[i] - range_min_[i]));
    }
    return global_indices_[unrolled_index];
}


template <int dim>
std::vector<types::loc_t>
GridToIndexMap<dim>::find(  std::array<types::loc_t, dim> search_range_min, 
                            std::array<types::loc_t, dim> search_range_max,
                            bool include_invalid_vertices) const
{
    std::vector<types::loc_t> neighborhood;

    /* Bounds checking */
    for (size_t i=0; i<dim; ++i)
    {
        if (search_range_min[i] < range_min_[i]) search_range_min[i] = range_min_[i];
        if (search_range_max[i] > range_max_[i]) search_range_max[i] = range_max_[i];
        if (search_range_min[i] > search_range_max[i]) return neighborhood; // empty vector
    }
    std::array<types::loc_t, dim+1>  search_stride;
    search_stride[0] = 1;
    for (size_t i=1; i<dim+1; i++)
        search_stride[i] = search_stride[i-1]*(search_range_max[i-1] - search_range_min[i-1] + 1); // Guaranteed to be positive

    types::loc_t search_size = search_stride[dim];
    neighborhood.reserve(search_size);

    for (types::loc_t search_index = 0; search_index < search_size; ++search_index)
    {
        types::loc_t unrolled_index = 0;
        for (types::loc_t i = 0; i<dim; ++i)
        {
            types::loc_t coord = (search_index / search_stride[i]) % search_stride[i+1] + search_range_min[i] - range_min_[i];
            unrolled_index = unrolled_index + static_cast<types::loc_t>(strides_[i]*coord);
        }
        if (include_invalid_vertices || global_indices_[unrolled_index] != types::invalid_local_index)
            neighborhood.push_back(global_indices_[unrolled_index]);
    }
    neighborhood.shrink_to_fit();
    return neighborhood;
}



/**
 * Explicit instantiations
 */

template
class GridToIndexMap<1>;
template
class GridToIndexMap<2>;
