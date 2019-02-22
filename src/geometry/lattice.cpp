/* 
 * File:   lattice.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 30, 2017, 6:30 PM
 */


#include "geometry/lattice.h"

/* Constructors */
template<int dim>
Lattice<dim>::Lattice(const dealii::Tensor<2,dim> basis, const double radius)
    :
    radius(radius),
    basis(basis),
    inverse_basis(dealii::invert(basis)),
    unit_cell_inscribed_radius_(Lattice<dim>::compute_unit_cell_inscribed_radius(basis))
{
    /* Compute the grid indices range that has to be explored */
    types::loc_t search_size = static_cast<types::loc_t>( std::floor(radius / unit_cell_inscribed_radius_) );

    /* Deduce the strides for each dimension in a corresponding flattened multi-dimensional array */
    types::loc_t  strides[dim+1];
    strides[0] = 1;
    for (types::loc_t i = 0; i<dim; i++)
        strides[i+1] = strides[i] * (2*search_size + 1);

    /* Allocate a corresponding amount of memory for the dynamic class members */
    vertices_.reserve( strides[dim] );
    index_to_grid_map_.reserve( strides[dim] );

    /* Walk through the flattened multi-dimensional array and populate the index_to_grid_map_
    * and vertices_ data containers */
    types::loc_t     indices[dim];
    for (types::loc_t unrolled_index = 0; unrolled_index < strides[dim]; ++unrolled_index)
    {
        for (types::loc_t i=0; i<dim; ++i)
            indices[i] = (unrolled_index / strides[i]) % strides[i+1] - search_size;

        dealii::Point<dim> vertex (basis * dealii::Tensor<1,dim,types::loc_t>(indices));
        if (vertex.square() < radius*radius)
            {
                vertices_.push_back(vertex);
                std::array<types::loc_t, dim> indices_array;
                for (types::loc_t i=0; i<dim; ++i) 
                    indices_array[i] = indices[i];
                index_to_grid_map_.push_back(indices_array);
            }
    }
    vertices_.shrink_to_fit();
    index_to_grid_map_.shrink_to_fit();

    /* Initialize the grid_to_index_map_ data structure */
    std::array<types::loc_t, dim> range_min, range_max;
    for (types::loc_t i=0; i<dim; ++i)
    {
        range_min[i] = -search_size; 
        range_max[i] = search_size;
    }
    grid_to_index_map_.reinit(index_to_grid_map_, range_min, range_max);

    /* Initialize the subdomain ids to zero (no partition) */
    n_vertices = vertices_.size();
    /* Find the index of the origin and cache it */
    std::array<types::loc_t, dim> lattice_indices_0;
    for (size_t i=0; i<dim; ++i)
        lattice_indices_0[i] = 0;
    index_origin = get_vertex_global_index(lattice_indices_0);
}

template<int dim>
std::vector<types::loc_t>
Lattice<dim>::list_neighborhood_indices(const dealii::Point<dim>& X, const double radius) const
{
    /* Rotate X into the grid basis */
    dealii::Point<dim> Xg (inverse_basis * X);
    /* Compute the grid indices range that has to be explored */
    double search_radius = radius / unit_cell_inscribed_radius_;
    std::array<types::loc_t, dim> search_range_min, search_range_max;

    for (types::loc_t i = 0; i < dim; ++i) 
    {
        search_range_min[i] = static_cast<types::loc_t>( std::ceil(Xg(i) - search_radius) );
        search_range_max[i] = static_cast<types::loc_t>( std::floor(Xg(i) + search_radius) );
    }
    /* Use the grid_to_index_map_ object to get a list of relevant indices to explore */
    std::vector<types::loc_t>   search_range = grid_to_index_map_.find(search_range_min, search_range_max, false);
    
    std::vector<types::loc_t> neighborhood;
    neighborhood.reserve(search_range.size());
    for (const auto & idx : search_range)
        if (X.distance(vertices_[idx]) < radius)
            neighborhood.push_back(idx);
    neighborhood.shrink_to_fit();
    return neighborhood;
}

/* Basic getters and setters */
template<int dim>
types::loc_t    
Lattice<dim>::get_vertex_global_index(const std::array<types::loc_t, dim>& indices) const
{ 
    return grid_to_index_map_.find(indices);
}


template<int dim>
std::array<types::loc_t, dim>    
Lattice<dim>::get_vertex_grid_indices(const types::loc_t& index) const
{   return index_to_grid_map_.at(index);    }


template<int dim>
dealii::Point<dim>
Lattice<dim>::get_vertex_position(const types::loc_t& index) const
{   return vertices_.at(index); }


template<int dim>
types::loc_t
Lattice<dim>::offset_global_index(const types::loc_t& index, const std::array<types::loc_t, dim>& offset) const
{
    std::array<types::loc_t, dim> indices = index_to_grid_map_.at(index);
    for (int i=0; i<dim; ++i)
        indices[i] += offset[i];
    return grid_to_index_map_.find(indices);
}

template<int dim>
std::array<types::loc_t, dim>
Lattice<dim>::round_to_grid_indices(const dealii::Point<dim>& X) const
{
    const dealii::Tensor<1,dim> Xg = inverse_basis * X;
    std::array<types::loc_t, dim> indices;
    for (size_t i = 0; i<dim; ++i)
        indices[i] = std::floor(Xg[i] + 0.5);
    return indices;
}

template<int dim>
types::loc_t
Lattice<dim>::round_to_global_index(const dealii::Point<dim>& X) const
{
    const dealii::Tensor<1,dim> Xg = inverse_basis * X;
    std::array<types::loc_t, dim> indices;
    for (size_t i = 0; i<dim; ++i)
        indices[i] = std::floor(Xg[i] + 0.5);

    return get_vertex_global_index(indices);
}


template<int dim>
double                      
Lattice<dim>::compute_unit_cell_inscribed_radius(const dealii::Tensor<2,dim>& basis)
{
    switch (dim) {
        case 1: return basis[0][0];
        case 2: return dealii::determinant(basis) / std::sqrt(std::max(
                                                basis[0][0]*basis[0][0] + basis[1][0]*basis[1][0],
                                                basis[0][1]*basis[0][1] + basis[1][1]*basis[1][1])
                                                                );
        default: return 0; // Should never happen (dimension is 1 or 2)
    }
}

/**
 * Explicit instantiations
 */

template class Lattice<1>;
template class Lattice<2>;
