/* 
 * File:   unit_cell.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 30, 2017, 6:30 PM
 */


#include "geometry/unit_cell.h"

/* Constructors */
template<int dim, int degree>
UnitCell<dim,degree>::UnitCell(const dealii::Tensor<2,dim> basis, const types::loc_t refinement_level)
    :
    refinement_level(refinement_level),
    basis(basis),
    inverse_basis(dealii::invert(basis)),
    bounding_radius(UnitCell<dim,degree>::compute_bounding_radius(basis)),
    area(dealii::determinant(basis))
{
    /* Compute the number of grid points first */
    types::loc_t interior_line_count = 2 * degree;
    for (types::loc_t i=0; i<refinement_level; ++i) 
        interior_line_count *= 2;

    n_closure_nodes     = dealii::Utilities::fixed_power<dim, types::loc_t>(interior_line_count+1);
    n_nodes             = dealii::Utilities::fixed_power<dim, types::loc_t>(interior_line_count);
    n_boundary_nodes    = n_closure_nodes - n_nodes;
    for (size_t i =0; i<dim; ++i)
        n_nodes_per_dim[i] = interior_line_count;
    
    n_elements          = dealii::Utilities::fixed_power<dim, types::loc_t>(interior_line_count/degree);

    /* Allocate a corresponding amount of memory for the dynamic class members */
    nodes_.reserve( n_closure_nodes );
    index_to_grid_map_.reserve( n_closure_nodes );

    /* Deduce the strides for each dimension in a corresponding flattened multi-dimensional array 
     * for the !! overall !! points grid 
     */
    types::loc_t  strides[dim+1];
    strides[0] = 1;
    for (size_t i = 0; i<dim; i++)
        strides[i+1] = strides[i] * (interior_line_count+1);

    /* Walk through the flattened multi-dimensional array and populate 
    *  the index_to_grid_map_ and nodes_ data containers */
    std::array<types::loc_t, dim>    indices;
    dealii::Point<dim> node;

    std::vector<dealii::Point<dim>>     boundary_nodes;
    std::vector<std::array<types::loc_t, dim>>   boundary_index_to_grid_map;
    boundary_nodes.reserve( n_boundary_nodes );
    boundary_index_to_grid_map.reserve( n_boundary_nodes );

    for (types::loc_t unrolled_index = 0; unrolled_index < n_closure_nodes; ++unrolled_index)
    {
        bool is_interior = true;
        for (size_t i=0; i<dim; ++i)
        {
            indices[i] = static_cast<types::loc_t>( (unrolled_index / strides[i]) % strides[i+1]) - static_cast<types::loc_t>(interior_line_count/2);
            node(i) = static_cast<double>( indices[i] ) / static_cast<double>( interior_line_count );
            if (indices[i] == static_cast<types::loc_t>( interior_line_count/2 )) 
                is_interior = false; // Our grid point is on the boundary!
        }
        if (is_interior)
        {
            nodes_.emplace_back(basis*node);
            index_to_grid_map_.emplace_back(indices);
        }
        else
        {
            boundary_nodes.emplace_back(basis*node);
            boundary_index_to_grid_map.emplace_back(indices);
        }
    }
    /* Now we add at the end the boundary points */
    for (types::loc_t j = 0; j < n_boundary_nodes; ++j)
    {
        nodes_.push_back(boundary_nodes[j]);
        index_to_grid_map_.push_back(boundary_index_to_grid_map[j]);
    }

    /* Initialize the grid_to_index_map_ data structure */
    std::array<types::loc_t, dim> range_min, range_max;
    for (size_t i=0; i<dim; ++i)
    {
        range_min[i] = - (interior_line_count/2); 
        range_max[i] = interior_line_count/2;
    }
    grid_to_index_map_.reinit(index_to_grid_map_, range_min, range_max);

    /* Initialize the interior index map (wrapping boundary back inside) */
    boundary_to_interior_map_.resize(n_boundary_nodes);
    for (types::loc_t j = 0; j < n_boundary_nodes; ++j)
    {
        indices = boundary_index_to_grid_map[j];
        std::array<types::loc_t,dim> grid_offset;
        for (size_t i=0; i<dim; ++i)
            if (indices[i] == static_cast<types::loc_t>( interior_line_count/2 ))
            {
                indices[i] = -indices[i];
                grid_offset[i] = 1;
            }
            else
                grid_offset[i] = 0;
        boundary_to_interior_map_[j] = std::make_tuple(grid_to_index_map_.find(indices), grid_offset);
    }

    /* Cache the number of elements appearing in a line as it is used by the find_element_index function */
    line_element_count_ = interior_line_count / degree;

    /* Build the list of elements */

    subcell_list.reserve(n_elements);

    std::array<dealii::Point<dim>, Element<dim,degree>::vertices_per_cell>  vertices;
    
    /* Sadly this is impossible to write in a dimension-independent manner (mainly because of correct vertex ordering) */
    switch (dim)
    {
        case 1: 
        {
            for (types::loc_t element_index=0; element_index<n_elements; ++element_index)
            {
                vertices[0] = nodes_.at( element_index * degree );
                vertices[1] = nodes_.at( (element_index + 1) * degree );
                subcell_list.emplace_back(vertices);
                for (size_t i=0; i<degree+1; ++i)
                    subcell_list.back().unit_cell_dof_index_map[i] = element_index * degree + i;
            }
        break;
        }

        case 2: 
        {
            for (types::loc_t element_index=0; element_index<n_elements; ++element_index)
            {
                /* Find the indices of vertex 0 */
                std::array<types::loc_t, dim>    indices;
                indices[0] = degree * (element_index  % line_element_count_ - (line_element_count_/2) );
                indices[1] = degree * (element_index  / line_element_count_ - (line_element_count_/2) );

                /* First enumerate the vertices real-space coordinates */
                                            vertices[0] = nodes_.at( grid_to_index_map_.find(indices) );
                indices[0] += degree;       vertices[1] = nodes_.at( grid_to_index_map_.find(indices) );
                indices[1] += degree;       vertices[2] = nodes_.at( grid_to_index_map_.find(indices) );
                indices[0] -= degree;       vertices[3] = nodes_.at( grid_to_index_map_.find(indices) );
                indices[1] -= degree;

                /* Create the element */
                subcell_list.emplace_back(vertices);

                /* Assign dof index */
                for (size_t j=0; j<degree+1; ++j)
                {
                    for (size_t i=0; i<degree+1; ++i)
                    {
                        subcell_list.back().unit_cell_dof_index_map[(degree+1)*j + i] = grid_to_index_map_.find( indices );
                        ++indices[0];
                    }
                    indices[0] -= degree+1;
                    ++indices[1];
                }
            }
            break;
        }
    }
}

template<int dim,int degree>
types::loc_t
UnitCell<dim,degree>::find_element(const dealii::Tensor<1,dim>& X) const
{
    dealii::Point<dim> Xg (inverse_basis * X);
    switch (dim) {
        case 1: 
        {
            Xg(0) += .5;
            return static_cast<types::loc_t>(    Xg(0) > 0 ? 
                                                (Xg(0) < 1 ? 
                                                    std::floor(line_element_count_ * Xg(0)) 
                                                    : line_element_count_ - 1) 
                                                    : 0 );
        }
        case 2:
        {
            Xg(0) += .5;
            Xg(1) += .5;
            return static_cast<types::loc_t>(   (Xg(0) > 0 ? 
                                                (Xg(0) < 1 ? 
                                                    std::floor(line_element_count_ * Xg(0)) 
                                                    : line_element_count_ - 1) 
                                                    : 0 )
                        + line_element_count_ * (Xg(1) > 0 ? 
                                                (Xg(1) < 1 ? 
                                                    std::floor(line_element_count_ * Xg(1)) 
                                                    : line_element_count_ - 1) 
                                                    : 0 ));            
        }
    }
}

/* Basic getters and setters */
template<int dim,int degree>
types::loc_t    
UnitCell<dim,degree>::get_node_global_index(const std::array<types::loc_t, dim>& indices) const
{ 
    return grid_to_index_map_.find(indices);
}


template<int dim,int degree>
std::array<types::loc_t, dim>    
UnitCell<dim,degree>::get_node_grid_indices(const types::loc_t& index) const
{   return index_to_grid_map_.at(index);    }


template<int dim,int degree>
dealii::Point<dim>
UnitCell<dim,degree>::get_node_position(const types::loc_t& index) const
{   return nodes_.at(index);    }


template<int dim,int degree>
bool 
UnitCell<dim,degree>::is_node_interior(const types::loc_t& index) const
{   return (index < n_nodes);   }

template<int dim, int degree>
std::tuple<types::loc_t, std::array<types::loc_t, dim>>
UnitCell<dim,degree>::map_boundary_point_interior(const types::loc_t& index) const
{   return boundary_to_interior_map_[index]; }

template<int dim,int degree>
double  
UnitCell<dim,degree>::compute_bounding_radius(const dealii::Tensor<2,dim>& basis)
{
        switch (dim) {
        case 1: return .5 * basis[0][0];
        case 2: return .5 * std::sqrt(basis.norm_square() + 2. * std::abs(  basis[0][0] * basis[0][1] + basis[1][0] * basis[1][1]) );
        default: return 0; // Should never happen (dimension is 1 or 2)
    }
}


/**
 * Explicit instantiations
 */

template class UnitCell<1,1>;
template class UnitCell<1,2>;
template class UnitCell<1,3>;
template class UnitCell<2,1>;
template class UnitCell<2,2>;
template class UnitCell<2,3>;