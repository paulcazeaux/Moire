/* 
* File:   lattice.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/


#ifndef LATTICE_H
#define LATTICE_H

#include <algorithm>
#include <exception>
#include <cmath>
#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

#include "tools/types.h"
#include "tools/grid_to_index_map.h"


/**
 * Geometrical layout of a finite circular cutout of lattice points centered at the origin
 * with a global linear indexing.
 * This class provides in particular a method to determine the indexes of lattice points within the cutout
 * that are within a certain distance of a certain point of the plane.
 */

template <int dim>
class Lattice {

static_assert( (dim == 1 || dim == 2), "Lattice dimension must be 1 or 2!");

public:
	/* Announce to the world the cut-off radius, number of vertices and array basis */
	const double 							radius;
	types::global_index 		 			number_of_vertices;
	const dealii::Tensor<2,dim>				basis;
	const dealii::Tensor<2,dim>				inverse_basis;

	/* Creator and destructor */
	Lattice(const dealii::Tensor<2,dim> basis, const double radius);
	~Lattice() {};

	/* List all indices in a disk-like neighborhood of any radius around any point */
	std::vector<types::global_index> 		list_neighborhood_indices(const dealii::Point<dim>& X, const double radius) const;

	/* Utilities for locating the global index of a vertex from its grid index,
	*	or vice versa the grid index from the global index */

	types::global_index 					get_vertex_global_index(const std::array<types::grid_index, dim>& indices) const;
	std::array<types::grid_index, dim> 		get_vertex_grid_indices(const types::global_index& index) const;
	dealii::Point<dim>						get_vertex_position(const types::global_index& index) const;

private:

		/* Array of vertex positions */
	std::vector<dealii::Point<dim> >				vertices_;	

		/* Maps from global index to grid index */
	GridToIndexMap<dim>								grid_to_index_map_;
	std::vector<std::array<types::grid_index, dim>>	index_to_grid_map_; 

		/* A useful quantity for determining necessary search sizes when looking for neighborhoods */
	const double 									unit_cell_inscribed_radius_;
	static double 									compute_unit_cell_inscribed_radius(const dealii::Tensor<2,dim>& basis);
};




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
	types::grid_index search_size = static_cast<types::grid_index>( std::floor(radius / unit_cell_inscribed_radius_) );

	/* Deduce the strides for each dimension in a corresponding flattened multi-dimensional array */
	types::grid_index  strides[dim+1];
	strides[0] = 1;
	for (unsigned int i = 0; i<dim; i++)
		strides[i+1] = strides[i] * (2*search_size + 1);

	/* Allocate a corresponding amount of memory for the dynamic class members */
	vertices_.reserve( strides[dim] );
	index_to_grid_map_.reserve( strides[dim] );

	/* Walk through the flattened multi-dimensional array and populate the index_to_grid_map_
	* and vertices_ data containers */
	types::grid_index 	indices[dim];
	for (types::grid_index unrolled_index = 0; unrolled_index < strides[dim]; ++unrolled_index)
	{
		for (unsigned int i=0; i<dim; ++i)
			indices[i] = (unrolled_index / strides[i]) % strides[i+1] - search_size;

		dealii::Point<dim> vertex (basis * dealii::Tensor<1,dim,types::grid_index>(indices));
		if (vertex.square() < radius*radius)
			{
				vertices_.push_back(vertex);
				std::array<types::grid_index, dim> indices_array;
				for (unsigned int i=0; i<dim; ++i) 
					indices_array[i] = indices[i];
				index_to_grid_map_.push_back(indices_array);
			}
	}
	vertices_.shrink_to_fit();
	index_to_grid_map_.shrink_to_fit();

	/* Initialize the grid_to_index_map_ data structure */
	std::array<types::grid_index, dim> range_min, range_max;
	for (unsigned int i=0; i<dim; ++i)
	{
		range_min[i] = -search_size; 
		range_max[i] = search_size;
	}
	grid_to_index_map_.reinit(index_to_grid_map_, range_min, range_max);

	/* Initialize the subdomain ids to zero (no partition) */
	number_of_vertices = vertices_.size();
}

template<int dim>
std::vector<types::global_index>
Lattice<dim>::list_neighborhood_indices(const dealii::Point<dim>& X, const double radius) const
{
	/* Rotate X into the grid basis */
	dealii::Point<dim> Xg (inverse_basis * X);
	/* Compute the grid indices range that has to be explored */
	double search_radius = radius / unit_cell_inscribed_radius_;
	std::array<types::grid_index, dim> search_range_min, search_range_max;

	for (unsigned int i = 0; i < dim; ++i) 
	{
		search_range_min[i] = static_cast<types::grid_index>( std::ceil(Xg(i) - search_radius) );
		search_range_max[i] = static_cast<types::grid_index>( std::floor(Xg(i) + search_radius) );
	}
	/* Use the grid_to_index_map_ object to get a list of relevant indices to explore */
	std::vector<types::global_index>	search_range = grid_to_index_map_.find(search_range_min, search_range_max, false);
	
	std::vector<types::global_index> neighborhood;
	neighborhood.reserve(search_range.size());
	for (const auto & idx : search_range)
		if (X.distance(vertices_[idx]) < radius)
			neighborhood.push_back(idx);
	neighborhood.shrink_to_fit();
	return neighborhood;
};





/* Basic getters and setters */
template<int dim>
types::global_index 	
Lattice<dim>::get_vertex_global_index(const std::array<types::grid_index, dim>& indices) const
{ 
	return grid_to_index_map_.find(indices);
};


template<int dim>
std::array<types::grid_index, dim> 	
Lattice<dim>::get_vertex_grid_indices(const types::global_index& index) const
{	return index_to_grid_map_.at(index);	};


template<int dim>
dealii::Point<dim>
Lattice<dim>::get_vertex_position(const types::global_index& index) const
{	return vertices_.at(index);	};


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
};


#endif /* LATTICE_H */