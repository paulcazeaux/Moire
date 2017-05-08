/* 
* File:   grid_to_index_map.h
* Author: Paul Cazeaux
*
* Created on May 1, 2017, 9:00 PM
*/


#ifndef GRID_TO_INDEX_MAP_H
#define GRID_TO_INDEX_MAP_H
#include <vector>
#include <array>

#include "tools/types.h"


/* A simple object to relate grid points belonging to a subset of a rectangular regular grid to linear global indexing  */
template <int dim>
class GridToIndexMap {

public:


	/* Constructors and destructor */
	GridToIndexMap() {};
	GridToIndexMap(const std::vector<std::array<types::grid_index, dim>>& grid_vertices,
								const std::array<types::grid_index, dim> range_min, 
								const std::array<types::grid_index, dim> range_max);
	~GridToIndexMap() {};

	/* Discard the present state of the object and reinitialize with new data */
	void				reinit(const std::vector<std::array<types::grid_index, dim>>& grid_vertices,
								const std::array<types::grid_index, dim> range_min, 
								const std::array<types::grid_index, dim> range_max);

	/* Find the index of a single vertex in the grid */
	types::global_index 				find(std::array<types::grid_index, dim> grid_vertex) const;
	/* Find the index of a rectangular range of vertices in the grid */
	std::vector<types::global_index> 	find(std::array<types::grid_index, dim> search_range_min, 
										std::array<types::grid_index, dim> search_range_max,
										bool include_invalid_vertices) const;

private:
	std::vector<types::global_index> 			global_indices_;
	std::array<types::grid_index, dim>			range_min_;
	std::array<types::grid_index, dim>	 		range_max_;
	std::array<types::grid_index, dim+1>	 	strides_;
};

template <int dim>
GridToIndexMap<dim>::GridToIndexMap(
			const std::vector<std::array<types::grid_index, dim>>& grid_vertices,
			const std::array<types::grid_index, dim> range_min, 
			const std::array<types::grid_index, dim> range_max 	) 
	:
	range_min_(range_min),
	range_max_(range_max)
{
	/* Recursively build the stride information */
	strides_[0] = 1;
	for (unsigned int i=0; i<dim; ++i)
		strides_[i+1] = strides_[i] * (range_max_[i] - range_min_[i] + 1);
	/* Initialize the indices array */
	global_indices_.resize(strides_[dim], types::invalid_global_index);


	/* Populate the indices array with the vector of grid vertices */
	types::global_index size = static_cast<types::global_index>( grid_vertices.size() );
	types::global_index index = 0;
	for (const auto & grid_vertex : grid_vertices)
	{
		unsigned int unrolled_index = 0;
		for (unsigned int i=0; i<dim; ++i)
		{
			/* Bounds checking */
			if (grid_vertex[i] < range_min[i] || grid_vertex[i] > range_max[i])
				throw std::runtime_error("Vertices are out of bounds!");
			unrolled_index = unrolled_index + static_cast<unsigned int>(strides_[i] * (grid_vertex[i] - range_min[i]));
		}
		global_indices_[unrolled_index] = index;
		++index;
	}
};


template <int dim>
void GridToIndexMap<dim>::reinit(
			const std::vector<std::array<types::grid_index, dim>>& grid_vertices,
			const std::array<types::grid_index, dim> range_min, 
			const std::array<types::grid_index, dim> range_max 	)
{
	range_min_ = range_min;
	range_max_ = range_max;

	/* Recursively build the stride information */
	strides_[0] = 1;
	for (unsigned int i=0; i<dim; ++i)
		strides_[i+1] = strides_[i] * (range_max_[i] - range_min_[i] + 1);
	/* Initialize the indices array */
	global_indices_.resize(strides_[dim], types::invalid_global_index);


	/* Populate the indices array with the vector of grid vertices */
	types::global_index size = static_cast<types::global_index>( grid_vertices.size() );
	types::global_index index = 0;
	for (const auto & grid_vertex : grid_vertices)
	{
		unsigned int unrolled_index = 0;
		for (unsigned int i=0; i<dim; ++i)
		{
			/* Bounds checking */
			if (grid_vertex[i] < range_min[i] || grid_vertex[i] > range_max[i])
				throw std::runtime_error("Vertices are out of bounds!");
			unrolled_index = unrolled_index + static_cast<unsigned int>(strides_[i] * (grid_vertex[i] - range_min[i]));
		}
		global_indices_[unrolled_index] = index;
		++index;
	}
};


template <int dim>
types::global_index
GridToIndexMap<dim>::find(const std::array<types::grid_index,dim> grid_vertex ) const
{
	unsigned int unrolled_index = 0;
	for (unsigned int i=0; i<dim; ++i)
	{
		/* Bounds checking */
		if (grid_vertex[i] < range_min_[i] || grid_vertex[i] > range_max_[i])
			return types::invalid_global_index;
		unrolled_index = unrolled_index + static_cast<unsigned int>(strides_[i]*(grid_vertex[i] - range_min_[i]));
	}
	return global_indices_[unrolled_index];
};


template <int dim>
std::vector<types::global_index>
GridToIndexMap<dim>::find(	std::array<types::grid_index, dim> search_range_min, 
							std::array<types::grid_index, dim> search_range_max,
							bool include_invalid_vertices) const
{
	std::vector<types::global_index> neighborhood;

	/* Bounds checking */
	for (unsigned int i=0; i<dim; ++i)
	{
		if (search_range_min[i] < range_min_[i]) search_range_min[i] = range_min_[i];
		if (search_range_max[i] > range_max_[i]) search_range_max[i] = range_max_[i];
		if (search_range_min[i] > search_range_max[i]) return neighborhood; // empty vector
	}
	std::array<types::grid_index, dim+1> 	search_stride;
	search_stride[0] = 1;
	for (unsigned int i=1; i<dim+1; i++)
		search_stride[i] = search_stride[i-1]*(search_range_max[i-1] - search_range_min[i-1] + 1); // Guaranteed to be positive

	unsigned int search_size = search_stride[dim];
	neighborhood.reserve(search_size);

	for (unsigned int search_index = 0; search_index < search_size; ++search_index)
	{
		unsigned int unrolled_index = 0;
		for (unsigned int i = 0; i<dim; ++i)
		{
			types::grid_index coord = (search_index / search_stride[i]) % search_stride[i+1] + search_range_min[i] - range_min_[i];
			unrolled_index = unrolled_index + static_cast<unsigned int>(strides_[i]*coord);
		}
		if (include_invalid_vertices || global_indices_[unrolled_index] != types::invalid_global_index)
			neighborhood.push_back(global_indices_[unrolled_index]);
	}
	neighborhood.shrink_to_fit();
	return neighborhood;
};


#endif /* GRID_TO_INDEX_MAP_H */