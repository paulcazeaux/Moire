/* 
 * File:   unitcell.h
 * Author: Paul Cazeaux
 *
 * Created on May 4, 2017, 12:58AM
 */


#ifndef UNITCELL_H
#define UNITCELL_H



#include <algorithm>
#include <exception>
#include <cmath>

#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/utilities.h"

#include "tools/types.h"
#include "tools/grid_to_index_map.h"
#include "tools/transformation.h"
#include "fe/element.h"

/**
 * Discretization of the unit cell A*[-1/2, 1/2)^dim of a given lattice
 * with a global linear indexing of degrees of freedom.
 * The unit cell is paved with continuous Lagrange finite elements with 
 * equidistributed points used for interpolation, forming a regular grid.
 * 
 * The unrefined mesh for such a cell contains 2^dim elements (as in the figure below)
 * which can then be uniformly refined by a constructor argument.
 *
 * We differentiate between 'interior' and 'boundary' grid points!
 * This is needed for implementation of periodic as well as continuous boundary conditions
 * 
 *				|		|		|
 *				o	o	o	o	x 		(1d)
 *
 *						or
 *
 *				x - x - x - x - x					where
 *				|		|		|
 *				o	o	o	o	x						o  => interior point
 *				|		|		|						x  => boundary point
 *				o - o - o - o - x		(2d)
 *				|		|		|						|  => indicates degree 2 finite element
 *				o	o	o	o	x								used for interpolation
 *				|		|		|
 *				o - o - o - o - x
 *
 * Interior grid points are stored first (0 <= index < number_of_interior_grid_points) and then
 * boundary grid points.
 */

template <int dim, int degree>
class UnitCell {

static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!");
static_assert( (degree == 1 || degree == 2 || degree == 3), "Local element degree must be 1, 2 or 3!");

public:

	/* Announce to the world the refinement level, number of grid points and array basis */
	const unsigned int						refinement_level;
	types::global_index 		 			number_of_grid_points;
	types::global_index 		 			number_of_interior_grid_points;
	types::global_index 		 			number_of_boundary_grid_points;

	types::global_index 					number_of_elements;

	const dealii::Tensor<2,dim>				basis;
	const dealii::Tensor<2,dim>				inverse_basis;
	const double							bounding_radius;

	/* Creator and destructor */
	UnitCell(const dealii::Tensor<2,dim> basis, const unsigned int refinement_level);
	~UnitCell() {};


	/* Construction of the list of finite elements paving the unit cell, with each one storing for each support point
	 * the corresponding global index within the cell 
	 */
	std::vector<Element<dim,degree>>			build_subcell_list() const;

	/* Utilities for locating the cell-wide index of a degree of freedom from its grid index,
	 *	or vice versa the grid index from the global index 
	 */
	types::global_index 					get_grid_point_global_index(const std::array<types::grid_index, dim>& indices) const;
	std::array<types::grid_index, dim> 		get_grid_point_grid_indices(const types::global_index& index) const;
	dealii::Point<dim>						get_grid_point_position(const types::global_index& index) const;


	bool 									is_grid_point_interior(const types::global_index& index) const;

private:

		/* Array of grid point positions */
	std::vector<dealii::Point<dim> >					grid_points_;
	

		/* Maps from global index to grid index */
	GridToIndexMap<dim>									grid_to_index_map_;
	std::vector<std::array<types::grid_index, dim>>		index_to_grid_map_;

		/* Static method for computing the bounding radius of the cell */
	static double 	compute_bounding_radius(const dealii::Tensor<2,dim>& basis);
};




/* Constructors */
template<int dim, int degree>
UnitCell<dim,degree>::UnitCell(const dealii::Tensor<2,dim> basis, const unsigned int refinement_level)
	:
	refinement_level(refinement_level),
	basis(basis),
	inverse_basis(dealii::invert(basis)),
	bounding_radius(UnitCell<dim,degree>::compute_bounding_radius(basis))
{
	/* Compute the number of grid points first */
	types::global_index interior_line_count = 2 * degree;
	for (unsigned int i=0; i<refinement_level; ++i) 
		interior_line_count *= 2;

	number_of_grid_points 			= dealii::Utilities::fixed_power<dim, types::global_index>(interior_line_count+1);
	number_of_interior_grid_points 	= dealii::Utilities::fixed_power<dim, types::global_index>(interior_line_count);
	number_of_boundary_grid_points	= number_of_grid_points - number_of_interior_grid_points;
	
	number_of_elements 				= dealii::Utilities::fixed_power<dim, types::global_index>(interior_line_count/degree);

	/* Allocate a corresponding amount of memory for the dynamic class members */
	grid_points_.reserve( number_of_grid_points );
	index_to_grid_map_.reserve( number_of_grid_points );

	/* Deduce the strides for each dimension in a corresponding flattened multi-dimensional array 
	 * for the !! overall !! points grid 
	 */
	types::grid_index  strides[dim+1];
	strides[0] = 1;
	for (unsigned int i = 0; i<dim; i++)
		strides[i+1] = strides[i] * (interior_line_count+1);

	/* Walk through the flattened multi-dimensional array and populate 
	*  the index_to_grid_map_ and grid_points_ data containers */
	std::array<types::grid_index, dim> 	indices;
	dealii::Point<dim> grid_point;

	std::vector<dealii::Point<dim>> boundary_grid_points;
	std::vector<std::array<types::grid_index, dim>>		boundary_index_to_grid_map;
	boundary_grid_points.reserve( number_of_boundary_grid_points );
	boundary_index_to_grid_map.reserve( number_of_boundary_grid_points );

	for (types::global_index unrolled_index = 0; unrolled_index < number_of_grid_points; ++unrolled_index)
	{
		bool is_interior = true;
		for (unsigned int i=0; i<dim; ++i)
		{
			indices[i] = (unrolled_index / strides[i]) % strides[i+1] - (interior_line_count/2);
			grid_point(i) = (double) indices[i] / (double) interior_line_count;
			if (indices[i] == interior_line_count/2) 
				is_interior = false; // Our grid point is on the boundary!
		}
		if (is_interior)
		{
			grid_points_.emplace_back(basis*grid_point);
			index_to_grid_map_.emplace_back(indices);
		}
		else
		{
			boundary_grid_points.emplace_back(basis*grid_point);
			boundary_index_to_grid_map.emplace_back(indices);
		}
	}
	/* Now we add at the end the boundary points */
	for (unsigned int j = 0; j < number_of_boundary_grid_points; ++j)
	{
		grid_points_.push_back(boundary_grid_points[j]);
		index_to_grid_map_.push_back(boundary_index_to_grid_map[j]);
	}

	/* Initialize the grid_to_index_map_ data structure */
	std::array<types::grid_index, dim> range_min, range_max;
	for (unsigned int i=0; i<dim; ++i)
	{
		range_min[i] = - (interior_line_count/2); 
		range_max[i] = interior_line_count/2;
	}
	grid_to_index_map_.reinit(index_to_grid_map_, range_min, range_max);
};

template<int dim,int degree>
std::vector<Element<dim,degree>>			
UnitCell<dim,degree>::build_subcell_list() const
{
	types::global_index line_element_count = 2;
	for (unsigned int i=0; i<refinement_level; ++i) 
		line_element_count *= 2;


	std::vector<Element<dim,degree>> subcells;
	subcells.reserve(number_of_elements);

	std::array<dealii::Point<dim>, Element<dim,degree>::vertices_per_cell>	vertices;
	
	/* Sadly this is impossible to write in a dimension-independent manner (mainly because of correct vertex ordering) */
	switch (dim)
	{
		case 1: {
				for (unsigned int element_index=0; element_index<number_of_elements; ++element_index)
				{
					vertices[0] = grid_points_.at( element_index * degree );
					vertices[1] = grid_points_.at( (element_index + 1) * degree );
					subcells.emplace_back(vertices);
					for (unsigned int i=0; i<degree+1; ++i)
						subcells.back().unit_cell_dof_index_map[i] = element_index * degree + i;
				}
				return subcells;
		}

		case 2:	{
				for (unsigned int element_index=0; element_index<number_of_elements; ++element_index)
				{
					/* Find the indices of vertex 0 */
					std::array<types::grid_index, dim> 	indices;
					indices[0] = degree * (element_index  % line_element_count - (line_element_count/2) );
					indices[1] = degree * (element_index  / line_element_count - (line_element_count/2) );

					/* First enumerate the vertices real-space coordinates */
												vertices[0] = grid_points_.at( grid_to_index_map_.find(indices) );
					indices[0] += degree;		vertices[1] = grid_points_.at( grid_to_index_map_.find(indices) );
					indices[1] += degree;		vertices[2] = grid_points_.at( grid_to_index_map_.find(indices) );
					indices[0] -= degree;		vertices[3] = grid_points_.at( grid_to_index_map_.find(indices) );
					indices[1] -= degree;

					/* Create the element */
					subcells.emplace_back(vertices);

					/* Assign dof index */
					for (unsigned int j=0; j<degree+1; ++j)
					{
						for (unsigned int i=0; i<degree+1; ++i)
						{
							subcells.back().unit_cell_dof_index_map[(degree+1)*j + i] = grid_to_index_map_.find( indices );
							++indices[0];
						}
						indices[0] -= degree+1;
						++indices[1];
					}
				}
				return subcells;
		}
	}
};

/* Basic getters and setters */
template<int dim,int degree>
types::global_index 	
UnitCell<dim,degree>::get_grid_point_global_index(const std::array<types::grid_index, dim>& indices) const
{ 
	return grid_to_index_map_.find(indices);
};


template<int dim,int degree>
std::array<types::grid_index, dim> 	
UnitCell<dim,degree>::get_grid_point_grid_indices(const types::global_index& index) const
{	return index_to_grid_map_.at(index);	};


template<int dim,int degree>
dealii::Point<dim>
UnitCell<dim,degree>::get_grid_point_position(const types::global_index& index) const
{	return grid_points_.at(index);	};


template<int dim,int degree>
bool 
UnitCell<dim,degree>::is_grid_point_interior(const types::global_index& index) const
{	return (index < number_of_interior_grid_points);	};


template<int dim,int degree>
double 	
UnitCell<dim,degree>::compute_bounding_radius(const dealii::Tensor<2,dim>& basis)
{
		switch (dim) {
		case 1: return .5 * basis[0][0];
		case 2: return .5 * std::sqrt(basis.norm_square() + 2. * std::abs(	basis[0][0] * basis[0][1] + basis[1][0] * basis[1][1]) );
		default: return 0; // Should never happen (dimension is 1 or 2)
	}
};



#endif /* UNITCELL_H */
