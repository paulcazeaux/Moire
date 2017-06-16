/* 
 * File:   unitcell.h
 * Author: Paul Cazeaux
 *
 * Created on May 4, 2017, 12:58AM
 */


#ifndef moire__geometry_unitcell_h
#define moire__geometry_unitcell_h



#include <algorithm>
#include <exception>
#include <cmath>

#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/utilities.h"

#include "tools/types.h"
#include "tools/grid_to_index_map.h"
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
 * We differentiate between 'interior' (owned by the cell) and 'boundary' grid points!
 * This is needed for implementation of periodic as well as continuous boundary conditions
 * 
 *				|		|		|
 *				o	o	o	o	x 		(1d)
 *
 *						or
 *
 *				x - x - x - x - x					where
 *				|		|		|
 *				o	o	o	o	x						o  => interior node (owned by the cell)
 *				|		|		|						x  => boundary node
 *				o - o - o - o - x		(2d)
 *				|		|		|						|  => indicates degree 2 finite element
 *				o	o	o	o	x								used for interpolation
 *				|		|		|
 *				o - o - o - o - x
 *
 * Interior grid nodes are stored first (0 <= index < n_nodes) and then
 * boundary grid nodes.
 */

template <int dim, int degree>
class UnitCell {

static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!");
static_assert( (degree == 1 || degree == 2 || degree == 3), "Local element degree must be 1, 2 or 3!");

public:

	/* Announce to the world the refinement level, number of grid points and array basis */
	const unsigned int						refinement_level;
	unsigned int 		 					n_closure_nodes;
	unsigned int 		 					n_nodes;
	unsigned int 		 					n_boundary_nodes;

	/* Needed for the FFT */
	std::array<int,dim>						n_nodes_per_dim;

	/* A list of finite elements paving the unit cell, with each one storing for each support point
	 * the corresponding global index within the cell 
	 */
	unsigned int 							n_elements;
	std::vector<Element<dim,degree>>		subcell_list;

	const dealii::Tensor<2,dim>				basis;
	const dealii::Tensor<2,dim>				inverse_basis;
	const double							bounding_radius;
	const double 							area;

	/* Creator and destructor */
	UnitCell(const dealii::Tensor<2,dim> basis, const unsigned int refinement_level);
	~UnitCell() {};

	unsigned int 							find_element(const dealii::Tensor<1,dim>& X) const;

	/**
	 * Utilities for locating the cell-wide index of a degree of freedom from its grid index,
	 * or vice versa the grid index from the global index 
	 */
	unsigned int 							get_node_global_index(const std::array<int, dim>& indices) const;
	std::array<int, dim> 					get_node_grid_indices(const unsigned int& index) const;
	dealii::Point<dim>						get_node_position(const unsigned int& index) const;


	bool 									is_node_interior(const unsigned int& index) const;
	std::tuple<unsigned int, std::array<int, dim>>
											map_boundary_point_interior(const unsigned int& index) const;

private:

	/* Array of grid point positions */
	std::vector<dealii::Point<dim> >		nodes_;

	/* Maps from global index to grid index */
	GridToIndexMap<dim>						grid_to_index_map_;
	std::vector<std::array<int, dim>>		index_to_grid_map_;

	/* Map boundary grid point indexes to interior points in another cell and the corresponding lattice grid offset */
	std::vector<std::tuple<unsigned int, std::array<int, dim>>> 	boundary_to_interior_map_;

	/* Cache number of elements along one direction */
	unsigned int 							line_element_count_;

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
	bounding_radius(UnitCell<dim,degree>::compute_bounding_radius(basis)),
	area(dealii::determinant(basis))
{
	/* Compute the number of grid points first */
	unsigned int interior_line_count = 2 * degree;
	for (unsigned int i=0; i<refinement_level; ++i) 
		interior_line_count *= 2;

	n_closure_nodes 	= dealii::Utilities::fixed_power<dim, unsigned int>(interior_line_count+1);
	n_nodes 			= dealii::Utilities::fixed_power<dim, unsigned int>(interior_line_count);
	n_boundary_nodes	= n_closure_nodes - n_nodes;
	for (unsigned int i =0; i<dim; ++i)
		n_nodes_per_dim[i] = interior_line_count;
	
	n_elements 			= dealii::Utilities::fixed_power<dim, unsigned int>(interior_line_count/degree);

	/* Allocate a corresponding amount of memory for the dynamic class members */
	nodes_.reserve( n_closure_nodes );
	index_to_grid_map_.reserve( n_closure_nodes );

	/* Deduce the strides for each dimension in a corresponding flattened multi-dimensional array 
	 * for the !! overall !! points grid 
	 */
	unsigned int  strides[dim+1];
	strides[0] = 1;
	for (unsigned int i = 0; i<dim; i++)
		strides[i+1] = strides[i] * (interior_line_count+1);

	/* Walk through the flattened multi-dimensional array and populate 
	*  the index_to_grid_map_ and nodes_ data containers */
	std::array<int, dim> 	indices;
	dealii::Point<dim> node;

	std::vector<dealii::Point<dim>> 	boundary_nodes;
	std::vector<std::array<int, dim>>	boundary_index_to_grid_map;
	boundary_nodes.reserve( n_boundary_nodes );
	boundary_index_to_grid_map.reserve( n_boundary_nodes );

	for (unsigned int unrolled_index = 0; unrolled_index < n_closure_nodes; ++unrolled_index)
	{
		bool is_interior = true;
		for (unsigned int i=0; i<dim; ++i)
		{
			indices[i] = static_cast<int>( (unrolled_index / strides[i]) % strides[i+1]) - static_cast<int>(interior_line_count/2);
			node(i) = static_cast<double>( indices[i] ) / static_cast<double>( interior_line_count );
			if (indices[i] == static_cast<int>( interior_line_count/2 )) 
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
	for (unsigned int j = 0; j < n_boundary_nodes; ++j)
	{
		nodes_.push_back(boundary_nodes[j]);
		index_to_grid_map_.push_back(boundary_index_to_grid_map[j]);
	}

	/* Initialize the grid_to_index_map_ data structure */
	std::array<int, dim> range_min, range_max;
	for (unsigned int i=0; i<dim; ++i)
	{
		range_min[i] = - (interior_line_count/2); 
		range_max[i] = interior_line_count/2;
	}
	grid_to_index_map_.reinit(index_to_grid_map_, range_min, range_max);

	/* Initialize the interior index map (wrapping boundary back inside) */
	boundary_to_interior_map_.resize(n_boundary_nodes);
	for (unsigned int j = 0; j < n_boundary_nodes; ++j)
	{
		indices = boundary_index_to_grid_map[j];
		std::array<int,dim> grid_offset;
		for (unsigned int i=0; i<dim; ++i)
			if (indices[i] == static_cast<int>( interior_line_count/2 ))
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

	std::array<dealii::Point<dim>, Element<dim,degree>::vertices_per_cell>	vertices;
	
	/* Sadly this is impossible to write in a dimension-independent manner (mainly because of correct vertex ordering) */
	switch (dim)
	{
		case 1: 
		{
			for (unsigned int element_index=0; element_index<n_elements; ++element_index)
			{
				vertices[0] = nodes_.at( element_index * degree );
				vertices[1] = nodes_.at( (element_index + 1) * degree );
				subcell_list.emplace_back(vertices);
				for (unsigned int i=0; i<degree+1; ++i)
					subcell_list.back().unit_cell_dof_index_map[i] = element_index * degree + i;
			}
		break;
		}

		case 2:	
		{
			for (unsigned int element_index=0; element_index<n_elements; ++element_index)
			{
				/* Find the indices of vertex 0 */
				std::array<int, dim> 	indices;
				indices[0] = degree * (element_index  % line_element_count_ - (line_element_count_/2) );
				indices[1] = degree * (element_index  / line_element_count_ - (line_element_count_/2) );

				/* First enumerate the vertices real-space coordinates */
											vertices[0] = nodes_.at( grid_to_index_map_.find(indices) );
				indices[0] += degree;		vertices[1] = nodes_.at( grid_to_index_map_.find(indices) );
				indices[1] += degree;		vertices[2] = nodes_.at( grid_to_index_map_.find(indices) );
				indices[0] -= degree;		vertices[3] = nodes_.at( grid_to_index_map_.find(indices) );
				indices[1] -= degree;

				/* Create the element */
				subcell_list.emplace_back(vertices);

				/* Assign dof index */
				for (unsigned int j=0; j<degree+1; ++j)
				{
					for (unsigned int i=0; i<degree+1; ++i)
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
unsigned int
UnitCell<dim,degree>::find_element(const dealii::Tensor<1,dim>& X) const
{
	dealii::Point<dim> Xg (inverse_basis * X);
	bool test_in_cell = true;
	for (unsigned int i=0; i<dim; ++i)
		test_in_cell = test_in_cell && ( Xg(i) + .5 >= 0) && ( Xg(i) + .5 < 1);
	if (test_in_cell)
		switch (dim) {
			case 1: 
				return static_cast<unsigned int>(std::floor(line_element_count_ * (Xg(0) + .5) ));
			case 2:
				return static_cast<unsigned int>(std::floor(line_element_count_ * (Xg(0) + .5) )
						+ std::floor(line_element_count_ * (Xg(1) + .5)) * line_element_count_);
		}
	else 
		return types::invalid_lattice_index;
}

/* Basic getters and setters */
template<int dim,int degree>
unsigned int 	
UnitCell<dim,degree>::get_node_global_index(const std::array<int, dim>& indices) const
{ 
	return grid_to_index_map_.find(indices);
}


template<int dim,int degree>
std::array<int, dim> 	
UnitCell<dim,degree>::get_node_grid_indices(const unsigned int& index) const
{	return index_to_grid_map_.at(index);	}


template<int dim,int degree>
dealii::Point<dim>
UnitCell<dim,degree>::get_node_position(const unsigned int& index) const
{	return nodes_.at(index);	}


template<int dim,int degree>
bool 
UnitCell<dim,degree>::is_node_interior(const unsigned int& index) const
{	return (index < n_nodes);	}

template<int dim, int degree>
std::tuple<unsigned int, std::array<int, dim>>
UnitCell<dim,degree>::map_boundary_point_interior(const unsigned int& index) const
{	return boundary_to_interior_map_[index]; }

template<int dim,int degree>
double 	
UnitCell<dim,degree>::compute_bounding_radius(const dealii::Tensor<2,dim>& basis)
{
		switch (dim) {
		case 1: return .5 * basis[0][0];
		case 2: return .5 * std::sqrt(basis.norm_square() + 2. * std::abs(	basis[0][0] * basis[0][1] + basis[1][0] * basis[1][1]) );
		default: return 0; // Should never happen (dimension is 1 or 2)
	}
}




#endif
