/* 
* File:   element.h
* Author: Paul Cazeaux
*
* Created on May 4, 2017, 5:00 PM
*/



#ifndef moire__fe_element_h
#define moire__fe_element_h

#include <array>
#include <vector>

#include "deal.II/base/tensor.h"
#include "deal.II/base/point.h"
#include "deal.II/base/utilities.h"

#include "tools/types.h"

/*		Generic template		*/

template<int dim, int degree>
struct Element
{
	static_assert( (dim == 1 || dim == 2), "Element dimension must be 1 or 2!");
	static_assert( (degree == 1 || degree == 2 || degree == 3), "Element degree must be 1, 2 or 3!");
};


/* One should use the specializations below */

/*********************************/
/*         Dimension one         */
/*********************************/
/*	   
 *   Vertices and geometry:
 *
 *     0           1
 *		o - - - - o
 */


template<int degree>
struct Element<1,degree>
{
	static const types::loc_t 					dofs_per_cell = degree+1;
	static const types::loc_t 					vertices_per_cell = 2;

	std::array<dealii::Point<1>,2>				vertices;
	double 										jacobian;

	std::array<types::loc_t, dofs_per_cell> 	unit_cell_dof_index_map;

	Element(std::array<dealii::Point<1>,2> vertices);
	Element(const Element&);

	/**
	 * This is the main use of elements in this code.
	 * For speed, we assume that the quadrature points are inside the element.
	 * Checking this is the responsibility of the caller.
	 */
	void 				get_interpolation_weights(const dealii::Point<1> quadrature_point, 
													std::vector<double> & weights) const;
};

/*********************************/
/*         Dimension two         */
/*********************************/

/*   Vertices and geometry:
 *	   
 *     3          2
 *		o- - - - o
 *		|		 |
 *		|		 |
 *		|		 |
 *		o- - - - o
 *     0          1
 */


template<int degree>
struct Element<2,degree>
{
	static const types::loc_t 		dofs_per_cell = (degree+1)*(degree+1);
	static const types::loc_t 		vertices_per_cell = 4;

	std::array<dealii::Point<2>,4>	vertices;
	dealii::Tensor<2,2>				jacobian;

	std::array<types::loc_t, dofs_per_cell> 	unit_cell_dof_index_map;

	Element(std::array<dealii::Point<2>,4> vertices);
	Element(const Element&);

	/**
	 * This is the main use of elements in this code.
	 * For speed, we assume that the quadrature points are inside the element.
	 * Checking this is the responsibility of the caller.
	 */
	void 		get_interpolation_weights(const dealii::Point<2> quadrature_point, 
													std::vector<double> & weights) const;
};


/**
 * Declaration of explicit specializations 
 */

template <>
void	Element<1,1>::get_interpolation_weights(
						const dealii::Point<1> quadrature_point, 
						std::vector<double> & weights) const;
template<>
void	Element<1,2>::get_interpolation_weights(
						const dealii::Point<1> quadrature_point, 
						std::vector<double> & weights) const;
template<>
void	Element<1,3>::get_interpolation_weights(
						const dealii::Point<1> quadrature_point, 
						std::vector<double> & weights) const;

template<>
void	Element<2,1>::get_interpolation_weights(
						const dealii::Point<2> quadrature_point, 
						std::vector<double> & weights) const;
template<>
void	Element<2,2>::get_interpolation_weights(
						const dealii::Point<2> quadrature_point, 
						std::vector<double> & weights) const;
template<>
void	Element<2,3>::get_interpolation_weights(
						const dealii::Point<2> quadrature_point, 
						std::vector<double> & weights) const;


#endif
