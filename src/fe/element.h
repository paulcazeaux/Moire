/* 
* File:   element.h
* Author: Paul Cazeaux
*
* Created on May 4, 2017, 5:00 PM
*/



#ifndef ELEMENT_H
#define ELEMENT_H

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
	static const unsigned int 					dofs_per_cell = degree+1;
	static const unsigned int 					vertices_per_cell = 2;

	std::array<dealii::Point<1>,2>				vertices;
	double 										jacobian;

	std::array<unsigned int, dofs_per_cell> 	unit_cell_dof_index_map;

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

template <int degree>
Element<1,degree>::Element(std::array<dealii::Point<1>,2> vertices)
	:
	vertices(vertices)
	{
		jacobian = 1./(vertices[1](0) - vertices[0](0));
	}

template <int degree>
Element<1,degree>::Element(const Element<1,degree>& orig)
	:
	vertices(orig.vertices),
	jacobian(orig.jacobian),
	unit_cell_dof_index_map(orig.unit_cell_dof_index_map) {}

/*		Order 1
 *
 *     0           1
 *		o - - - - o
 */

template <>
void	Element<1,1>::get_interpolation_weights(
						const dealii::Point<1> quadrature_point, 
						std::vector<double> & weights) const
{
	weights.resize(dofs_per_cell);

	double x = jacobian * (quadrature_point(0) - vertices[0](0));
	weights[0] = -(x-1.);
	weights[1] = x;
}


/*		Order 2
 *
 *     0            1
 *		o - - x - - o
 */

template<>
void	Element<1,2>::get_interpolation_weights(
						const dealii::Point<1> quadrature_point, 
						std::vector<double> & weights) const
{
	weights.resize(dofs_per_cell);

	double x = jacobian * (quadrature_point(0) - vertices[0](0));
	weights[0] = (x-1.)*(2.*x-1.);
	weights[1] = -4. * x * (x-1.);
	weights[2] = x * (2.*x-1.);
}

/*		Order 3
 *
 *     0             1
 *		o - x - x - o
 */

template<>
void	Element<1,3>::get_interpolation_weights(
						const dealii::Point<1> quadrature_point, 
						std::vector<double> & weights) const
{
	weights.resize(dofs_per_cell);

	double x = jacobian * (quadrature_point(0) - vertices[0](0));
	weights[0] = -0.5 * (3.*x-1.) * (3.*x-2.) * (x-1.);
	weights[1] =  4.5 * x * (3.*x-2.) * (x-1.);
	weights[2] = -4.5 * x * (3.*x-1.) * (x-1.);
	weights[3] =  0.5 * x * (3.*x-1.) * (3.*x-2.);
}


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
	static const unsigned int 		dofs_per_cell = (degree+1)*(degree+1);
	static const unsigned int 		vertices_per_cell = 4;

	std::array<dealii::Point<2>,4>	vertices;
	dealii::Tensor<2,2>				jacobian;

	std::array<unsigned int, dofs_per_cell> 	unit_cell_dof_index_map;

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

template<int degree>
Element<2,degree>::Element(std::array<dealii::Point<2>,4> vertices)
	:
	vertices(vertices)
	{
		jacobian[0][0] = vertices[1](0) - vertices[0](0);
		jacobian[1][0] = vertices[1](1) - vertices[0](1);
		jacobian[0][1] = vertices[3](0) - vertices[0](0);
		jacobian[1][1] = vertices[3](1) - vertices[0](1);
		jacobian = dealii::invert(jacobian);
	}


template <int degree>
Element<2,degree>::Element(const Element<2,degree>& orig)
	:
	vertices(orig.vertices),
	jacobian(orig.jacobian),
	unit_cell_dof_index_map(orig.unit_cell_dof_index_map) {}

/*		Order 1		
 *
 *     3          2
 *		o- - - - o
 *		|		 |
 *		|		 |
 *		|		 |
 *		o- - - - o
 *     0          1
 */


template<>
void	Element<2,1>::get_interpolation_weights(
						const dealii::Point<2> quadrature_point, 
						std::vector<double> & weights) const
{
	weights.resize(dofs_per_cell);

	dealii::Tensor<1,2> X = jacobian * (quadrature_point - vertices[0]);
	double x = X[0], y = X[1];
	double px[2] = {1.-x , x};
	double py[2] = {1.-y , y};
	weights[0] = px[0] * py[0];
	weights[1] = px[1] * py[0];
	weights[2] = px[0] * py[1];
	weights[3] = px[1] * py[1];
}

/*		Order 2	
 *	   
 *     3         2
 *		o - x - o
 *		|		|
 *		x   x   x
 *		|		|
 *		o - x - o
 *     0         1
 */


template<>
void	Element<2,2>::get_interpolation_weights(
						const dealii::Point<2> quadrature_point, 
						std::vector<double> & weights) const
{
	weights.resize(dofs_per_cell);

	dealii::Tensor<1,2> X = jacobian * (quadrature_point - vertices[0]);
	double x = X[0], y = X[1];
	double px[3] = {          (2.*x-1.) * (x-1.), 
		                x   *    -4.    * (x-1.), 
		                x   * (2.*x-1.)            };
	double py[3] = {          (2.*y-1.) * (y-1.), 
		                y   *    -4.    * (y-1.), 
		                y   * (2.*y-1.)             };

	weights[0] = px[0] * py[0];
	weights[1] = px[1] * py[0];
	weights[2] = px[2] * py[0];
	weights[3] = px[0] * py[1];
	weights[4] = px[1] * py[1];
	weights[5] = px[2] * py[1];
	weights[6] = px[0] * py[2];
	weights[7] = px[1] * py[2];
	weights[8] = px[2] * py[2];
}

/*		Order 3
 *	   
 *     3             2
 *		o - x - x - o
 *		|		    |
 *		x   x   x   x
 *		|		    |
 *		x   x   x   x
 *		|		    |
 *		o - x - x - o
 *     0             1
 */

template<>
void	Element<2,3>::get_interpolation_weights(
						const dealii::Point<2> quadrature_point, 
						std::vector<double> & weights) const
{
	weights.resize(dofs_per_cell);

	dealii::Tensor<1,2> X = jacobian * (quadrature_point - vertices[0]);
	double x = X[0], y = X[1];
	double px[4] = {     -0.5 * (3.*x-1.) * (3.*x-2.) * (x-1.) , 
						   x  *    4.5    * (3.*x-2.) * (x-1.) ,
						   x  * (3.*x-1.) *   -4.5    * (x-1.) ,
						   x  * (3.*x-1.) * (3.*x-2.) *   0.5  };
	double py[4] = {     -0.5 * (3.*y-1.) * (3.*y-2.) * (y-1.) , 
						   y  *    4.5    * (3.*y-2.) * (y-1.) ,
						   y  * (3.*y-1.) *   -4.5    * (y-1.) ,
						   y  * (3.*y-1.) * (3.*y-2.) *   0.5  };

	weights[0] = px[0] * py[0];
	weights[1] = px[1] * py[0];
	weights[2] = px[2] * py[0];
	weights[3] = px[3] * py[0];
	weights[4] = px[0] * py[1];
	weights[5] = px[1] * py[1];
	weights[6] = px[2] * py[1];
	weights[7] = px[3] * py[1];
	weights[8] = px[0] * py[2];
	weights[9] = px[1] * py[2];
	weights[10] = px[2] * py[2];
	weights[11] = px[3] * py[2];
	weights[12] = px[0] * py[3];
	weights[13] = px[1] * py[3];
	weights[14] = px[2] * py[3];
	weights[15] = px[3] * py[3];
}


#endif /* ELEMENT_H */
