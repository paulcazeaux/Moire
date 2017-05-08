/* 
* File:   groupoid.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/



#ifndef BILAYER_H
#define BILAYER_H

#include <memory>
#include <deal.II/base/mpi.h>
#include "deal.II/base/tensor.h"
#include "deal.II/physics/transformations.h"

#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unitcell.h"


/**
* This class encapsulates the underlying bilayer discrete structure:
* assembly of distributed meshes and lattice points for a bilayer groupoid.
* It is in particular responsible for building a sparsity pattern 
* and handling partitioning of the degrees of freedom by metis.
*/

namespace Bilayer {

template <int dim, int degree>
class Groupoid : public Multilayer<dim, 2>
{


static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!\n");

public:
	Groupoid(const Multilayer<dim, 2>& parameters);
	~Groupoid() {};



	const Lattice<dim>&			lattice(const int & idx) const { return *lattices_[idx]; };
	const UnitCell<dim,degree>&	unit_cell(const int & idx) const { return *unit_cells_[idx]; };


private:
	std::array<std::unique_ptr<Lattice<dim>>, 2> 				lattices_;
	std::array<std::unique_ptr<UnitCell<dim,degree>>, 2> 		unit_cells_;

	// std::array<std::unique_ptr<Triangulation<dim,degree>>, num_layers>	local_triangulations_;

	void						make_coarse_sparsity_pattern();
	void 						partition();
	void 						distribute_dofs();
	void						make_sparsity_pattern();

};

template<int dim, int degree>
Groupoid<dim,degree>::Groupoid(const Multilayer<dim, 2>& parameters)
	:
	Multilayer<dim, 2>(parameters)
{
	auto layers = parameters.layer_data;
	for (unsigned int i = 0; i<2; ++i)
	{
		dealii::Tensor<2,dim> rotated_basis;
		rotated_basis = Transformation<dim>::matrix(layers[i].get_dilation(), layers[i].get_angle()) * layers[i].get_lattice();
			
		lattices_[i]   = std::make_unique<Lattice<dim>>(rotated_basis, parameters.cutoff_radius);
		unit_cells_[i] = std::make_unique<UnitCell<dim,degree>>(rotated_basis, parameters.refinement_level);

	}
}

template<int dim, int degree>
void
Groupoid<dim,degree>::make_coarse_sparsity_pattern()
{

};

template<int dim, int degree>
void
Groupoid<dim,degree>::partition()
{

};


template<int dim, int degree>
void
Groupoid<dim,degree>::distribute_dofs()
{

};


template<int dim, int degree>
void
Groupoid<dim,degree>::make_sparsity_pattern()
{

};
}

#endif /* BILAYER_H */
