/* 
* File:   monolayer.h
* Author: Paul Cazeaux
*
* Created on May 8, 2017, 14:15 PM
*/



#ifndef MONOLAYER_H
#define MONOLAYER_H

#include <memory>
#include <deal.II/base/mpi.h>
#include "deal.II/base/tensor.h"
#include "deal.II/physics/transformations.h"

#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unitcell.h"


/**
* This class encapsulates the underlying monolayer discrete structure:
* assembly of Brillouin zone mesh and direct lattice points for a monolayer group.
* It is in particular responsible for holding the structure of degrees of freedom
* and building a sparsity pattern.
* This is not meant be work in an MPI distributed fashion (as it should be used 
* e.g. as a local preconditioner for the bilayer structures).
*/

namespace Monolayer {

template <int dim, int degree>
class Group : public Multilayer<dim, 1>
{

static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!\n");

public:
	Group(const Multilayer<dim, 1>& parameters);
	~Group() {};



	const Lattice<dim>&							lattice() const { return *lattice_; };
	const UnitCell<dim,degree>&					brillouin_zone() const { return *brillouin_zone_; };


private:
	std::unique_ptr<Lattice<dim>> 				lattice_;
	std::unique_ptr<UnitCell<dim,degree>> 		brillouin_zone_;

	void 										distribute_dofs();
	void										make_sparsity_pattern();
};

template<int dim, int degree>
Group<dim,degree>::Group(const Multilayer<dim, 1>& parameters)
	:
	Multilayer<dim, 1>(parameters)
{
	const  LayerData<dim>& layer = parameters.layer_data[0];

	dealii::Tensor<2,dim> rotated_basis = Transformation<dim>::matrix(layer.get_dilation(), layer.get_angle()) 
											* layer.get_lattice();
		
	lattice_   = std::make_unique<Lattice<dim>>(rotated_basis, parameters.cutoff_radius);
	brillouin_zone_ = std::make_unique<UnitCell<dim,degree>>(
			2*dealii::numbers::PI*dealii::transpose(dealii::invert(rotated_basis)), parameters.refinement_level);
}

template<int dim, int degree>
void
Group<dim,degree>::distribute_dofs()
{

};


template<int dim, int degree>
void
Group<dim,degree>::make_sparsity_pattern()
{

};
}

#endif /* MONOLAYER_H */