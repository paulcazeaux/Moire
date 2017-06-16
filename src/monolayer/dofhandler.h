/* 
* File:   monolayer.h
* Author: Paul Cazeaux
*
* Created on May 8, 2017, 14:15 PM
*/



#ifndef moire__monolayer_dofhandler_h
#define moire__monolayer_dofhandler_h

#include <memory>
#include "deal.II/base/mpi.h"
#include "deal.II/base/tensor.h"

#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unitcell.h"
#include "tools/transformation.h"


/**
* This class encapsulates the discretized underlying monolayer structure:
* assembly of Brillouin zone mesh and direct lattice points for a monolayer group.
* It is in particular responsible for building a sparsity pattern 
* and handling partitioning of the degrees of freedom by metis.
* This is not meant be work in an MPI distributed fashion (as it should be used 
* e.g. as a local preconditioner for the bilayer structures).
*/

namespace Monolayer {

template <int dim, int degree>
class DoFHandler : public Multilayer<dim, 1>
{

static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!\n");

public:
	DoFHandler(const Multilayer<dim, 1>& parameters);
	~DoFHandler() {};



	const Lattice<dim>&							lattice() const { return *lattice_; };
	const UnitCell<dim,degree>&					brillouin_zone() const { return *brillouin_zone_; };


private:
	std::unique_ptr<Lattice<dim>> 				lattice_;
	std::unique_ptr<UnitCell<dim,degree>> 		brillouin_zone_;

	void 										distribute_dofs();
	void										make_sparsity_pattern();
};

template<int dim, int degree>
DoFHandler<dim,degree>::DoFHandler(const Multilayer<dim, 1>& parameters)
	:
	Multilayer<dim, 1>(parameters)
{
	const  LayerData<dim>& layer = parameters.layer_data[0];

	dealii::Tensor<2,dim> rotated_basis = Transformation<dim>::matrix(layer.dilation, layer.angle)
											* layer.lattice_basis;
		
	lattice_   = std::make_unique<Lattice<dim>>(rotated_basis, parameters.cutoff_radius);
	brillouin_zone_ = std::make_unique<UnitCell<dim,degree>>(
			2*numbers::PI*dealii::transpose(dealii::invert(rotated_basis)), parameters.refinement_level);
}

template<int dim, int degree>
void
DoFHandler<dim,degree>::distribute_dofs()
{
	// TODO
};


template<int dim, int degree>
void
DoFHandler<dim,degree>::make_sparsity_pattern()
{
	// TODO
};
}

#endif
