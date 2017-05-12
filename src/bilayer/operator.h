/* 
* File:   operator.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/



#ifndef BILAYER_OPERATOR_H
#define BILAYER_OPERATOR_H

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <utility>
#include <metis.h>

#include "deal.II/base/mpi.h"
#include "deal.II/base/exceptions.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/index_set.h"
#include "deal.II/base/utilities.h"

#include "deal.II/lac/dynamic_sparsity_pattern.h"
#include "deal.II/lac/sparsity_pattern.h"
#include "deal.II/lac/sparsity_tools.h"

#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unitcell.h"
#include "tools/transformation.h"
#include "bilayer/pointdata.h"


/**
* This class encapsulates a discretized operator in the bilayer C* algebra.
*/



namespace Bilayer {

template <int dim, int degree>
class Operator : public Multilayer<dim, 2>
{

}

}/* Namespace Bilayer */

#endif /* BILAYER_OPERATOR_H */
