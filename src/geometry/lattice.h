/* 
* File:   lattice.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/


#ifndef moire__geometry_lattice_h
#define moire__geometry_lattice_h

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
    /* Announce to the world the cut-off radius, number of vertices, the index of the origin and array basis */
    const double                            radius;
    types::loc_t                            n_vertices;
    types::loc_t                            index_origin;
    const dealii::Tensor<2,dim>             basis;
    const dealii::Tensor<2,dim>             inverse_basis;

    /* Creator and destructor */
    Lattice(const dealii::Tensor<2,dim> basis, const double radius);
    ~Lattice() {};

    /* List all indices in a disk-like neighborhood of any radius around any point */
    std::vector<types::loc_t>               list_neighborhood_indices(const dealii::Point<dim>& X, const double radius) const;

    /* Utilities for locating the global index of a vertex from its grid index,
    *   or vice versa the grid index from the global index */

    types::loc_t                            get_vertex_global_index(const std::array<types::loc_t, dim>& indices) const;
    std::array<types::loc_t, dim>           get_vertex_grid_indices(const types::loc_t& index) const;
    dealii::Point<dim>                      get_vertex_position(const types::loc_t& index) const;

    /* Utilities for rounding up a given point to a lattice point modulo the unit cell  */
    std::array<types::loc_t, dim>           round_to_grid_indices(const dealii::Point<dim>& X) const;
    types::loc_t                            round_to_global_index(const dealii::Point<dim>& X) const;
    
    /* Utility to find the global index of the lattice point at an offset from a known lattice point */
    types::loc_t                            offset_global_index(const types::loc_t& index, const std::array<types::loc_t, dim>& offset) const;

private:

        /* Array of vertex positions */
    std::vector<dealii::Point<dim> >                vertices_;  

        /* Maps from global index to grid index */
    GridToIndexMap<dim>                             grid_to_index_map_;
    std::vector<std::array<types::loc_t, dim>>      index_to_grid_map_;

        /* The origin is special, we want to remember its index */


        /* A useful quantity for determining necessary search sizes when looking for neighborhoods */
    const double                                    unit_cell_inscribed_radius_;
    static double                                   compute_unit_cell_inscribed_radius(const dealii::Tensor<2,dim>& basis);
};

#endif
