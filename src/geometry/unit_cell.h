/* 
 * File:   unit_cell.h
 * Author: Paul Cazeaux
 *
 * Created on May 4, 2017, 12:58AM
 */


#ifndef moire__geometry_unit_cell_h
#define moire__geometry_unit_cell_h



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
 *              |       |       |
 *              o   o   o   o   x       (1d)
 *
 *                      or
 *
 *              x - x - x - x - x                   where
 *              |       |       |
 *              o   o   o   o   x                       o  => interior node (owned by the cell)
 *              |       |       |                       x  => boundary node
 *              o - o - o - o - x       (2d)
 *              |       |       |                       |  => indicates degree 2 finite element
 *              o   o   o   o   x                               used for interpolation
 *              |       |       |
 *              o - o - o - o - x
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
    const types::loc_t                      refinement_level;
    types::loc_t                            n_closure_nodes;
    types::loc_t                            n_nodes;
    types::loc_t                            n_boundary_nodes;

    /* Needed for the FFT */
    std::array<types::loc_t,dim>                     n_nodes_per_dim;

    /* A list of finite elements paving the unit cell, with each one storing for each support point
     * the corresponding global index within the cell 
     */
    types::loc_t                            n_elements;
    std::vector<Element<dim,degree>>        subcell_list;

    const dealii::Tensor<2,dim>             basis;
    const dealii::Tensor<2,dim>             inverse_basis;
    const double                            bounding_radius;
    const double                            area;

    /* Creator and destructor */
    UnitCell(const dealii::Tensor<2,dim> basis, const types::loc_t refinement_level);
    ~UnitCell() {};

    types::loc_t                            find_element(const dealii::Tensor<1,dim>& X) const;

    /**
     * Utilities for locating the cell-wide index of a degree of freedom from its grid index,
     * or vice versa the grid index from the global index 
     */
    types::loc_t                            get_node_global_index(const std::array<types::loc_t, dim>& indices) const;
    std::array<types::loc_t, dim>           get_node_grid_indices(const types::loc_t& index) const;
    dealii::Point<dim>                      get_node_position(const types::loc_t& index) const;


    bool                                    is_node_interior(const types::loc_t& index) const;
    std::tuple<types::loc_t, std::array<types::loc_t, dim>>
                                            map_boundary_point_interior(const types::loc_t& index) const;

private:

    /* Array of grid point positions */
    std::vector<dealii::Point<dim> >        nodes_;

    /* Maps from global index to grid index */
    GridToIndexMap<dim>                        grid_to_index_map_;
    std::vector<std::array<types::loc_t, dim>> index_to_grid_map_;

    /* Map boundary grid point indexes to interior points in another cell and the corresponding lattice grid offset */
    std::vector<std::tuple<types::loc_t, std::array<types::loc_t, dim>>>     boundary_to_interior_map_;

    /* Cache number of elements along one direction */
    types::loc_t                            line_element_count_;

    /* Static method for computing the bounding radius of the cell */
    static double   compute_bounding_radius(const dealii::Tensor<2,dim>& basis);
};

#endif
