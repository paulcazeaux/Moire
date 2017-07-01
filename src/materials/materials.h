/* 
 * File:   materials.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#ifndef moire__materials_h
#define moire__materials_h

#include <array>
#include <string>
#include <map>

#include "materials/1d_model.h"
#include "materials/graphene.h"
#include "materials/strained_graphene.h"
#include "materials/TMDC.h"


/**
 * This module is intended as an interface between our data library about 2D materials
 * and the computational part of this program.
 *
 * In particular, it provides a list of materials, and for each one its geometrical 
 * characteristics (lattice basis, search cutoff radius for the hamiltonian elements)
 * as well as a tight-binding model (number of orbitals and hopping terms.)
 *
 * The intralayer term is computed as a function of orbital indices as well
 * as the vector between lattice sites in integer (lattice) coordinates.
 * The interlayer term is computed as a function of orbital indices as well as real-space 
 * vector between site positions, as well as some angular information if necessary.
 */

namespace Materials {
    /**
     * An enum type reflecting known materials in this library 
     */
    enum class Mat {Toy1D, Graphene, StrainedGraphene, MoS2, WS2, MoSe2, WSe2, Invalid};
    /** A utility function to translate the name of the material (as a string) into
     * a member of the enum type above.
     * This is e.g. for use when initializing from file.
     */
    Mat string_to_mat(std::string in_str);

    /*********************************************************************/
    /* Methods for returning each material's geometry and overall useful */
    /* spatial quantities related to tight-binding:                      */
    /*********************************************************************/
    /**
     * First, the lattice basis, templated on the dimension (1 or 2)
     */
    template<int dim>
    const std::array<std::array<double, dim>, dim>& lattice(const Mat mat);
    /** 
     * The number of orbitals in the tight-binding model implemented in this library
     * for this material.
     */
    const int&                                  n_orbitals(const Mat mat);
    /**
     * The cut-off radius for nonzero terms in the intra-layer Hamiltonian.
     */
    const double&                               intra_search_radius(const Mat mat);
    /**
     * The cut-off radius for nonzero terms in the inter-layer Hamiltonian.
     * This depends only on the broader category of materials (graphene, TMDCs...)
     * at this point.
     */
    const double&                               inter_search_radius(const Mat mat);
    /**
     * The vertical position of a given orbital in a given layered material, provided the
     * unit cell is centered around the z=0 plane.
     */
    const double&                               orbital_height(const Mat mat, const int idx);

    /**
     * Returns an intra-layer tight-binding hopping term for a one dimensional material,
     * for a given pair of orbitals and a lattice vector in grid coordinates computed 
     * for the material mat.
     */
    double
    intralayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<int, 1>& vector, 
                    const Mat mat);
    bool
    is_intralayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<int, 1>& vector, 
                    const Mat mat);
    /**
     * Returns an intra-layer tight-binding hopping term for a two dimensional material,
     * for a given pair of orbitals and a lattice vector in grid coordinates computed 
     * for the material mat.
     */
    double
    intralayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<int, 2>& vector, 
                    const Mat mat);
    bool
    is_intralayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<int, 2>& vector, 
                    const Mat mat);
    /**
     * Returns an inter-layer tight-binding hopping term between two one dimensional materials,
     * for a given pair of orbitals and a real-space vector in cartesian coordinates.
     *
     * The two angle variables are there for compatibility with the two-dimensional case
     * and does nothing in the case of 1 dimensional materials.
     */

    double
    interlayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<double, 2>& vector, 
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col);
    bool
    is_interlayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<double, 2>& vector, 
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col);
    /**
     * Returns an inter-layer tight-binding hopping term between a pair of two dimensional 
     * materials, for a given pair of orbitals and a real-space vector in cartesian coordinates.
     *
     * The two angle variables are the twist angles for the corresponding layer, with respect
     * to the un-rotated lattice as given in this library in the corresponding header file.
     */
    double
    interlayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<double, 3>& vector, 
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col);
    bool
    is_interlayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<double, 3>& vector, 
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col);

    /**
     * Explicit template specializations for the lattice method
     */
    template<>
    const std::array<std::array<double, 1>, 1>&
    lattice<1>(const Mat mat);

    template<>
    const std::array<std::array<double, 2>, 2>&
    lattice<2>(const Mat mat);

}   /* End namespace Materials */
#endif