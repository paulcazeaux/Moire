/* 
 * File:   materials.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#ifndef MATERIALS_H
#define MATERIALS_H

#include <array>
#include <string>
#include <map>

#include "1d_model.h"
#include "graphene.h"
#include "strained_graphene.h"
#include "TMDC.h"


/* This module is intended as an interface between our data library about 2D materials
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
    /* An enum type reflecting known materials in this library */
    enum class Mat {Toy1D, Graphene, StrainedGraphene, MoS2, WS2, MoSe2, WSe2, Invalid};
    Mat string_to_mat(std::string in_str);

    /* Methods for returning each material's geometry and overall tight-binding space */
    template<int dim>
    const std::array<std::array<double, dim>, dim>& lattice(const Mat mat);
    const int&                                  n_orbitals(const Mat mat);
    const double&                               intra_search_radius(const Mat mat);
    const double&                               inter_search_radius(const Mat mat);
    const double&                               orbital_height(const Mat mat, const int idx);

    /* Methods for returning intralayer and interlayer terms */
    double
    intralayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<int, 1>& vector, 
                    const Mat mat);

    double
    intralayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<int, 2>& vector, 
                    const Mat mat);

    double
    interlayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<double, 2>& vector, 
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col);
    double
    interlayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<double, 3>& vector, 
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col);
}   /* End namespace Materials */

/* Explicit specializations for the lattice method */
template<>
const std::array<std::array<double, 1>, 1>&
Materials::lattice<1>(const Materials::Mat mat);

template<>
const std::array<std::array<double, 2>, 2>&
Materials::lattice<2>(const Materials::Mat mat);

#endif