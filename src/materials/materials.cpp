/* 
 * File:   materials.cpp
 * Author:  Paul Cazeaux
 * 
 * Created on June 13, 2017, 10:00 AM
 */

#include "materials/materials.h"
using namespace Materials;

/* Utility function for translating strings to material type */
Mat 
Materials::string_to_mat(std::string in_str)
{
    const std::map<std::string, Mat> mat_map {
        {"Toy1D", Mat::Toy1D},
        {"Graphene", Mat::Graphene},
        {"StrainedGraphene", Mat::StrainedGraphene},
        {"MoS2", Mat::MoS2},
        {"WS2", Mat::WS2},
        {"MoSe2", Mat::MoSe2},
        {"WSe2", Mat::WSe2} };
    try 
    {
        return mat_map.at(in_str);
    }
    catch (...)
    {
        throw std::runtime_error("Failed to find material! Aborting... \n");
    }

    return Mat::Invalid;
}

/* Methods for returning each material's geometry and overall tight-binding space */

template<>
const std::array<std::array<double, 1>, 1>&
Materials::lattice<1>(const Mat mat)
{
    assert(mat == Mat::Toy1D);
    return Toy1D::lattice;
}

template<>
const std::array<std::array<double, 2>, 2>&
Materials::lattice<2>(const Mat mat)
{
    switch (mat)
    {
        case Mat::Graphene:
            return Graphene::lattice;
        case Mat::StrainedGraphene:
            return StrainedGraphene::lattice;
        case Mat::MoS2:
            return MoS2::lattice;
        case Mat::WS2:
            return WS2::lattice;
        case Mat::MoSe2:
            return MoSe2::lattice;
        case Mat::WSe2:
            return WSe2::lattice;
        default:
            throw std::runtime_error("Failed to find material. \n");
    }
}

const int&
Materials::n_orbitals(Mat mat)
{
    switch (mat)
    {
        case Mat::Toy1D: 
            return Toy1D::n_orbitals;

        case Mat::Graphene:
        case Mat::StrainedGraphene:
            return Graphene::n_orbitals;

        case Mat::MoS2:
            return MoS2::n_orbitals;
        case Mat::WS2:
            return WS2::n_orbitals;
        case Mat::MoSe2:
            return MoSe2::n_orbitals;
        case Mat::WSe2:
            return WSe2::n_orbitals;
        default:
            throw std::runtime_error("Failed to find material. \n");
    }
}

const double&
Materials::intra_search_radius(Mat mat)
{
    switch (mat)
    {
        case Mat::Toy1D: 
            return Toy1D::intra_search_radius;

        case Mat::Graphene:
            return Graphene::intra_search_radius;
        case Mat::StrainedGraphene:
            return StrainedGraphene::intra_search_radius;
            
        case Mat::MoS2:
            return MoS2::intra_search_radius;
        case Mat::WS2:
            return WS2::intra_search_radius;
        case Mat::MoSe2:
            return MoSe2::intra_search_radius;
        case Mat::WSe2:
            return WSe2::intra_search_radius;
        default:
            throw std::runtime_error("Failed to find material. \n");
    }
}

const double&
Materials::inter_search_radius(Mat mat)
{
    switch (mat)
    {
        case Mat::Toy1D: 
            return Toy1D::inter_search_radius;

        case Mat::Graphene:
            return Graphene::inter_search_radius;
        case Mat::StrainedGraphene:
            return StrainedGraphene::inter_search_radius;
            
        case Mat::MoS2:
            return MoS2::inter_search_radius;
        case Mat::WS2:
            return WS2::inter_search_radius;
        case Mat::MoSe2:
            return MoSe2::inter_search_radius;
        case Mat::WSe2:
            return WSe2::inter_search_radius;
        default:
            throw std::runtime_error("Failed to find material. \n");
    }
}
const double&
Materials::orbital_height(const Mat mat, const int idx)
{
    switch (mat)
    {
        case Mat::Toy1D: 
            return Toy1D::atom_pos;

        case Mat::Graphene:
            return Graphene::atom_pos.at (Graphene::atom(Graphene::orbital(idx)))[3];
        case Mat::StrainedGraphene:
            return StrainedGraphene::atom_pos.at (Graphene::atom(Graphene::orbital(idx)))[3];
            
        case Mat::MoS2:
            return MoS2::atom_pos.at (TMDC::atom(TMDC::orbital(idx)))[3];
        case Mat::WS2:
            return WS2::atom_pos.at (TMDC::atom(TMDC::orbital(idx)))[3];
        case Mat::MoSe2:
            return MoSe2::atom_pos.at (TMDC::atom(TMDC::orbital(idx)))[3];
        case Mat::WSe2:
            return WSe2::atom_pos.at (TMDC::atom(TMDC::orbital(idx)))[3];
        default:
            throw std::runtime_error("Failed to find material. \n");
    }
}

/* Methods for returning intralayer and interlayer terms */

double
Materials::intralayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<int, 1>& vector, 
                    const Mat mat)
{
    assert(mat == Mat::Toy1D);
    return Coupling::Intralayer::one_d_model(vector[0]);
}

bool
Materials::is_intralayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<int, 1>& vector, 
                    const Mat mat)
{
    assert(mat == Mat::Toy1D);
    return IsNonZero::Intralayer::one_d_model(vector[0]);
}

double
Materials::interlayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<double, 2>& vector,
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col)
{
    assert(mat_row == Mat::Toy1D && mat_col == Mat::Toy1D);
    return Coupling::Interlayer::one_d_model(vector[0]);
}

bool
Materials::is_interlayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<double, 2>& vector,
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col)
{
    assert(mat_row == Mat::Toy1D && mat_col == Mat::Toy1D);
    return IsNonZero::Interlayer::one_d_model(vector[0]);
}

double
Materials::intralayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<int, 2>& vector, 
                    const Mat mat)
{
    switch (mat)
    {
        case Mat::Graphene: // graphene
            return Coupling::Intralayer::graphene( 
                        Graphene::orbital(orbital_row), Graphene::orbital(orbital_col),
                            vector);
        case Mat::StrainedGraphene: // strained graphene
            return Coupling::Intralayer::strained_graphene(
                        Graphene::orbital(orbital_row), Graphene::orbital(orbital_col),
                            vector);

        case Mat::MoS2:
            return Coupling::Intralayer::MoS2(
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col),
                            vector);
        case Mat::WSe2:
            return Coupling::Intralayer::WSe2(
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col),
                            vector);
        case Mat::MoSe2:
            return Coupling::Intralayer::MoSe2(
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col),
                            vector);
        case Mat::WS2:
            return Coupling::Intralayer::WS2(
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col),
                            vector);

        default:
            throw std::runtime_error("Failed to find material. \n");    
    }
    return 0.;
}

bool
Materials::is_intralayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<int, 2>& vector, 
                    const Mat mat)
{
    switch (mat)
    {
        case Mat::Graphene: // graphene
            return IsNonZero::Intralayer::graphene( 
                        Graphene::orbital(orbital_row), Graphene::orbital(orbital_col),
                            vector);
        case Mat::StrainedGraphene: // strained graphene
            return IsNonZero::Intralayer::strained_graphene(
                        Graphene::orbital(orbital_row), Graphene::orbital(orbital_col),
                            vector);

        case Mat::MoS2:
        case Mat::WSe2:
        case Mat::MoSe2:
        case Mat::WS2:
            return IsNonZero::Intralayer::TMDC(
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col),
                            vector);

        default:
            throw std::runtime_error("Failed to find material. \n");    
    }
    return false;
}

double
Materials::interlayer_term(const int orbital_row, const int orbital_col, 
                    const std::array<double, 3>& vector,
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col)
{
    // graphene case
    if ( (mat_row == Mat::Graphene || mat_row == Mat::StrainedGraphene) && (mat_col == Mat::Graphene || mat_col == Mat::StrainedGraphene) )
        return Coupling::Interlayer::C_to_C( 
                        Graphene::orbital(orbital_row), Graphene::orbital(orbital_col), vector,
                             angle_row, angle_col);

    // MoS2 and WS2 cases
    if ( (mat_row == Mat::MoS2 || mat_row == Mat::WS2) && (mat_col == Mat::MoS2 || mat_col == Mat::WS2) )
        return Coupling::Interlayer::S_to_S( 
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col), vector,
                             angle_row, angle_col);

    // MoSe2 and WSe2 cases
    if ( (mat_row == Mat::MoSe2 || mat_row == Mat::WSe2) && (mat_col == Mat::MoSe2 || mat_col == Mat::WSe2) )
        return Coupling::Interlayer::Se_to_Se( 
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col), vector,
                             angle_row, angle_col);

    throw std::runtime_error("Failed to find interlayer term for the selected materials. \n");
    return 0.;
}

bool
Materials::is_interlayer_term_nonzero(const int orbital_row, const int orbital_col, 
                    const std::array<double, 3>& vector,
                    const double angle_row, const double angle_col,
                    const Mat mat_row, const Mat mat_col)
{
    // graphene case
    if ( (mat_row == Mat::Graphene || mat_row == Mat::StrainedGraphene) && (mat_col == Mat::Graphene || mat_col == Mat::StrainedGraphene) )
        return IsNonZero::Interlayer::C_to_C( 
                        Graphene::orbital(orbital_row), Graphene::orbital(orbital_col), vector,
                             angle_row, angle_col);

    // MoS2 and WS2 cases
    if ( (mat_row == Mat::MoS2 || mat_row == Mat::WS2) && (mat_col == Mat::MoS2 || mat_col == Mat::WS2) )
        return IsNonZero::Interlayer::S_to_S( 
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col), vector,
                             angle_row, angle_col);
    if ( (mat_row == Mat::MoSe2 || mat_row == Mat::WSe2) && (mat_col == Mat::MoSe2 || mat_col == Mat::WSe2) )
        return IsNonZero::Interlayer::Se_to_Se( 
                        TMDC::orbital(orbital_row), TMDC::orbital(orbital_col), vector,
                             angle_row, angle_col);

    throw std::runtime_error("Failed to find interlayer term for the selected materials. \n");
    return false;
}
