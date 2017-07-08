/* 
 * File:   layer_data.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 30, 2017, 6:30 PM
 */

#include "parameters/layer_data.h"

template<int dim>
LayerData<dim>::LayerData(): material(Materials::Mat::Invalid), n_orbitals(0), height(0), intra_search_radius(0), angle(0), dilation(1.) {}

template<int dim>
LayerData<dim>::LayerData(      Materials::Mat material,
                                double height, double angle, double dilation):
                        material(material),
                        n_orbitals(Materials::n_orbitals(material)),
                        height(height),
                        intra_search_radius(dilation * Materials::intra_search_radius(material)),
                        angle(angle),
                        dilation(dilation)  
{
    std::array<std::array<double, dim>, dim> 
    lattice = Materials::lattice<dim>(material);
    for (size_t i=0; i<dim; ++i)
        for (size_t j=0; j<dim; ++j)
            lattice_basis[i][j] = lattice[j][i];
    lattice_basis       = Transformation<dim>::matrix(dilation, angle) * lattice_basis;

    for (types::loc_t i=0; i<n_orbitals; ++i)
        orbital_height.push_back(height + Materials::orbital_height(material, i));
}

template<int dim>
LayerData<dim>::LayerData(const LayerData<dim>& ld):
    material        (ld.material),
    n_orbitals      (ld.n_orbitals),
    height          (ld.height),
    lattice_basis   (ld.lattice_basis),
    intra_search_radius (ld.intra_search_radius),
    orbital_height  (ld.orbital_height),
    angle           (ld.angle),
    dilation        (ld.dilation)
    {}

template<int dim>
void
LayerData<dim>::set_angle(const double angle)
{
    std::array<std::array<double, dim>, dim> 
    lattice = Materials::lattice<dim>(this->material);
    for (size_t i=0; i<dim; ++i)
        for (size_t j=0; j<dim; ++j)
            this->lattice_basis[i][j] = lattice[j][i];

    this->angle              = angle;
    this->lattice_basis       = Transformation<dim>::matrix(this->dilation, angle) * this->lattice_basis;
}

template<int dim>
void
LayerData<dim>::set_dilation(const double dilation)
{
    std::array<std::array<double, dim>, dim> 
    lattice = Materials::lattice<dim>(material);
    for (size_t i=0; i<dim; ++i)
        for (size_t j=0; j<dim; ++j)
            this->lattice_basis[i][j] = lattice[j][i];

    this->dilation            = dilation;
    this->lattice_basis       = Transformation<dim>::matrix(dilation, this->angle) * this->lattice_basis;
    this->intra_search_radius = dilation * Materials::intra_search_radius(this->material);
}

/**
 * Explicit instantiations
 */

template struct LayerData<1>;
template struct LayerData<2>;
