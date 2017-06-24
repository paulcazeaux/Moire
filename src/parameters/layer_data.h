/* 
 * File:   layer_data.h
 * Author: Stephen Carr
 * Modified by Paul Cazeaux
 *
 * Created on January 29, 2016, 4:28 PM
 */


#ifndef moire__parameters_layer_data_h
#define moire__parameters_layer_data_h
#include <vector>
#include <string>
#include "deal.II/base/tensor.h"
#include "deal.II/base/point.h"

#include "materials/materials.h"
#include "tools/transformation.h"
#include "tools/types.h"

/** 
 * A simple container for the information associated with a single layer of dimension dim:
 * Material type,
 * Rotated lattice basis, 
 * Height: this layer exists as a hyperplane in a dim+1 space,
 *                  with a specific height in the last coordinate,
 * Rotation angle.                                                                                
 */

template<int dim>
struct LayerData {

    public:
        /* constuctors and destructor ============================================================ */
        LayerData();
        LayerData(Materials::Mat material,
                    double height, double angle, double dilation);
        LayerData(const LayerData&);
        ~LayerData() {}

        Materials::Mat              material;
        types::loc_t                n_orbitals;
        double                      height;

        dealii::Tensor<2,dim>       lattice_basis;
        double                      intra_search_radius;
        std::vector<double>         orbital_height;

        /* dilation and angle should be handled carefuly because their value modify implicitely */
        /* the lattice basis and intralayer search radius, and should not be changed directly.  */

        double              angle;
        double              dilation;

        void                set_angle(const double new_angle);
        void                set_dilation(const double new_dilation);
};

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
            lattice_basis[i][j] = lattice[i][j];
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
            this->lattice_basis[i][j] = lattice[i][j];

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
            this->lattice_basis[i][j] = lattice[i][j];

    this->dilation            = dilation;
    this->lattice_basis       = Transformation<dim>::matrix(dilation, this->angle) * this->lattice_basis;
    this->intra_search_radius = dilation * Materials::intra_search_radius(this->material);
}

#endif
