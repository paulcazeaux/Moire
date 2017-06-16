/* 
 * File:   layerdata.h
 * Author: Stephen Carr
 * Modified by Paul Cazeaux
 *
 * Created on January 29, 2016, 4:28 PM
 */


#ifndef LAYERDATA_H
#define LAYERDATA_H
#include <vector>
#include <string>
#include "deal.II/base/tensor.h"
#include "deal.II/base/point.h"

#include "materials/materials.h"
#include "tools/transformation.h"

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

        Materials::Mat             material;
        dealii::Tensor<2,dim>     lattice_basis;
        int                       n_orbitals;
        double                    intra_search_radius;

        double              height;
        std::vector<double> orbital_height;
        double              angle;
        double              dilation;
};

template<int dim>
LayerData<dim>::LayerData(): material(Materials::Mat::Invalid), n_orbitals(0), intra_search_radius(0), height(0), angle(0), dilation(1.) {}

template<int dim>
LayerData<dim>::LayerData(      Materials::Mat material,
                                double height, double angle, double dilation):
                        material(material),
                        n_orbitals(Materials::n_orbitals(material)),
                        intra_search_radius(dilation * Materials::intra_search_radius(material)),
                        height(height),
                        angle(angle),
                        dilation(dilation)  
{
    std::array<std::array<double, dim>, dim> 
    lattice = Materials::lattice<dim>(material);
    for (int i=0; i<dim; ++i)
        for (int j=0; j<dim; ++j)
            lattice_basis[i][j] = lattice[i][j];
    lattice_basis       = Transformation<dim>::matrix(dilation, angle) * lattice_basis;

    for (int i=0; i<n_orbitals; ++i)
        orbital_height.push_back(height + Materials::orbital_height(material, i));
}

template<int dim>
LayerData<dim>::LayerData(const LayerData<dim>& layerdata) {
    material           = layerdata.material;
    lattice_basis      = layerdata.lattice_basis;
    n_orbitals         = layerdata.n_orbitals;
    intra_search_radius= layerdata.intra_search_radius;
    height             = layerdata.height;
    orbital_height     = layerdata.orbital_height;
    angle              = layerdata.angle;
    dilation           = layerdata.dilation;
}

#endif /* LAYERDATA_H */