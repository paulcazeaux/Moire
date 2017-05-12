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

/** 
 * A simple container for the information associated with a single layer of dimension dim:
 * Unrotated lattice basis, 
 * Height: this layer exists as a hyperplane in a dim+1 space,
 *					with a specific height in the last coordinate,
 * Orbital positions in the unit cell: these include a (possibly zero) offset 
 *										in the last coordinate (e.g. for a TMDC layer),
 * Orbital types (currently unused),
 * Rotation angle.  																			  
 */

template<int dim>
struct LayerData {

    public:
		/* constuctors and destructor ============================================================ */
		LayerData();
        LayerData(unsigned char material,
                    dealii::Tensor<2,dim> lattice, 
        			std::vector<dealii::Point<dim+1> > orbital_pos, 
                    std::vector<int> orbital_types, 
                    double height, double angle, double dilation);
        LayerData(const LayerData&);
        ~LayerData() {}

        unsigned char                           material;
        dealii::Tensor<2,dim>                   lattice_basis;
        unsigned int                            n_orbitals;
        std::vector<dealii::Point<dim+1> >      orbital_positions;
        std::vector<int>                        orbital_types;

        double height;
        double angle;
        double dilation;
};

template<int dim>
LayerData<dim>::LayerData(): material(-1), n_orbitals(0), height(0), angle(0), dilation(1.) {}

template<int dim>
LayerData<dim>::LayerData(      unsigned char material,
                                dealii::Tensor<2,dim> lattice, 
                                std::vector<dealii::Point<dim+1> > orbital_pos, 
                                std::vector<int> orbital_types, 
                                double height, double angle, double dilation):
                        material(material),
                        lattice_basis(lattice), 
                        n_orbitals(orbital_pos.size()),
                        orbital_positions(orbital_pos), 
                        orbital_types(orbital_types), 
                        height(height), 
                        angle(angle),
                        dilation(dilation) 
                        {
                            assert(orbital_types.size() == n_orbitals);
                        }

template<int dim>
LayerData<dim>::LayerData(const LayerData<dim>& layerdata) {
    material           = layerdata.material;
    lattice_basis      = layerdata.lattice_basis;
    n_orbitals         = layerdata.n_orbitals;
    orbital_positions  = layerdata.orbital_positions;
    orbital_types      = layerdata.orbital_types;
    height             = layerdata.height;
    angle              = layerdata.angle;
    dilation           = layerdata.dilation;
}

#endif /* LAYERDATA_H */