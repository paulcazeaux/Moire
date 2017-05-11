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
class LayerData {
	private:
		/* class members ======================================================================== */
		dealii::Tensor<2,dim> lattice_;
		std::vector<dealii::Point<dim+1> > orbital_positions_;
		std::vector<int> orbital_types_;
		int material_;

		double height_;
		double angle_;
        double dilation_;
        
    public:
		/* constuctors and destructor ============================================================ */
		LayerData();
        LayerData(dealii::Tensor<2,dim> lattice, 
        			std::vector<dealii::Point<dim+1> > orbital_pos, 
                    std::vector<int> orbital_types, 
                    double height, double angle, double dilation);
        LayerData(const LayerData&);
        ~LayerData() {}

		/* getters */
        dealii::Tensor<2,dim> 					get_lattice()			const { return lattice_; }
		std::vector<dealii::Point<dim+1> > 		get_orbital_positions()	const { return orbital_positions_; }
        std::vector<int>						get_orbital_types()		const { return orbital_types_; }
        unsigned int                            get_num_orbitals()      const { return orbital_types_.size(); }
        int 									get_material()			const { return material_; }

		double 									get_height()			const { return height_; }
		double 									get_angle()				const { return angle_; }
        double                                  get_dilation()             const { return dilation_; }
};

template<int dim>
LayerData<dim>::LayerData(): material_(-1), height_(0), angle_(0) {}

template<int dim>
LayerData<dim>::LayerData(   dealii::Tensor<2,dim> lattice, 
                                std::vector<dealii::Point<dim+1> > orbital_pos, 
                                std::vector<int> orbital_types, 
                                double height, double angle, double dilation):
                        lattice_(lattice), 
                        orbital_positions_(orbital_pos), 
                        orbital_types_(orbital_types), 
                        height_(height), 
                        angle_(angle),
                        dilation_(dilation) {}

template<int dim>
LayerData<dim>::LayerData(const LayerData<dim>& layerdata) {
    lattice_            = layerdata.lattice_;
    orbital_positions_  = layerdata.orbital_positions_;
    orbital_types_      = layerdata.orbital_types_;
    material_           = layerdata.material_;
    height_             = layerdata.height_;
    angle_              = layerdata.angle_;
    dilation_           = layerdata.dilation_;
}

#endif /* LAYERDATA_H */