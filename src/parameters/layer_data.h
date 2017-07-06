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
#include "tools/numbers.h"

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

        friend std::ostream& operator<<( std::ostream& os, const LayerData<dim>& l)
        {
            os << " | Material type: "                  << Materials::mat_to_string(l.material)           << std::endl;
            os << " | Number of orbitals: "             << l.n_orbitals                                   << std::endl;
            os << " | Dilation parameter: "             << l.dilation                                     << std::endl;
            os << " | Vertical position: "              << l.height                                       << std::endl;
            os << " | Twist angle (counterclockwise): " << l.angle * 180/ numbers::PI  << "°"             << std::endl;
            return os;
        };
};


#endif
