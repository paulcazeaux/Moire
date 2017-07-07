/* 
 * File:   multilayer.h
 * Author: Paul Cazeaux
 *
 * Created on April 22, 2017, 12:28AM
 */


#ifndef moire__parameters_multilayer_h
#define moire__parameters_multilayer_h
 
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

#include "deal.II/base/tensor.h"
#include "deal.II/base/point.h"

#include "parameters/layer_data.h"
#include "materials/materials.h"

/** 
 * An enum class listing the available observable types (DoS, conductivity)
 */
enum class ObservableType {DoS, Invalid};
/** 
 * A utility function to translate the name of the observable type (as a string) 
 * into a member of the enum type above.
 * This is e.g. for use when initializing from file.
 */
ObservableType string_to_obs(std::string in_str);
/** 
 * A utility function to translate a member of the enum type above into a string
 * containing the human-readable name of the observable type.
 */
std::string obs_to_string(ObservableType obs);

/**
 * This class holds the data read from input file at the start of the program. 
 * In particular, it assumes a particular input file format (see the constructor)
 * with the data for n_layers individual layers of dimension dim.
 * Note that each of these layers exist as a hyperplane in a dim+1 space with
 * a specific height in the last coordinate, and the position of orbitals in 
 * the unit cell may include an offset in this last coordinate (e.g. for TMDC layers) 
 */

template<int dim, int n_layers>
struct Multilayer {
public:
    virtual
    const LayerData<dim>&   layer(const int & idx) const;
    void                    set_angles(std::array<double, n_layers> angles);
    void                    set_dilation_factors(std::array<double, n_layers> factors);

    Multilayer<dim,1>       extract_monolayer(const unsigned char layer_index) const;

    double                  intralayer_term(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const std::array<types::loc_t, dim> grid_vector, 
                                            const int layer_index) const;
    double                  interlayer_term(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const dealii::Tensor<1, dim> arrow_vector, 
                                            const int layer_index_row, const int layer_index_col) const;

    bool                    is_intralayer_term_nonzero(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const std::array<types::loc_t, dim> grid_vector, 
                                            const int layer_index) const;
    bool                    is_interlayer_term_nonzero(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const dealii::Tensor<1, dim> arrow_vector, 
                                            const int layer_index_row, const int layer_index_col) const;


    // Generic job information
    std::string     job_name;
    std::string     output_file;
    ObservableType observable_type;
    
    double          inter_search_radius;
    
    int             poly_degree;
    double          energy_rescale;
    double          energy_shift;

    double          B;
    double          E;

    // Determines the cutoff radius in real space and refinement level
    double          cutoff_radius;
    int             refinement_level;

    // Layer data
    std::array<std::unique_ptr<const LayerData<dim>>, n_layers >    layer_data;


    /* constuctors and destructor ============================================================ */
    Multilayer( std::string job_name = "UNKNOWN_JOB", 
                std::string output_file = "UNKNOWN_JOB_DATA.jld",
                ObservableType observable_type = ObservableType::Invalid,
                double inter_search_radius = 0,
                int poly_degree = 0,   
                double energy_rescale = 1, double energy_shift = 0,
                double B = 0,              double E = 0,
                double cutoff_radius = 0,  int refinement_level = 0);

    Multilayer(int argc, char **argv);
    Multilayer(const Multilayer&);

        /* operator << ========================================================================== */
    friend std::ostream& operator<<( std::ostream& os, const Multilayer<dim, n_layers>& ml)
    {
            os                                                                              << std::endl;
            os << " T Job name: "                   << ml.job_name                          << std::endl;
            os << " T Output file: "                << ml.output_file                       << std::endl;
            os << " |"                                                                      << std::endl;
            os << " | Input parameters for multilayer object:"                              << std::endl;
            os << " | ------------"                                                         << std::endl;
            os << " | Observable type: "            << obs_to_string(ml.observable_type)    << std::endl;
            os << " | Refinement level: "           << ml.refinement_level                  << std::endl;
            os << " | Interlayer search radius: "   << ml.inter_search_radius               << std::endl;
            
            os << " | B = "                         << ml.B                                 << std::endl;
            os << " | E = "                         << ml.E                                 << std::endl;
            
            os << " | Energy rescaling: "           << ml.energy_rescale                    << std::endl;
            os << " | Energy shift: "               << ml.energy_shift                      << std::endl;
            os << " | Polynomial degree: "          << ml.poly_degree                       << std::endl;
            os << " | Cutoff radius: "              << ml.cutoff_radius                     << std::endl;
            os << " L ------------"                                                         << std::endl;

            for (int i=0; i<n_layers; ++i)
            {
            os << " T Layer " << i+1 << " data:"                                            << std::endl; 
            os << * (ml.layer_data .at(i));
            os << " L ------------"                                                         << std::endl;
            os                                                                              << std::endl;
            }

            return os;
    }
};


#endif
