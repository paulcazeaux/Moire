/* 
 * File:   multilayer.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 30, 2017, 6:00PM
 */

#include "parameters/multilayer.h"

/* Default Constructor */
template <int dim, int n_layers>
Multilayer<dim,n_layers>::Multilayer(   std::string job_name, 
                std::string output_file,
                int observable_type,
                double inter_search_radius,
                int poly_degree,    
                double energy_rescale,              double energy_shift,
                double B,                           double E,
                double cutoff_radius)
    :
    job_name(job_name),     output_file(output_file),
    observable_type(observable_type),
    inter_search_radius(inter_search_radius),
    poly_degree(poly_degree),                       
    energy_rescale(energy_rescale),     energy_shift(energy_shift),
    B(B),                               E(E),
    cutoff_radius(cutoff_radius) {}



/* File Constructor */
template <int dim, int n_layers>
Multilayer<dim,n_layers>::Multilayer(int argc, char **argv) {
    /* Generate default values */
    job_name = "UNKNOWN_JOB"; 
    output_file = "UNKNOWN_JOB_DATA.jld";
    observable_type = 0;
    inter_search_radius = 0.;
    poly_degree = 20;
    energy_rescale = 20;
    energy_shift = 0;
    B = 0;
    E = 0;
    cutoff_radius = 0;

    // ------------------------------
    // Generate input for simulation.
    // ------------------------------

    int current_layer = -1;

    // ---------------------------------------------------------
    // Next three Categories define a single layer's information
    // ---------------------------------------------------------
    Materials::Mat mat = Materials::Mat::Invalid;
    
    // Height (in angstroms) and twist angle (CCW, in radians)
    double height = 0;
    double angle = 0;
    double dilation = 1.;
    
    // -----------------------------------------------------------
    // Now we parse the command-line input file for these settings
    // -----------------------------------------------------------
    
    std::string line;
    std::ifstream in_file;
    in_file.open(argv[2]);
    if (in_file.is_open())
    {
        while ( std::getline(in_file,line) )
        {
            
            std::istringstream in_line(line);
            std::string in_string;
            
            while ( std::getline(in_line, in_string, ' ') )  {
                
                if (in_string == "JOB_NAME"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    job_name = in_string;
                    }

                if (in_string == "OUTPUT_FILE"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    output_file = in_string;
                    }
                
                if (in_string == "CUTOFF"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    cutoff_radius = std::stod(in_string);
                    }

                if (in_string == "REFINEMENT_LEVEL"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    refinement_level = std::stoi(in_string);
                    }
                    
                                    
                if (in_string == "NUM_LAYERS") {
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    int N = std::stoi(in_string);
                    if (n_layers > N)
                        throw std::runtime_error("Mismatched number of layers. The input file declares " + in_string 
                            + " layers while the templated Multilayer parameters expects " + std::to_string(n_layers) + ".\n");
                    else if (n_layers < N)
                        std::cout << "Warning ! Mismatched number of layers. The input file declares " + in_string 
                            + " layers while the templated Multilayer parameters expects " + std::to_string(n_layers) + ".\n" 
                                << "We will proceed regardless with only the first layer.\n" ;
                    }
                
                if (in_string == "START_LAYER") {
                    std::getline(in_line,in_string,' ');
                    current_layer = std::stoi(in_string) - 1;
                }
                
                if (in_string == "END_LAYER" && current_layer < n_layers) {
                    std::getline(in_line,in_string,' ');
                    if (current_layer == std::stoi(in_string) - 1) {
                        layer_data[current_layer] = std::make_unique<LayerData<dim>>(mat, height, angle, dilation);
                        if (inter_search_radius < Materials::inter_search_radius(mat))
                            inter_search_radius = Materials::inter_search_radius(mat);
                    }
                }
                
                if (in_string == "MATERIAL") {
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    mat = Materials::string_to_mat(in_string);
                }
                
                if (in_string == "HEIGHT"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    height = std::stod(in_string);
                }
                
                if (in_string == "ANGLE"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    angle = (2.*numbers::PI /360.) * std::stod(in_string);
                }

                if (in_string == "DILATION"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    dilation = std::stod(in_string);
                }

                
                if (in_string == "POLY_ORDER"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    poly_degree = std::stod(in_string);
                }
                
                if (in_string == "OBSERVABLE_TYPE"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    if (in_string == "DOS"){
                        observable_type = 0;
                    } 
                    else if (in_string == "COND"){
                        observable_type = 1;
                    }
                }

                if (in_string == "B_FIELD"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    B = std::stod(in_string);
                }

                if (in_string == "E_FIELD"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    E = std::stod(in_string);
                }   
                
                if (in_string == "ENERGY_RESCALE"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    energy_rescale = std::stod(in_string);
                }
                
                if (in_string == "ENERGY_SHIFT"){
                    std::getline(in_line,in_string,' ');
                    std::getline(in_line,in_string,' ');
                    energy_shift = std::stod(in_string);
                }
            }
        }
        in_file.close();
    }
}

/* copy constructor */
template <int dim, int n_layers>
Multilayer<dim,n_layers>::Multilayer(const Multilayer& ml)
    :
    job_name(ml.job_name),     output_file(ml.output_file),
    observable_type(ml.observable_type),
    inter_search_radius(ml.inter_search_radius),
    poly_degree(ml.poly_degree),                       
    energy_rescale(ml.energy_rescale),     energy_shift(ml.energy_shift),
    B(ml.B),                               E(ml.E),
    cutoff_radius(ml.cutoff_radius) 
{
    for (int n = 0; n<n_layers; ++n)
        layer_data[n] = std::make_unique<LayerData<dim>> (ml.layer(n));
}



template<int dim,int n_layers>
const LayerData<dim>&   
Multilayer<dim, n_layers>::layer(const int & idx) const 
{ return *layer_data[idx]; }

template<int dim,int n_layers>
void
Multilayer<dim, n_layers>::set_angles(std::array<double, n_layers> angles)
{
    for (int n = 0; n<n_layers; ++n)
    {
        LayerData<dim> new_layer (layer(n));
        new_layer.set_angle(angles[n]);
        layer_data[n] = std::make_unique<LayerData<dim>> (new_layer);
    }
}

template<int dim,int n_layers>
void
Multilayer<dim, n_layers>::set_dilation_factors(std::array<double, n_layers> factors)
{
    for (int n = 0; n<n_layers; ++n)
    {
        LayerData<dim> new_layer (layer(n));
        new_layer.set_dilation(factors[n]);
        layer_data[n] = std::make_unique<LayerData<dim>> (new_layer);
    }
}


template<int dim,int n_layers>
Multilayer<dim,1>
Multilayer<dim,n_layers>::extract_monolayer(const unsigned char layer_index) const
{
    Multilayer<dim,1> 
    monolayer = Multilayer<dim,1>(
                this->job_name,                         this->output_file,
                this->observable_type,
                this->inter_search_radius,
                this->poly_degree,                      
                this->energy_rescale,                   this->energy_shift,
                this->B,                                this->E,
                this->cutoff_radius);
    monolayer.layer_data[0] = std::make_unique<LayerData<dim>> (* this->layer_data[layer_index]);
    return monolayer;
}


template<int dim, int n_layers>
double
Multilayer<dim,n_layers>::intralayer_term(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const std::array<types::loc_t, dim> grid_vector, 
                                            const int layer_index) const
{
    /* Check if diagonal element; if yes, add offset due to vertical electrical field */
    if (orbital_row == orbital_col && std::all_of(grid_vector.begin(), grid_vector.end(), [](types::block_t v) { return v==0; }))
        return ( Materials::intralayer_term(orbital_row, orbital_col, grid_vector, layer(layer_index).material)
                     + this->energy_shift + this->E * layer(layer_index).orbital_height .at(orbital_row)
		     	) / this->energy_rescale;
    else
        return Materials::intralayer_term(orbital_row, orbital_col, grid_vector,
                                                layer(layer_index).material) / this->energy_rescale;
}


template<int dim, int n_layers>
double
Multilayer<dim,n_layers>::interlayer_term(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const dealii::Tensor<1, dim> arrow_vector, 
                                            const int layer_index_row, const int layer_index_col) const
{
    std::array<double, dim+1> vector;
    for (size_t i=0; i<dim; ++i)
        vector[i] = arrow_vector[i];
    vector[dim] = layer(layer_index_col).height - layer(layer_index_row).height;

    return Materials::interlayer_term(orbital_row, orbital_col, vector,
                                                layer(layer_index_row).angle, 
                                                layer(layer_index_col).angle,
                                                layer(layer_index_row).material, 
                                                layer(layer_index_col).material)
				/ this->energy_rescale;
}


template<int dim, int n_layers>
bool
Multilayer<dim,n_layers>::is_intralayer_term_nonzero(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const std::array<types::loc_t, dim> grid_vector, 
                                            const int layer_index) const
{
    return Materials::is_intralayer_term_nonzero(orbital_row, orbital_col, grid_vector,
                                                layer(layer_index).material) / this->energy_rescale;
}


template<int dim, int n_layers>
bool
Multilayer<dim,n_layers>::is_interlayer_term_nonzero(const types::loc_t orbital_row, const types::loc_t orbital_col, 
                                            const dealii::Tensor<1, dim> arrow_vector, 
                                            const int layer_index_row, const int layer_index_col) const
{
    std::array<double, dim+1> vector;
    for (size_t i=0; i<dim; ++i)
        vector[i] = arrow_vector[i];
    vector[dim] = layer(layer_index_col).height - layer(layer_index_row).height;

    return Materials::is_interlayer_term_nonzero(orbital_row, orbital_col, vector,
                                                layer(layer_index_row).angle, 
                                                layer(layer_index_col).angle,
                                                layer(layer_index_row).material, 
                                                layer(layer_index_col).material);
}


/**
 * Explicit instantiations
 */

template class Multilayer<1, 1>;
template class Multilayer<1, 2>;
template class Multilayer<2, 1>;
template class Multilayer<2, 2>;

