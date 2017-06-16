/* 
 * File:   multilayer.h
 * Author: Paul Cazeaux
 *
 * Created on April 22, 2017, 12:28AM
 */


#ifndef MULTILAYER_H
#define MULTILAYER_H
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

#include "parameters/layerdata.h"
#include "materials/materials.h"


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
    // Generic job information
    std::string     job_name;
    std::string     output_file;
    int             observable_type;
    
    double          inter_search_radius;
    
    unsigned int    poly_degree;
    double          energy_rescale;
    double          energy_shift;

    double          B;
    double          E;
            
    
    // Layer data
    std::array<LayerData<dim>,n_layers >    layer_data;

    // Determines the cutoff radius in real and reciprocal space (equal for now)
    double  cutoff_radius;
    int     refinement_level;



    /* constuctors and destructor ============================================================ */
    Multilayer( std::string job_name = "UNKNOWN_JOB", 
                std::string output_file = "UNKNOWN_JOB_DATA.jld",
                int observable_type = 0,
                double inter_search_radius = 5,
                int poly_degree = 20,   
                double energy_rescale = 20, double energy_shift = 0,
                double B = 0,           double E = 0,
                double cutoff_radius = 0);

    Multilayer(int argc, char **argv);
    ~Multilayer() {}


    Multilayer<dim,1>   extract_monolayer(const unsigned char layer_index) const;

    double              intralayer_term(const int orbital_row, const int orbital_column, 
                                            const std::array<int, dim> grid_vector, 
                                            const unsigned char layer_index);
    double              interlayer_term(const int orbital_row, const int orbital_col, 
                                            const dealii::Tensor<1, dim> arrow_vector, 
                                            const unsigned char layer_index_row, const unsigned char layer_index_col);

    /* operator << ========================================================================== */
    friend std::ostream& operator<<( std::ostream& os, const Multilayer<dim, n_layers>& ml)
    {
            os << " T Input parameters for multilayer object:"                      << std::endl;
            os << " | observable_type = "           << ml.observable_type           << std::endl;
            os << " | poly_degree = "               << ml.poly_degree               << std::endl;
            os << " | cutoff_radius = "             << ml.cutoff_radius             << std::endl;
            os << " | refinement_level = "          << ml.refinement_level          << std::endl;
            os << " | inter_search_radius = "       << ml.inter_search_radius       << std::endl;
            
            os << " | energy_rescale = "            << ml.energy_rescale            << std::endl;
            os << " | energy_shift = "              << ml.energy_shift              << std::endl;
            os << " | B = "                         << ml.B                         << std::endl;
            os << " | E = "                         << ml.E                         << std::endl;
            
            os << " L job_name = "                  << ml.job_name                  << std::endl;

            return os;
    }
};


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
    cutoff_radius(cutoff_radius) {};



/* File Constructor */
template <int dim, int n_layers>
Multilayer<dim,n_layers>::Multilayer(int argc, char **argv) {
    /* Generate default values */
    job_name = "UNKNOWN_JOB"; 
    output_file = "UNKNOWN_JOB_DATA.jld";
    observable_type = 0;
    inter_search_radius = 12.5;
    poly_degree = 20;
    energy_rescale = 20;
    energy_shift = 0;
    B = 0;
    E = 0;
    cutoff_radius = 0;

    // ------------------------------
    // Generate input for simulation.
    // ------------------------------

    int current_layer = static_cast<unsigned int>(-1);

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
        while ( getline(in_file,line) )
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
                        layer_data[current_layer] = LayerData<dim>(mat, height, angle, dilation);
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
                    angle = std::stod(in_string);
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

template<int dim,int n_layers>
Multilayer<dim,1>
Multilayer<dim,n_layers>::extract_monolayer(const unsigned char layer_index) const
{
    Multilayer<dim,1> 
    monolayer = Multilayer<dim,1>(
                this->job_name,                         this->observable_type,
                this->inter_search_radius,
                this->poly_degree,                      
                this->energy_rescale,                   this->energy_shift,
                this->B,                                this->E,
                this->cutoff_radius);
    monolayer.layer_data[0] = this->layer_data[layer_index];
    return monolayer;
};


template<int dim, int n_layers>
double
Multilayer<dim,n_layers>::intralayer_term(const int orbital_row, const int orbital_col, 
                                            const std::array<int, dim> grid_vector, 
                                            const unsigned char layer_index)
{
    /* Check if diagonal element; if yes, add offset due to vertical electrical field */
    if (orbital_row == orbital_col && std::all_of(grid_vector.begin(), grid_vector.end(), [](int v) { return v==0; }))
        return Materials::intralayer_term(orbital_row, orbital_col, grid_vector,
                                                layer_data[layer_index].material)
                    + this->E * this->layer_data[layer_index].orbital_height .at(orbital_row);
    else
        return Materials::intralayer_term(orbital_row, orbital_col, grid_vector,
                                                layer_data[layer_index].material);
}


template<int dim, int n_layers>
double
Multilayer<dim,n_layers>::interlayer_term(const int orbital_row, const int orbital_col, 
                                            const dealii::Tensor<1, dim> arrow_vector, 
                                            const unsigned char layer_index_row, const unsigned char layer_index_col)
{
    std::array<double, dim+1> vector;
    for (int i=0; i<dim; ++i)
        vector[i] = arrow_vector[i];
    vector[dim] = this->layer_data[layer_index_col].height - this->layer_data[layer_index_row].height;

    return Materials::interlayer_term(orbital_row, orbital_col, vector,
                                                layer_data[layer_index_row].angle, 
                                                layer_data[layer_index_col].angle,
                                                layer_data[layer_index_row].material, 
                                                layer_data[layer_index_col].material);
}


#endif /* MULTILAYER_H */
