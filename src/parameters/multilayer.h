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
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>

#include "deal.II/base/tensor.h"
#include "deal.II/base/point.h"
#include "parameters/layerdata.h"


/**
 * This class holds the data read from input file at the start of the program. 
 * In particular, it assumes a particular input file format (see the constructor)
 * with the data for num_layers individual layers of dimension dim.
 * Note that each of these layers exist as a hyperplane in a dim+1 space with
 * a specific height in the last coordinate, and the position of orbitals in 
 * the unit cell may include an offset in this last coordinate (e.g. for TMDC layers) 
 */

template<int dim, int num_layers>
struct Multilayer {
public:
	// Generic job information
	std::string 	job_name;
	int 			observable_type;
	
	double 			intra_searchsize_radius;
	double 			inter_searchsize_radius;
	
	int 			poly_degree;
	double 			energy_rescale;
	double 			energy_shift;

	bool 			magOn;
	bool 			elecOn;
	double 			B;
	double 			E;
			
	
	// Layer data
	std::array<LayerData<dim>,num_layers > 	layer_data;

	// Determines the cutoff radius in real and reciprocal space (equal for now)
	double 	cutoff_radius;
	int 	refinement_level;



	/* constuctors and destructor ============================================================ */
    Multilayer(	std::string job_name = "UNKNOWN_JOB", 
    			int observable_type = 0,
				double intra_searchsize_radius = 5, double inter_searchsize_radius = 5,
				int poly_degree = 20,	
				double energy_rescale = 20,	double energy_shift = 0, 
				bool magOn = false, 	bool elecOn = false,
				double B = 0, 			double E = 0,
				double cutoff_radius = 0);

    Multilayer(int argc, char **argv);
    ~Multilayer() {}


    Multilayer<dim,1>	extract_monolayer(const int layer_index) const;

	/* operator << ========================================================================== */
	friend std::ostream& operator<<( std::ostream& os, const Multilayer<dim, num_layers>& ml)
	{
			os << " T Input parameters for multilayer object:" 						<< std::endl;
			os << " | observable_type = " 			<< ml.observable_type 			<< std::endl;
			os << " | poly_degree = " 				<< ml.poly_degree 				<< std::endl;
			os << " | cutoff_radius = "				<< ml.cutoff_radius 			<< std::endl;
			os << " | refinement_level = "			<< ml.refinement_level 			<< std::endl;
			os << " | intra_searchsize_radius = "	<< ml.intra_searchsize_radius	<< std::endl;
			os << " | inter_searchsize_radius = "	<< ml.inter_searchsize_radius	<< std::endl;
			os << (ml.magOn  ? " | magOn  = true \n" : " | magOn  = false \n")		<< std::endl;
			os << (ml.elecOn ? " | elecOn = true \n" : " | elecOn = false \n")		<< std::endl;
			
			os << " | energy_rescale = " 			<< ml.energy_rescale 			<< std::endl;
			os << " | energy_shift = "				<< ml.energy_shift 				<< std::endl;
			os << " | B = "							<< ml.B 						<< std::endl;
			os << " | E = "							<< ml.E 						<< std::endl;
			
			os << " L job_name = " 					<< ml.job_name 					<< std::endl;

			return os;
	}
};


/* Default Constructor */
template <int dim, int num_layers>
Multilayer<dim,num_layers>::Multilayer(	std::string job_name, 
    			int observable_type,
				double intra_searchsize_radius, 	double inter_searchsize_radius,
				int poly_degree,	
				double energy_rescale,				double energy_shift, 
				bool magOn, 						bool elecOn,
				double B, 							double E,
				double cutoff_radius)
	:
	job_name(job_name),		observable_type(observable_type),
	intra_searchsize_radius(intra_searchsize_radius),	
	inter_searchsize_radius(inter_searchsize_radius),
	poly_degree(poly_degree),						
	energy_rescale(energy_rescale),		energy_shift(energy_shift),
	magOn(magOn),						elecOn(elecOn),
	B(B),								E(E),
	cutoff_radius(cutoff_radius) {};



/* File Constructor */
template <int dim, int num_layers>
Multilayer<dim,num_layers>::Multilayer(int argc, char **argv) {
	/* Generate default values */
	job_name = "UNKNOWN_JOB"; 
	observable_type = 0;
	intra_searchsize_radius = 12.5;
	inter_searchsize_radius = 12.5;
	poly_degree = 20;
	energy_rescale = 20;
	energy_shift = 0;
	magOn = false;
	elecOn = false;
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
	int mat = -1;
	// Unit cell information, gets put into an sdata object
	double a = 0;
	dealii::Tensor<2,dim> lattice_basis;

	// number of orbitals per unit cell
	int num_orbitals = 0;
	std::vector<int> types;
	std::vector<dealii::Point<dim+1> > pos;
	
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
			
			while ( getline(in_line, in_string, ' ') )	{
				
				if (in_string == "JOB_NAME"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					job_name = in_string;
					}
				
				if (in_string == "CUTOFF"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					cutoff_radius = atof(in_string.c_str());
					}

				if (in_string == "REFINEMENT_LEVEL"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					refinement_level = atoi(in_string.c_str());
					}
					
									
				if (in_string == "NUM_LAYERS") {
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					int N = atoi(in_string.c_str());
					if (num_layers > N)
						throw std::runtime_error("Mismatched number of layers. The input file declares " + in_string 
							+ " layers while the templated Multilayer parameters expects " + std::to_string(num_layers) + ".\n");
					else if (num_layers < N)
						std::cout << "Warning ! Mismatched number of layers. The input file declares " + in_string 
							+ " layers while the templated Multilayer parameters expects " + std::to_string(num_layers) + ".\n" 
								<< "We will proceed regardless with only the first layer.\n" ;
					}
				
				if (in_string == "START_LAYER") {
					getline(in_line,in_string,' ');
					current_layer = atoi(in_string.c_str()) - 1;
				}
				
				if (in_string == "END_LAYER" && current_layer < num_layers) {
					getline(in_line,in_string,' ');
					if (current_layer == atoi(in_string.c_str()) - 1) {
						layer_data[current_layer] = LayerData<dim>(lattice_basis, pos, types, height, angle, dilation);
					}
				}
				
				if (in_string == "MATERIAL") {
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					mat = atoi(in_string.c_str());
				}
				
				if (in_string == "ALPHA") {
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					a = atof(in_string.c_str());
				}
				
				if (in_string == "UNITCELL1" && dim > 0){
					getline(in_line,in_string,' ');
					for (int i = 0; i < dim; ++i) {
						getline(in_line,in_string,' ');
						lattice_basis[i][0] = a*atof(in_string.c_str());
					}
				}
				
				if (in_string == "UNITCELL2" && dim > 1){
					getline(in_line,in_string,' ');
					for (int i = 0; i < dim; ++i) {
						getline(in_line,in_string,' ');
						lattice_basis[i][1] = a*atof(in_string.c_str());
					}
				}
				
				if (in_string == "NUM_ORBITALS"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					num_orbitals = atoi(in_string.c_str());
					types.resize(num_orbitals);
					pos.resize(num_orbitals);
				}
				
				if (in_string == "TYPES"){
					getline(in_line,in_string,' ');
					for (int i = 0; i < num_orbitals; ++i) {
						getline(in_line,in_string,' ');
						types[i] = atoi(in_string.c_str());
					}
				}
					
				if (in_string == "POS"){
					getline(in_line,in_string,' ');
					for (int i = 0; i < num_orbitals; ++i) {
						for (int j = 0; j < dim+1; ++j) {
							getline(in_line,in_string,' ');
							pos[i](j) = a*atof(in_string.c_str());
						}
					}
				}
				
				if (in_string == "HEIGHT"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					height = atof(in_string.c_str());
				}
				
				if (in_string == "ANGLE"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					angle = atof(in_string.c_str());
				}

				if (in_string == "DILATION"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					dilation = atof(in_string.c_str());
				}
				
				if (in_string == "POLY_ORDER"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					poly_degree = atoi(in_string.c_str());
				}
				
				if (in_string == "OBSERVABLE_TYPE"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					if (in_string == "DOS"){
						observable_type = 0;
					} else if (in_string == "COND"){
						observable_type = 1;
					}
				}
				
				if (in_string == "USE_B_FIELD"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					magOn = atoi(in_string.c_str());
				}	

				if (in_string == "B_FIELD"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					B = atof(in_string.c_str());
				}
				
				if (in_string == "USE_E_FIELD"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					elecOn = atoi(in_string.c_str());
				}	

				if (in_string == "E_FIELD"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					E = atof(in_string.c_str());
				}	

				if (in_string == "INTRA_SEARCHSIZE"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					intra_searchsize_radius = atof(in_string.c_str());
				}
				
				if (in_string == "INTER_SEARCHSIZE"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					inter_searchsize_radius = atof(in_string.c_str());
				}
				
				if (in_string == "ENERGY_RESCALE"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					energy_rescale = atof(in_string.c_str());
				}
				
				if (in_string == "ENERGY_SHIFT"){
					getline(in_line,in_string,' ');
					getline(in_line,in_string,' ');
					energy_shift = atof(in_string.c_str());
				}
			}
		}
		in_file.close();
	}

}

template<int dim,int num_layers>
Multilayer<dim,1>
Multilayer<dim,num_layers>::extract_monolayer(const int layer_index) const
{
	Multilayer<dim,1> monolayer = Multilayer<dim,1>(
				this->job_name,							this->observable_type,
				this->intra_searchsize_radius,			this->inter_searchsize_radius,
				this->poly_degree,						
				this->energy_rescale,					this->energy_shift,
				this->magOn,							this->elecOn,
				this->B,								this->E,
				this->cutoff_radius);
	monolayer.layer_data[0] = this->layer_data[layer_index];
	return monolayer;
};

#endif /* MULTILAYER_H */
