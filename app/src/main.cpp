/* 
 * File:   main.cpp
 * Author: Paul Cazeaux
 *
 * Created on December 20, 2016, 11:43 AM
 */

#include <iostream>
#include <memory>


#include <deal.II/base/mpi.h>
#include "deal.II/physics/transformations.h"

#include "geometry/bilayer.h"
#include "geometry/monolayer.h"
#include "fe/element.h"

static const int dim = 2;
static const int degree = 2;
static const int num_layers = 2;
 
int main(int argc, char** argv) {

	/*********************************************************/
	/*					Initialize MPI 						 */
	/*********************************************************/

	dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    int root = 0, size, myrank;
	
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	if (size == 1){
		printf("Error: Only 1 MPI rank detected (need to run with n > 1).\n");
		return -1;
	}

	/*********************************************************/
	/*	 Read input file, and initialize params and vars. 	 */
	/*********************************************************/
	Multilayer<dim, num_layers> 				parameters(argc,argv);
	Monolayer::Group<dim,degree> 				monolayer(parameters.extract_monolayer(0));
	Bilayer::Groupoid<dim, degree> 				bilayer(parameters);

	if (myrank == 0) {
		auto cell = monolayer.brillouin_zone();

		std::cout << "Number of elements: " << cell.number_of_elements << std::endl;

		std::vector<Element<dim, degree>> subcells = cell.build_subcell_list();
		for (const auto & v: subcells[2].vertices)
			std::cout << v << " ; ";
		std::cout << std::endl;
		for (const auto & i: subcells[2].unit_cell_dof_index_map)
			std::cout << i << " \t";
		std::cout << std::endl;
		for (const auto & i: subcells[2].unit_cell_dof_index_map)
			std::cout << cell.get_grid_point_position(i) << " ;";
		std::cout << std::endl;

		std::vector<dealii::Point<dim>> quad_points;
		for (const unsigned int i: subcells[2].unit_cell_dof_index_map)
			quad_points.push_back(cell.get_grid_point_position(i));

		dealii::FullMatrix<double> mat;
		subcells[2].get_interpolation_matrix(quad_points, mat);
		mat.print_formatted(std::cout, 1, false);
		
		std::cout << std::endl  << parameters << std::endl;
	}

	return 0;
}
