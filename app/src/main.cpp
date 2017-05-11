/* 
 * File:   main.cpp
 * Author: Paul Cazeaux
 *
 * Created on December 20, 2016, 11:43 AM
 */

#include <iostream>

#include <deal.II/base/mpi.h>

#include "tools/types.h"
#include "bilayer/bilayer.h"
#include "monolayer/monolayer.h"
#include "fe/element.h"

static const int dim = 2;
static const int degree = 1;
static const int num_layers = 2;
 
int main(int argc, char** argv) {

	/*********************************************************/
	/*					Initialize MPI 						 */
	/*********************************************************/

	dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);


	if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
	{
		printf("Error: Only 1 MPI rank detected (need to run with n > 1).\n");
		return -1;
	}

	/*********************************************************/
	/*	 Read input file, and initialize params and vars. 	 */
	/*********************************************************/
	Multilayer<dim, num_layers> 				parameters(argc,argv);
	Bilayer::Groupoid<dim, degree> 				bilayer(parameters);
	bilayer.coarse_setup(MPI_COMM_WORLD);
	bilayer.local_setup(MPI_COMM_WORLD);
	return 0;
}
