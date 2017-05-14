/* 
 * File:   main.cpp
 * Author: Paul Cazeaux
 *
 * Created on December 20, 2016, 11:43 AM
 */

#include <iostream>

#include <deal.II/base/mpi.h>

#include "tools/types.h"
#include "deal.II/lac/dynamic_sparsity_pattern.h"
#include "bilayer/dofhandler.h"

static const int dim = 1;
static const int degree = 3;
static const int n_layers = 2;
 
int main(int argc, char** argv) {

	/*********************************************************/
	/*					Initialize MPI 						 */
	/*********************************************************/

	dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
	MPI_Comm 					mpi(MPI_COMM_WORLD);

	if (dealii::Utilities::MPI::n_mpi_processes(mpi) == 1)
	{
		printf("Error: Only 1 MPI rank detected (need to run with n > 1).\n");
		return -1;
	}

	/*********************************************************/
	/*	 Read input file, and initialize params and vars. 	 */
	/*********************************************************/
	Multilayer<dim, n_layers> 				bilayer(argc,argv);
	Bilayer::DoFHandler<dim, degree> 		dof_handler(bilayer);
	dof_handler.initialize(mpi);

	dealii::DynamicSparsityPattern dsp (dof_handler.locally_owned_dofs());
	dof_handler.make_sparsity_pattern_rmultiply(dsp);

	// dealii::DynamicSparsityPattern dsp (dof_handler.locally_relevant_dofs());
	// dof_handler.make_sparsity_pattern_transpose(dsp);
	// dealii::SparsityTools::distribute_sparsity_pattern(dsp, 
	// 				dof_handler.n_locally_owned_dofs_per_processor(), 
	// 				mpi, dof_handler.locally_relevant_dofs());

	if (dof_handler.my_pid == 0)
		dsp.print_gnuplot(std::cout);

	return 0;
}
