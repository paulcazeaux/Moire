/* 
 * File:   main.cpp
 * Author: Paul Cazeaux
 *
 * Created on December 20, 2016, 11:43 AM
 */



#include "deal.II/base/mpi.h"

#include "parameters/multilayer.h"
#include "bilayer/computedos.h"

#include <iostream>

static const int dim = 1;
static const int degree = 3;
static const int n_layers = 2;


int main(int argc, char** argv) {

	try
	{
		/*********************************************************/
		/*					Initialize MPI 						 */
		/*********************************************************/

		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
		PetscInitialize(&argc, &argv, NULL, NULL);
		// if (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1)
		// {
		// 	printf("Error: Only 1 MPI rank detected (need to run with n > 1).\n");
		// 	return -1;
		// }

		/*********************************************************/
		/*	 Read input file, and initialize params and vars. 	 */
		/*********************************************************/
		Multilayer<dim, n_layers> 				bilayer(argc,argv);
		Bilayer::ComputeDoS<dim, degree> 		compute_dos(bilayer);
		compute_dos.run();
	}
	catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
	catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
		return 0;
}
