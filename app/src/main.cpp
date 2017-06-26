/* 
 * File:   main.cpp
 * Author: Paul Cazeaux
 *
 * Created on December 20, 2016, 11:43 AM
 */



#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Version.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include "parameters/multilayer.h"
#include "bilayer/compute_dos.h"

#include <iostream>
#include <fstream>

static const int dim = 1;
static const int degree = 3;


int main(int argc, char** argv) {

	try
	{
		/*********************************************************/
		/*					Initialize MPI 						 */
		/*********************************************************/
		Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
		Teuchos::RCP<const Teuchos::Comm<int> > 
    	comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();

		/*********************************************************/
		/*	 Read input file, and initialize params and vars. 	 */
		/*********************************************************/
		Multilayer<dim, 2> 					bilayer(argc,argv);
		int M = bilayer.poly_degree + 1;
		int N = 1;


		int my_pid = comm->getRank ();

		if (my_pid == 0)
		{
			std::ofstream output_file(bilayer.output_file, std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
			output_file.write((char*) &M, sizeof(int));
			output_file.write((char*) &N, sizeof(int));
			output_file.close();
		}

		Bilayer::ComputeDoS<dim, degree>
		compute_dos(bilayer);
		compute_dos.run();
		std::vector<Bilayer::ComputeDoS<dim, degree>::scalar_type> moments = compute_dos.output_results();
		if (my_pid == 0)
		{
			std::ofstream output_file(bilayer.output_file, std::ofstream::binary | std::ofstream::out | std::ofstream::app);
			for (unsigned int i = 0; i<moments.size(); ++i)
			{
				double m = moments[i].real();
				output_file.write((char*) & m, sizeof(double));
			}
			output_file.close();
		}
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
