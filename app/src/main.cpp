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

static const int dim = 2;
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
        Multilayer<dim, 2> bilayer(argc, argv);
        if (comm->getRank () == 0)
            std::cout << bilayer;

        /*********************************************************/
        /*   Run the Chebyshev recurrence and output moments.    */
        /*********************************************************/
        Bilayer::ComputeDoS<dim, degree, double>  
        compute_dos(bilayer);
        compute_dos.run();
        std::vector<std::array<std::vector<double>,2>> 
        moments = compute_dos.output_LDoS();

        /*********************************************************/
        /*                  Output to file                       */
        /*********************************************************/
		if (comm->getRank () == 0)
		{
			std::ofstream output_file(bilayer.output_file, std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
            int M = moments.size();
            int N = moments.at(0).at(0).size() + moments.at(0).at(1).size();

			output_file.write((char*) &M, sizeof(int));
			output_file.write((char*) &N, sizeof(int));
			for (const auto & moment : moments)
			{
				for (const double m : moment.at(0))
                    output_file.write((char*) & m, sizeof(double));
                for (const double m : moment.at(1))
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
