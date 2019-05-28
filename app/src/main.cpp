/* 
 * File:   main.cpp
 * Author: Paul Cazeaux
 *
 * Created on December 20, 2016, 11:43 AM
 */


#include <Tpetra_Core.hpp>

#include "parameters/multilayer.h"
#include "bilayer/compute_dos.h"
#include "bilayer/compute_resolvent.h"

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
		Tpetra::ScopeGuard tpetraScope (&argc, &argv);
		Teuchos::RCP<const Teuchos::Comm<int> > 
    	comm = Tpetra::getDefaultComm ();

		/*********************************************************/
		/*	 Read input file, and initialize params and vars. 	 */
		/*********************************************************/
        Multilayer<dim, 2> bilayer(argc, argv);

        /*********************************************************/
        /*   Run the Chebyshev recurrence and output moments.    */
        /*********************************************************/
        Bilayer::ComputeResolvent<dim,degree,double,types::DefaultNode>  
        compute_resolvent(bilayer);
        compute_resolvent.run();

        /*********************************************************/
        /*                  Output to file                       */
        /*********************************************************/
        compute_resolvent.write_to_file();
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
