/*
 * Test file:   bilayer.print_sparsity_patterns.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 16, 2017, 9:00 AM
 */

#include "tests.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "parameters/multilayer.h"
#include "bilayer/base_algebra.h"
#include "tools/numbers.h"

static const int degree = 1;
static const int dim = 2;
typedef double Scalar;
typedef typename types::DefaultNode Node;

struct TestAlgebra : public Bilayer::BaseAlgebra<dim,degree,Scalar,Node>
{
    public:
        typedef Bilayer::BaseAlgebra<dim,degree,Scalar,Node>  LA;

        TestAlgebra(Multilayer<dim, 2> bilayer);
        std::array<LA::MultiVector, 2> I, A, B;
};

void print_sp(Teuchos::RCP< const Bilayer::BaseAlgebra<2,degree,double>::Matrix::crs_graph_type > sp, size_t offset_row, size_t offset_col, std::ofstream& out)
{
    Teuchos::Array<types::glob_t> line(sp->getNodeNumCols ());
    for (size_t row=0; row< sp->getGlobalNumRows (); ++row)
       {
       	 size_t n;
         sp->getGlobalRowCopy (row, line, n);
         for (size_t j = 0; j < n; ++j)
           // while matrix entries are usually
           // written (i,j), with i vertical and
           // j horizontal, gnuplot output is
           // x-y, that is we have to exchange
           // the order of output
           out << offset_col+line[j] << " "
               << -static_cast<signed int>(offset_row+row)
               << std::endl;
       }
}

TestAlgebra::TestAlgebra(Multilayer<dim, 2> bilayer) :
    LA(bilayer)
{
    LA::assemble_hamiltonian_action();
    LA::assemble_adjoint_interpolant();
    
    std::ofstream 
    output_file("H_sparsity_pattern." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    size_t 
    M = hamiltonian_action.at(0)->getGlobalNumRows(),
    N = hamiltonian_action.at(0)->getGlobalNumCols();
    
    print_sp(hamiltonian_action.at(0)->getCrsGraph(),0,0, output_file);
    print_sp(hamiltonian_action.at(1)->getCrsGraph(),M,N, output_file);
    output_file.close(); 
    
    size_t
    M1 = adjoint_interpolant.at(0).at(0)->getGlobalNumRows(),
    N1 = adjoint_interpolant.at(0).at(0)->getGlobalNumCols(),
    M2 = adjoint_interpolant.at(1).at(0)->getGlobalNumRows(),
    N2 = adjoint_interpolant.at(0).at(1)->getGlobalNumCols();
    std::cout << M << " " << N << std::endl;

    output_file.open("A_sparsity_pattern." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    print_sp(adjoint_interpolant.at(0).at(0)->getCrsGraph(),0,0, output_file);
    print_sp(adjoint_interpolant.at(1).at(0)->getCrsGraph(),M,N1, output_file);
    print_sp(adjoint_interpolant.at(0).at(1)->getCrsGraph(),M1,N, output_file);
    print_sp(adjoint_interpolant.at(1).at(1)->getCrsGraph(),M+M2,N+N2, output_file);
    output_file.close();

    output_file.open("lattices." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    for (int b = 0; b<2; ++b)
     for (types::loc_t j=0; j < dof_handler.n_locally_owned_points(b); ++j)
      output_file << lattice(b).get_vertex_position(dof_handler.locally_owned_point(b,j).lattice_index) << " " << b << std::endl;
    output_file.close();

}

 void print_sparsity_patterns(Materials::Mat mat)
 {
    Multilayer<dim, 2> bilayer (
            "sparsity_pattern", 
            "none.jld",
            ObservableType::DoS,
            Materials::inter_search_radius(mat),
            1,   
            1, 0,
            0, 0,
            10., 2);

    double height;
    switch (mat)
    {
        case Materials::Mat::Graphene:
            height = 3.4;
        case Materials::Mat::MoS2:
        case Materials::Mat::WS2:
            height = 6.145;
        case Materials::Mat::MoSe2:
        case Materials::Mat::WSe2:
            height = 6.48;
        default:
            height = 0.;
    }
    bilayer.layer_data[0] = std::make_unique<LayerData<dim>>(mat, 0.,  0.,   1.);
    bilayer.layer_data[1] = std::make_unique<LayerData<dim>>(mat, height, 5.71 * numbers::PI/180, 1.);

    TestAlgebra test_algebra (bilayer);
 }


 int main(int argc, char** argv) {
        /*********************************************************/
        /*                  Initialize MPI                       */
        /*********************************************************/
        Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
    try
    {
        print_sparsity_patterns(Materials::Mat::MoS2);
    }
  catch (std::exception &exc)
    {
      std::cout << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cout << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return -1;
    }
  catch (...)
    {
      std::cout << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cout << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return -1;
    };

  return 0;
}
