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

struct TestAlgebra : public Bilayer::BaseAlgebra<2,degree,double>
{
    public:
        typedef Bilayer::BaseAlgebra<2,degree,double>  LA;

        TestAlgebra(Multilayer<2, 2> bilayer);
        std::array<MultiVector, 2> I, A, B;
};

TestAlgebra::TestAlgebra(Multilayer<2, 2> bilayer) :
    LA(bilayer)
{
    LA::base_setup();
    LA::assemble_hamiltonian_action();
    LA::assemble_adjoint_interpolant();
    
    std::ofstream 
    output_file("H_sparsity_pattern.0." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    hamiltonian_action.at(0).describe(std::cout, Teuchos::VERB_EXTREME);
    output_file.close(); 
    output_file.open("H_sparsity_pattern.1." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    hamiltonian_action.at(1).describe(std::cout, Teuchos::VERB_EXTREME);

    output_file.close(); 
    output_file.open("A_sparsity_pattern.0.0" + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    adjoint_interpolant.at(0).at(0).describe(std::cout, Teuchos::VERB_EXTREME);
    output_file.close();
    output_file.open("A_sparsity_pattern.0.1." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    adjoint_interpolant.at(0).at(1).describe(std::cout, Teuchos::VERB_EXTREME);
    output_file.close(); 
    output_file.open("A_sparsity_pattern.1.0." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    adjoint_interpolant.at(1).at(0).describe(std::cout, Teuchos::VERB_EXTREME);
    output_file.close();
    output_file.open("A_sparsity_pattern.1.1." + std::to_string(this->dof_handler.my_pid), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    adjoint_interpolant.at(1).at(1).describe(std::cout, Teuchos::VERB_EXTREME);
    output_file.close();
}

 void print_sparsity_patterns(Materials::Mat mat)
 {
    Multilayer<2, 2> bilayer (
            "sparsity_pattern", 
            "none.jld",
            ObservableType::DoS,
            Materials::inter_search_radius(mat),
            1,   
            1, 0,
            0, 0,
            6., 2);

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
    bilayer.layer_data[0] = std::make_unique<LayerData<2>>(mat, 0.,  0.,   1.);
    bilayer.layer_data[1] = std::make_unique<LayerData<2>>(mat, height, 5.71 * numbers::PI/180, 1.);

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