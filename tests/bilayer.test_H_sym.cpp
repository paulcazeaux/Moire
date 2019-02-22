/* 
 * Test file:   bilayer.test_H_sym.cpp
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
        MultiVector I, A, B;
};

TestAlgebra::TestAlgebra(Multilayer<2, 2> bilayer) :
    LA(bilayer)
{
    I  = LA::create_multivector();
    A  = Tpetra::createCopy(I);
    B  = Tpetra::createCopy(I);

    I.randomize();

    LA::assemble_hamiltonian_action();
    hamiltonian_rproduct(I, A, Teuchos::NO_TRANS);
    hamiltonian_rproduct(I, B, Teuchos::CONJ_TRANS);
}

 void do_test(Materials::Mat mat)
 {
    Multilayer<2, 2> bilayer (
            "test_hamiltonian", 
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

    test_algebra.A .update( -1., test_algebra.B , 1.);
    Teuchos::Array<double> norms (test_algebra.A .getNumVectors());
    test_algebra.A .normInf(norms);
    AssertThrow( std::accumulate(norms.begin(), norms.end(), 0.) < 1e-13,
                        dealii::ExcInternalError() );

    std::cout << "Symmetry test OK" << std::endl;
 }


 int main(int argc, char** argv) {
        /*********************************************************/
        /*                  Initialize MPI                       */
        /*********************************************************/
        Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
    try
    {
        do_test(Materials::Mat::Graphene);
        do_test(Materials::Mat::MoS2);
        do_test(Materials::Mat::WS2);
        do_test(Materials::Mat::MoSe2);
        do_test(Materials::Mat::WSe2);
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