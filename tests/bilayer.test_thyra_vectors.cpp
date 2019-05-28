/* 
 * Test file:   bilayer.test_thyra_vectors.cpp
 * Author: Paul Cazeaux
 *
 * Created on February 22, 2019, 9:00 AM
 */

#include "tests.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "parameters/multilayer.h"
#include "bilayer/vectorspace.h"
#include "bilayer/vector.h"
#include "tools/numbers.h"

static const int degree = 1;
typedef typename types::DefaultNode Node;

struct TestAlgebra : public Bilayer::BaseAlgebra<2,degree,double,Node>
{
    public:
        typedef typename Bilayer::BaseAlgebra<2,degree,double,Node> LA;
        typedef typename Bilayer::VectorSpace<2,degree,double,Node> VS;
        typedef typename Bilayer::Vector<2,degree,double,Node> Vec;
        typedef typename Bilayer::MultiVector<2,degree,double,Node> MVec;

        Teuchos::RCP<VS>    vectorSpace;

        Teuchos::RCP<Vec>   I_1, I_2;
        Teuchos::RCP<MVec>   M_1, M_2;

        TestAlgebra(Multilayer<2, 2> bilayer);
};

TestAlgebra::TestAlgebra(Multilayer<2, 2> bilayer):
    LA(bilayer)
{
    LA::assemble_base_matrices();

    vectorSpace = VS::create();
    vectorSpace->initialize(Teuchos::rcpFromRef(* this));

        Teuchos::RCP<Tpetra::MultiVector<double,types::loc_t,types::glob_t,Node>>
    mvec_init = Tpetra::createMultiVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getOrbVecMap(), vectorSpace->getNumOrbitals());


        Teuchos::RCP<Tpetra::Vector<double,types::loc_t,types::glob_t,Node>>
    vec_init = Tpetra::createVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() );

    I_1 = Bilayer::createVector<2,degree,double,Node>( vectorSpace, mvec_init);
    I_2 = Bilayer::createVector<2,degree,double,Node>( vectorSpace, vec_init );

        Teuchos::RCP<Tpetra::MultiVector<double,types::loc_t,types::glob_t,Node>>
    vec_init2 = Tpetra::createMultiVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), 3);
        Teuchos::RCP<Tpetra::MultiVector<double,types::loc_t,types::glob_t,Node>>
    mvec_init2 = Tpetra::createMultiVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getOrbVecMap(), 3*vectorSpace->getNumOrbitals());


    M_1 = Bilayer::createMultiVector<2,degree,double,Node>
        ( vectorSpace, mvec_init2);
    M_2 = Bilayer::createMultiVector<2,degree,double,Node>
        ( vectorSpace, vec_init2 );


    LA::set_to_identity(* I_1->getThyraOrbVector()->getTpetraMultiVector());
    LA::set_to_identity(* I_2->getThyraOrbVector()->getTpetraMultiVector());
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
    bilayer.layer_data[0] = std::make_unique<LayerData<2>>(mat, 0.,  0.,   1.);
    bilayer.layer_data[1] = std::make_unique<LayerData<2>>(mat, height, 5.71 * numbers::PI/180, 1.);

    TestAlgebra test_algebra (bilayer);

    test_algebra.I_2->update( -1.0, * test_algebra.I_1);
    Teuchos::Array<double> norms (1);
    test_algebra.I_2 ->norms_inf(norms);

    AssertThrow( std::accumulate(norms.begin(), norms.end(), 0.) < 1e-10,
                        dealii::ExcInternalError() );

    test_algebra.M_2->update( -1.0, * test_algebra.M_1);
    norms .resize(3);
    test_algebra.M_2 ->norms_inf(norms);

    AssertThrow( std::accumulate(norms.begin(), norms.end(), 0.) < 1e-10,
                        dealii::ExcInternalError() );
    std::cout << "Vector creation and manipulation OK" << std::endl;
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