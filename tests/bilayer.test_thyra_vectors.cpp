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

using Teuchos::RCP;

static const int degree = 1;
static const int dim = 2;
typedef double Scalar;
typedef typename types::loc_t LO;
typedef typename types::glob_t GO;
typedef typename types::DefaultNode Node;

struct TestAlgebra : public Bilayer::BaseAlgebra<dim,degree,Scalar,Node>
{
    public:
        typedef typename Bilayer::BaseAlgebra<dim,degree,Scalar,Node> LA;
        typedef typename Bilayer::VectorSpace<dim,degree,Scalar,Node> VS;
        typedef typename Thyra::VectorBase<Scalar> Vec;
        typedef typename Bilayer::Vector<dim,degree,Scalar,Node> BVec;
        typedef typename Thyra::MultiVectorBase<Scalar> MVec;
        typedef typename Bilayer::MultiVector<dim,degree,Scalar,Node> BMVec;

        RCP<VS>    vectorSpace;

        RCP<Vec>    I_1, I_2;
        RCP<MVec>   M_1, M_2;

        TestAlgebra(Multilayer<dim, 2> bilayer);
};

TestAlgebra::TestAlgebra(Multilayer<dim, 2> bilayer):
    LA(bilayer)
{
    LA::assemble_base_matrices();

    vectorSpace = VS::create();
    vectorSpace->initialize(Teuchos::rcpFromRef(* this));

        RCP<Tpetra::MultiVector<Scalar,LO,GO,Node>>
    mvec_init = Tpetra::createMultiVector<Scalar,LO,GO,Node>
                        ( vectorSpace->getOrbVecMap(), vectorSpace->getNumOrbitals());

        RCP<Tpetra::Vector<Scalar,LO,GO,Node>>
    vec_init = Tpetra::createVector<Scalar,LO,GO,Node>
                        ( vectorSpace->getVecMap() );

    LA::set_to_identity(* mvec_init);
    I_1 = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, mvec_init);
    I_2 = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, vec_init );
    RCP<BVec> BI_2 = Teuchos::rcp_dynamic_cast<BVec>(I_2);
    LA::set_to_identity(* BI_2->getThyraOrbVector()->getTpetraMultiVector());

        RCP<Tpetra::MultiVector<Scalar,LO,GO,Node>>
    vec_init2 = Tpetra::createMultiVector<Scalar,LO,GO,Node>
                        ( vectorSpace->getVecMap(), 3);
        RCP<Tpetra::MultiVector<Scalar,LO,GO,Node>>
    mvec_init2 = Tpetra::createMultiVector<Scalar,LO,GO,Node>
                        ( vectorSpace->getOrbVecMap(), 3*vectorSpace->getNumOrbitals());


    M_1 = Bilayer::createMultiVector<dim,degree,Scalar,Node>
        ( vectorSpace, mvec_init2);
    M_2 = Bilayer::createMultiVector<dim,degree,Scalar,Node>
        ( vectorSpace, vec_init2 );


    M_1->assign(1.0);
    M_2->assign(1.0);
    M_1->col(1)->scale(2.0);
    M_2->col(1)->scale(2.0);
    M_1->col(2)->assign(3.0);
    M_2->col(2)->assign(3.0);
}

 void do_test(Materials::Mat mat)
 {
    Multilayer<dim, 2> bilayer (
            "test_hamiltonian", 
            "none.jld",
            ObservableType::DoS,
            Materials::inter_search_radius(mat),
            1,   
            1, 0,
            0, 0,
            2.0, 2);

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