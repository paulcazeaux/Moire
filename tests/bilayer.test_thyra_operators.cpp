/* 
 * Test file:   bilayer.test_thyra_operators.cpp
 * Author: Paul Cazeaux
 *
 * Created on February 22, 2019, 9:00 AM
 */

#include "tests.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "parameters/multilayer.h"
#include "bilayer/vectorspace.h"
#include "bilayer/operator.h"
#include "tools/numbers.h"

using Teuchos::RCP;

static const int dim = 2;
static const int degree = 1;
typedef typename Kokkos::Compat::KokkosSerialWrapperNode Node;

struct TestAlgebra : public Bilayer::BaseAlgebra<dim,degree,double,Node>
{
    public:
        typedef typename Bilayer::BaseAlgebra<dim,degree,double,Node> LA;
        typedef typename Bilayer::VectorSpace<dim,degree,double,Node> VS;
        typedef typename Bilayer::Vector<dim,degree,double,Node> Vec;
        typedef typename Bilayer::MultiVector<dim,degree,double,Node> MVec;
        typedef typename Bilayer::Operator<dim,degree,double,Node> Op;

        RCP<VS>    vectorSpace;

        RCP<Vec>   I, H, tH;
        RCP<MVec>   dH, LdH;

        RCP<const Op> hamiltonianOp, transposeOp, liouvillianOp, derivationOp;

        TestAlgebra(Multilayer<dim, 2> bilayer);
};

TestAlgebra::TestAlgebra(Multilayer<dim, 2> bilayer):
    LA(bilayer)
{
    LA::assemble_base_matrices();

    vectorSpace = VS::create();
    vectorSpace->initialize(Teuchos::rcpFromRef(* this));

    
    I = Bilayer::createVector<dim,degree,double,Node>( vectorSpace, 
        Tpetra::createVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
    H = Bilayer::createVector<dim,degree,double,Node>( vectorSpace, 
        Tpetra::createVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
    tH = Bilayer::createVector<dim,degree,double,Node>( vectorSpace, 
        Tpetra::createVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
    dH = Bilayer::createMultiVector<dim,degree,double,Node>
        ( vectorSpace, 
        Tpetra::createMultiVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim) );
    LdH = Bilayer::createMultiVector<dim,degree,double,Node>
        ( vectorSpace, 
        Tpetra::createMultiVector<double,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim) );


    hamiltonianOp = Bilayer::createConstOperator<dim,degree,double,Node>
        ( vectorSpace, LA::HamiltonianAction);
    transposeOp = Bilayer::createConstOperator<dim,degree,double,Node>
        ( vectorSpace, LA::Transpose);
    liouvillianOp = Bilayer::createConstOperator<dim,degree,double,Node>
        (vectorSpace, LA::make_liouvillian_operator());
    derivationOp = Bilayer::createConstOperator<dim,degree,double,Node>
        (vectorSpace, LA::Derivation);


    LA::set_to_identity(* I->getThyraOrbVector()->getTpetraMultiVector());
    
    hamiltonianOp->apply(Thyra::NOTRANS, *I, H .ptr(), 1.0, 0.0);
    transposeOp  ->apply(Thyra::NOTRANS, *H, tH.ptr(), 1.0, 0.0);
    derivationOp ->apply(Thyra::NOTRANS, *H, dH.ptr(), 1.0, 0.0);
    liouvillianOp->apply(Thyra::NOTRANS, *dH, LdH.ptr(), 1.0, 0.0);

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


    // AssertThrow( std::accumulate(norms.begin(), norms.end(), 0.) < 1e-10,
    //                     dealii::ExcInternalError() );
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