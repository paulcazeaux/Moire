/* 
 * Test file:   bilayer.test_thyra_operators.cpp
 * Author: Paul Cazeaux
 *
 * Created on February 22, 2019, 9:00 AM
 */

#include "tests.h"

#include <Teuchos_GlobalMPISession.hpp>
#include <Thyra_VectorStdOps_decl.hpp>

#include "parameters/multilayer.h"
#include "bilayer/vectorspace.h"
#include "bilayer/operator.h"
#include "tools/numbers.h"

using Teuchos::RCP;

static const int dim = 2;
static const int degree = 2;
typedef double Scalar;
typedef typename types::DefaultNode Node;

struct TestAlgebra : public Bilayer::BaseAlgebra<dim,degree,Scalar,Node>
{
    public:
        typedef typename Bilayer::BaseAlgebra<dim,degree,Scalar,Node> LA;
        typedef typename Bilayer::VectorSpace<dim,degree,Scalar,Node> VS;
        typedef typename Thyra::VectorBase<Scalar> Vec;
        typedef typename Thyra::MultiVectorBase<Scalar> MVec;
        typedef typename Thyra::LinearOpBase<Scalar> Op;

        RCP<VS>    vectorSpace;

        RCP<Vec>   I, AI, H, AH, LH, LI; // AI, tHAI, tAtHAI, H, AH, tHAH, tAtHAH, tAH, LH, LI;
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

    Teuchos::RCP<Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    I_tpetra = Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getOrbVecMap(), vectorSpace->getNumOrbitals() );
    LA::set_to_identity(* I_tpetra);
    
    I = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
        I_tpetra);
    AI = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
        Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
    H = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
        Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
    AH = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
        Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));

    LI = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
        Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
    LH = Bilayer::createVector<dim,degree,Scalar,Node>( vectorSpace, 
        Tpetra::createVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap() ));
    dH = Bilayer::createMultiVector<dim,degree,Scalar,Node>
        ( vectorSpace, 
        Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim) );
    LdH = Bilayer::createMultiVector<dim,degree,Scalar,Node>
        ( vectorSpace, 
        Tpetra::createMultiVector<Scalar,types::loc_t,types::glob_t,Node>
                        ( vectorSpace->getVecMap(), dim) );

    hamiltonianOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
        ( vectorSpace, LA::HamiltonianAction);
    transposeOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
        ( vectorSpace, LA::Transpose);
    liouvillianOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
        (vectorSpace, LA::make_liouvillian_operator());
    derivationOp = Bilayer::createConstOperator<dim,degree,Scalar,Node>
        (vectorSpace, LA::Derivation);
    
    hamiltonianOp->apply(Thyra::NOTRANS, *I, H .ptr(), 1.0, 0.0);
    transposeOp  ->apply(Thyra::NOTRANS, *I, AI.ptr(), 1.0, 0.0);
    transposeOp  ->apply(Thyra::NOTRANS, *H, AH.ptr(), 1.0, 0.0);

    liouvillianOp->apply(Thyra::NOTRANS, *I, LI.ptr(), 1.0, 0.0);
    liouvillianOp->apply(Thyra::NOTRANS, *H, LH.ptr(), 1.0, 0.0);
    derivationOp ->apply(Thyra::NOTRANS, *H, dH.ptr(), 1.0, 0.0);
    liouvillianOp->apply(Thyra::NOTRANS, *dH, LdH.ptr(), 1.0, 0.0);
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
            1.0, 1);

    bilayer.layer_data[0] = std::make_unique<LayerData<dim>>(mat, 0.,  0.,   1.);

    double height;
    switch (mat)
    {
        case Materials::Mat::Graphene:
            height = 3.4;
            bilayer.layer_data[1] = std::make_unique<LayerData<dim>>
                (mat, height, 5.71 * numbers::PI/180, 1.);
            break;
        case Materials::Mat::MoS2:
        case Materials::Mat::WS2:
            height = 6.145;
            bilayer.layer_data[1] = std::make_unique<LayerData<dim>>
                (mat, height, 5.71 * numbers::PI/180, 1.);
            break;
        case Materials::Mat::MoSe2:
        case Materials::Mat::WSe2:
            height = 6.48;
            bilayer.layer_data[1] = std::make_unique<LayerData<dim>>
                (mat, height, 5.71 * numbers::PI/180, 1.);
            break;


        case Materials::Mat::Toy1D:
            bilayer.layer_data[1] = std::make_unique<LayerData<dim>>
                (mat, height, 0., 1.1);
            break;

        default:
            throw;
    }

    TestAlgebra test_algebra (bilayer);
    Teuchos::Array<Teuchos::ScalarTraits<Scalar>::magnitudeType> norms (1);


    test_algebra.AI->update( -1.0, * test_algebra.I);
    test_algebra.AI ->norms_2(norms);

    AssertThrow( std::accumulate(norms.begin(), norms.end(), 0.) < 1e-10,
                        dealii::ExcInternalError() );

    
    test_algebra.H ->norms_2(norms);
    std::cout << "Norm of H: " << norms[0] << std::endl;
    std::cout << "\t<H,AH>:\t" << Thyra::dot(*test_algebra.H,   *test_algebra.AH) << std::endl;

    test_algebra.AH->update( -1.0, * test_algebra.H);
    test_algebra.AH ->norms_2(norms);
    std::cout << "Norm of AH - H: " << norms[0] << std::endl << std::endl;

    test_algebra.LI ->norms_2(norms);
    std::cout << "Norm of L(I): " << norms[0] << std::endl;
    test_algebra.LH ->norms_2(norms);
    std::cout << "Norm of L(H): " << norms[0] << std::endl;
    test_algebra.LdH ->norms_inf(norms);

    std::cout << "\t<I,L(H)>:\t" << Thyra::dot(*test_algebra.I,   *test_algebra.LH) 
              << "\t<H,L(H)>:\t" << Thyra::dot(*test_algebra.H,   *test_algebra.LH) << std::endl
              << "\t<H,AH>-<H,H>:\t" << Thyra::dot(*test_algebra.H,   *test_algebra.AH) 
                                        - Thyra::dot(*test_algebra.H,  *test_algebra.H) << std::endl;

    std::cout << std::endl << "Norm of L(dH): " << norms[0] << std::endl << std::endl;


    std::cout << "Vector creation and manipulation OK" << std::endl;
 }


 int main(int argc, char** argv) {
        /*********************************************************/
        /*                  Initialize MPI                       */
        /*********************************************************/
        Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
    try
    {
        if (dim == 1)
        {
            do_test(Materials::Mat::Toy1D);
        }
        if (dim == 2)
        {
            do_test(Materials::Mat::Graphene);
            do_test(Materials::Mat::MoS2);
            do_test(Materials::Mat::WS2);
            do_test(Materials::Mat::MoSe2);
            do_test(Materials::Mat::WSe2);
        }
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