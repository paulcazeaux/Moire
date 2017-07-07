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
        std::array<MultiVector, 2> I, H;

        std::tuple<types::block_t,double,double,types::loc_t> locate_dof(const types::block_t range_block, const types::glob_t i); 
};

TestAlgebra::TestAlgebra(Multilayer<2, 2> bilayer) :
    LA(bilayer)
{
    LA::base_setup();

    I  = {{ MultiVector( dof_handler.locally_owned_dofs(0), dof_handler.n_range_orbitals(0,0) ), 
            MultiVector( dof_handler.locally_owned_dofs(1), dof_handler.n_range_orbitals(1,1) ) }};
    H  = {{ Tpetra::createCopy(I[0]),  Tpetra::createCopy(I[1]) }};

    LA::assemble_hamiltonian_action();
    LA::create_identity(I);
    LA::hamiltonian_rproduct(I, H);
}

std::tuple<types::block_t,double,double,types::loc_t>
TestAlgebra::locate_dof(const types::block_t range_block, const types::glob_t i)
{
    auto [domain_block, lattice_index, cell_index, orbital] 
            = dof_handler.get_dof_info(range_block, i);
    dealii::Point<2> point_position = dof_handler.lattice(domain_block).get_vertex_position (lattice_index);
    if (orbital == 1)
    {
        double theta = dof_handler.layer(domain_block).angle;
        point_position(0) +=  std::cos(theta) * Graphene::atom_pos.at (Graphene::Atom::B)[0] + std::sin(theta) * Graphene::atom_pos.at (Graphene::Atom::B)[1];
        point_position(1) += -std::sin(theta) * Graphene::atom_pos.at (Graphene::Atom::B)[0] + std::cos(theta) * Graphene::atom_pos.at (Graphene::Atom::B)[1];
    }
    if (range_block != domain_block)
        point_position += dof_handler.unit_cell(1-range_block).get_node_position (cell_index);

    return std::make_tuple(domain_block, point_position(0), point_position(1), orbital);
}

 void do_test()
 {
    Multilayer<2, 2> bilayer (
                "test_hamiltonian", 
                "none.jld",
                ObservableType::DoS,
                Graphene::inter_search_radius,
                1,   
                1, 0,
                0, 0,
                6., 2);
    bilayer.layer_data[0] = std::make_unique<LayerData<2>>(Materials::Mat::Graphene, 0.,  0.,   1.);
    bilayer.layer_data[1] = std::make_unique<LayerData<2>>(Materials::Mat::Graphene, 3.4, 5.71 * numbers::PI/180, 1.);

    TestAlgebra test_algebra (bilayer);

    Teuchos::ArrayRCP<const double> H = test_algebra.H.at(0).getData(0);

    std::cout << "X = [";
    for (size_t i=0; i<H.size(); ++i)
    {
        auto [b,x,y,o] = test_algebra.locate_dof(0, i);
        std::cout << static_cast<int>(b) << ", " << x << ", " << y << ", " << o << ", " << H[i] << (i+1 < H.size() ? "; " : "];\n");
    }
    std::cout << "x0 = X(X(:,1)==0, 2); y0 = X(X(:,1)==0, 3); o0 = X(X(:,1)==0, 4); h0 = X(X(:,1)==0, 5);" << std::endl;
    std::cout << "x1 = X(X(:,1)==1, 2); y1 = X(X(:,1)==1, 3); o1 = X(X(:,1)==1, 4); h1 = X(X(:,1)==1, 5);" << std::endl;
 }


 int main(int argc, char** argv) {
        /*********************************************************/
        /*                  Initialize MPI                       */
        /*********************************************************/
        Teuchos::GlobalMPISession mpiSession (&argc, &argv, NULL);
    try
    {
        do_test();
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