/**
* File:   bilayer/base_algebra.cpp
* Author: Paul Cazeaux
*
* Created on June 30, 2017, 7:00 PM
*/


#include "bilayer/base_algebra.h"

namespace Bilayer {

template<int dim, int degree, typename Scalar, class Node>
BaseAlgebra<dim,degree,Scalar,Node>::BaseAlgebra(const Multilayer<dim, 2>& bilayer)
    :
    mpi_communicator(Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ()),
    dof_handler(bilayer)
{
    dof_handler.initialize(mpi_communicator);
    Assert(dof_handler.locally_owned_dofs()->isContiguous(), dealii::ExcNotImplemented());
}


template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::assemble_base_matrices()
{
    this->assemble_transpose_interpolant();
    this->assemble_hamiltonian_action();
    Derivation = Teuchos::rcp(new DerivationOp(Teuchos::rcpFromRef(this->dof_handler)));
}    


template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::assemble_hamiltonian_action()
{
    for (types::block_t range_block = 0; range_block < 2; ++range_block)
    {
        hamiltonian_action.at(range_block) =  RCP<Matrix>(new Matrix(dof_handler.make_sparsity_pattern_hamiltonian_action(range_block)) );

        std::vector<types::glob_t> globalRows;
        std::vector<Teuchos::Array<types::glob_t>> ColIndices;
        std::vector<Teuchos::Array<Scalar>> Values;

        for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
            for (types::loc_t n=0; n < dof_handler.n_locally_owned_points(domain_block); ++n)
            {
                const PointData& this_point = dof_handler.locally_owned_point(domain_block, n);

                dealii::Point<dim> 
                this_point_position = lattice(domain_block).get_vertex_position(this_point.lattice_index);
                std::array<types::loc_t, dim> 
                this_point_grid_indices = lattice(domain_block).get_vertex_grid_indices(this_point.lattice_index);

                        /* Block b <-> b */
                globalRows.clear();
                for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(); ++cell_index)
                    for (size_t orbital = 0; orbital <  dof_handler.n_orbitals(domain_block); orbital++)
                        globalRows.push_back(dof_handler.get_dof_index(domain_block, this_point.lattice_index, cell_index, orbital));

                ColIndices.resize(globalRows.size());
                Values.resize(globalRows.size());
                for (auto & cols : ColIndices) 
                    cols.clear();
                for (auto & vals : Values) 
                    vals.clear();

                std::vector<types::loc_t>   
                neighbors = lattice(domain_block).list_neighborhood_indices(this_point_position, layer(domain_block).intra_search_radius);
                for (auto neighbor_lattice_index : neighbors)
                {
                    std::array<types::loc_t,dim> 
                    grid_vector = lattice(domain_block).get_vertex_grid_indices(neighbor_lattice_index);
                    for (size_t j=0; j<dim; ++j)
                        grid_vector[j] = this_point_grid_indices[j] - grid_vector[j];

                    auto it_cols = ColIndices.begin();
                    auto it_vals = Values.begin();
                    for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(); ++cell_index)
                        for (size_t orbital_row = 0; orbital_row <  dof_handler.n_orbitals(domain_block); orbital_row++)
                        {
                            for (size_t orbital_col = 0; orbital_col <  dof_handler.n_orbitals(domain_block); orbital_col++)
                            {
                                it_cols->push_back(dof_handler.get_dof_index(domain_block, neighbor_lattice_index, cell_index, orbital_col));
                                it_vals->push_back(dof_handler.intralayer_term(orbital_col, orbital_row, grid_vector, domain_block ));
                            }
                            it_cols++; 
                            it_vals++;
                        }
                }

                        /* Block a -> b, a != b */
                types::block_t other_domain_block = 1-this_point.domain_block;
                neighbors = lattice(other_domain_block).list_neighborhood_indices( this_point_position, 
                                            dof_handler.inter_search_radius + unit_cell(1-range_block).bounding_radius);
                        
                for (auto neighbor_lattice_index : neighbors)
                {
                    auto it_cols = ColIndices.begin();
                    auto it_vals = Values.begin();
                    for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(); ++cell_index)
                    {
                        /* Note: the unit cell node displacement follows the point which is in the interlayer block */
                        dealii::Tensor<1,dim> arrow_vector = this_point_position 
                                                            + (domain_block != range_block ? 1. : -1.) 
                                                                * unit_cell(1-range_block).get_node_position(cell_index)
                                                            - lattice(other_domain_block).get_vertex_position(neighbor_lattice_index);
                        
                        for (size_t orbital_row = 0; orbital_row <  dof_handler.n_orbitals(domain_block); orbital_row++)
                            for (size_t orbital_col = 0; orbital_col < dof_handler.n_orbitals(other_domain_block); orbital_col++)
                            {
                                it_cols[orbital_row].push_back(dof_handler.get_dof_index(other_domain_block, neighbor_lattice_index, cell_index, orbital_col));
                                it_vals[orbital_row].push_back(dof_handler.interlayer_term(orbital_col, orbital_row, arrow_vector, other_domain_block, domain_block ));
                            }
                        it_cols += dof_handler.n_orbitals(other_domain_block); 
                        it_vals += dof_handler.n_orbitals(other_domain_block);
                    }
                }

                for (size_t i = 0; i < globalRows.size(); ++i)
                    hamiltonian_action.at(range_block)->replaceGlobalValues(globalRows.at(i), ColIndices.at(i), Values.at(i));
            }
        hamiltonian_action.at(range_block)->fillComplete ();
    }

    HamiltonianAction = Teuchos::rcp(new RangeBlockOp(
                            hamiltonian_action, 
                            {dof_handler.n_orbitals(0), dof_handler.n_orbitals(1)}));
}

template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::assemble_transpose_interpolant()
{
    for (types::block_t range_block = 0; range_block < 2; ++range_block)
        for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
        {
            /* First, we initialize the matrix with a static CrsGraph computed by the dof_handler object */
            transpose_interpolant.at(range_block).at(domain_block) 
                =  RCP<Matrix>(new Matrix(dof_handler.make_sparsity_pattern_transpose_interpolant(range_block, domain_block)));

            
            const types::loc_t n_orbitals = dof_handler.n_orbitals(domain_block);
            std::vector<types::glob_t> globalRows;
            std::vector<Teuchos::Array<types::glob_t>> ColIndices;
            std::vector<Teuchos::Array<Scalar>> Values;


                /* First, the case of diagonal blocks */
            if (range_block == domain_block)
                for (types::loc_t n=0; n < dof_handler.n_locally_owned_points(range_block); ++n)
                {
                    const PointData& this_point = dof_handler.locally_owned_point(range_block, n);
                    assert(this_point.domain_block == domain_block);

                    size_t N = n_orbitals * this_point.intra_interpolating_nodes.size();
                    globalRows.resize(N);
                    ColIndices.resize(N);
                    Values.resize(N);

                    for (auto & cols : ColIndices) 
                        cols.clear();
                    for (auto & vals : Values)
                        vals.clear();

                    auto it_row = globalRows.begin();
                    auto it_cols = ColIndices.begin();
                    auto it_vals = Values.begin();
                    for (const auto & interp_point : this_point.intra_interpolating_nodes)
                    {
                        const auto [cell_index, interp_range_block, interp_domain_block, interp_lattice_index, interp_element_index, interp_weights] = interp_point;
                        assert(interp_domain_block == domain_block);
                        assert(interp_range_block == range_block);

                        for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                            it_row[orbital] = dof_handler.get_transpose_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index, orbital);

                        /* Iterate through the nodes of the interpolating element */
                        for (types::loc_t j = 0; j < Element<dim,degree>::dofs_per_cell; ++j)
                        {
                            types::loc_t interp_cell_index = unit_cell(domain_block).subcell_list [interp_element_index].unit_cell_dof_index_map.at(j);
                            if (unit_cell(domain_block).is_node_interior(interp_cell_index))
                                for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                                {
                                    it_cols[orbital].push_back(dof_handler.get_block_dof_index(domain_block, interp_lattice_index, interp_cell_index, orbital));
                                    it_vals[orbital].push_back(interp_weights.at(j));
                                }
                            else // Boundary point!
                            {
                                /* Periodic wrap */
                                auto [offset_interp_cell_index, offset_indices] = unit_cell(domain_block).map_boundary_point_interior(interp_cell_index);
                            
                                for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                                {
                                    it_cols[orbital].push_back(dof_handler.get_block_dof_index(domain_block, interp_lattice_index, offset_interp_cell_index, orbital));
                                    it_vals[orbital].push_back(interp_weights.at(j));
                                }
                            }
                        }
                        it_row += n_orbitals;
                        it_cols += n_orbitals;
                        it_vals += n_orbitals;
                    }
                    for (size_t i = 0; i < globalRows.size(); ++i)
                        transpose_interpolant.at(range_block).at(domain_block)->replaceGlobalValues(globalRows.at(i), ColIndices.at(i), Values.at(i));
                }
            else
                /* Now the case of extradiagonal blocks */
                for (types::loc_t n=0; n < dof_handler.n_locally_owned_points(range_block); ++n)
                {
                    const PointData& this_point = dof_handler.locally_owned_point(range_block, n);

                    size_t N = n_orbitals * this_point.inter_interpolating_nodes.size();
                    globalRows.resize(N);
                    ColIndices.resize(N);
                    Values.resize(N);
                    for (auto & cols : ColIndices) 
                        cols.clear();
                    for (auto & vals : Values)
                        vals.clear();

                    auto it_row = globalRows.begin();
                    auto it_cols = ColIndices.begin();
                    auto it_vals = Values.begin();
                    for (const auto & interp_point : this_point.inter_interpolating_nodes)
                    {
                        const auto [cell_index, interp_range_block, interp_domain_block, interp_lattice_index, interp_element_index, interp_weights] = interp_point;
                        assert(interp_domain_block == domain_block);
                        assert(interp_range_block == range_block);

                        /* Store row index for each orbital */
                        for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                            it_row[orbital] = dof_handler.get_transpose_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index, orbital);

                        /* Iterate through the nodes of the interpolating element */
                        for (types::loc_t j = 0; j < Element<dim,degree>::dofs_per_cell; ++j)
                        {
                            types::loc_t interp_cell_index = unit_cell(domain_block).subcell_list [interp_element_index].unit_cell_dof_index_map.at(j);
                            
                            /* Store column indices for each orbital */
                            if (unit_cell(domain_block).is_node_interior(interp_cell_index))
                                for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                                {
                                    it_cols[orbital].push_back(dof_handler.get_block_dof_index(domain_block, interp_lattice_index, interp_cell_index, orbital));
                                    it_vals[orbital].push_back(interp_weights.at(j));
                                }
                            else // Boundary point!
                            {
                                auto [offset_interp_cell_index, offset] = unit_cell(domain_block).map_boundary_point_interior(interp_cell_index);
                                const types::loc_t offset_interp_lattice_index = lattice(domain_block).offset_global_index(interp_lattice_index, offset);

                                /* Check that this point exists in our cutout */
                                if (offset_interp_lattice_index != types::invalid_local_index)
                                    for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                                    {
                                        it_cols[orbital].push_back(dof_handler.get_block_dof_index(domain_block, offset_interp_lattice_index, offset_interp_cell_index, orbital));
                                        it_vals[orbital].push_back(interp_weights.at(j));
                                    }
                            }
                        }
                        it_row += n_orbitals;
                        it_cols += n_orbitals;
                        it_vals += n_orbitals;
                    }
                    

                    for (size_t i = 0; i < globalRows.size(); ++i)
                        transpose_interpolant.at(range_block).at(domain_block)->replaceGlobalValues(globalRows.at(i), ColIndices.at(i), Values.at(i));
                }
            transpose_interpolant.at(range_block).at(domain_block)->fillComplete ();
        }

    Transpose = Teuchos::rcp(new TransposeOp(
                                    transpose_interpolant, 
                                    {dof_handler.n_orbitals(0), dof_handler.n_orbitals(1)},
                                    {unit_cell(0).area, unit_cell(1).area},
                                    this->dof_handler.locally_owned_dofs(),
                                    this->dof_handler.locally_owned_dofs()));
}

/* Create a basic MultiVector with the right data structure */
template<int dim, int degree, typename Scalar, class Node>
typename BaseAlgebra<dim,degree,Scalar,Node>::Vector
BaseAlgebra<dim,degree,Scalar,Node>::create_vector(bool ZeroOut) const
{
    return MultiVector( this->dof_handler.locally_owned_dofs(), this->dof_handler.n_orbitals(0) + this->dof_handler.n_orbitals(1), ZeroOut);
}

/* Create a basic MultiVector with the right data structure */
template<int dim, int degree, typename Scalar, class Node>
typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector
BaseAlgebra<dim,degree,Scalar,Node>::create_multivector(size_t numVecs, bool ZeroOut) const
{
    return MultiVector( this->dof_handler.locally_owned_dofs(), numVecs * (this->dof_handler.n_orbitals(0) + this->dof_handler.n_orbitals(1)), ZeroOut);
}


template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::set_to_identity(Vector& Id) const
{
    Id.putScalar(0.);

    std::array<types::loc_t, dim> lattice_indices_0;
    for (size_t i=0; i<dim; ++i)
        lattice_indices_0[i] = 0;
    
    types::loc_t n0 = dof_handler.n_orbitals(0);
    for (types::block_t b = 0; b < 2; ++b)
    {
        types::loc_t lattice_index_0 = lattice(b).get_vertex_global_index(lattice_indices_0);
        if (dof_handler.is_locally_owned_point(b,lattice_index_0))
            for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(); ++cell_index)
                for (size_t orbital = 0; orbital < dof_handler.n_orbitals(b); ++orbital)
                    Id.replaceGlobalValue(dof_handler.get_dof_index(b, lattice_index_0, cell_index, orbital), b == 0 ? orbital : orbital + n0, 1.);
    }
}


template<int dim, int degree, typename Scalar, class Node>
std::array<std::vector<Scalar>,2>
BaseAlgebra<dim,degree,Scalar,Node>::diagonal(const Vector& A) const
{
    std::array<std::vector<Scalar>,2> Diag;

    for (types::block_t b = 0; b<2; ++b)
    {
        Diag.at(b).resize(dof_handler.n_cell_nodes() * dof_handler.n_orbitals(b), 0.0);

        typename MultiVector::dual_view_type::t_dev_const 
        View = A.subView(dof_handler.column_range(b))-> template getLocalView<Kokkos::Serial>();

        std::array<types::loc_t, dim> lattice_indices_0;
        for (size_t i=0; i<dim; ++i)
            lattice_indices_0[i] = 0;
        /* Add diagonal values on the current process */

        int origin_owner = dof_handler.point_owner(b, lattice(b).index_origin);

        if (dof_handler.my_pid == origin_owner)
        {
            types::loc_t start_zero = dof_handler.locally_owned_dofs()->
                                                        getLocalElement( 
                                                        dof_handler.get_dof_index(b, lattice(b).index_origin, 0, 0) );
            size_t idx = 0;
            for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(); ++cell_index)
                for (size_t orbital = 0; orbital < dof_handler.n_orbitals(b); ++orbital)
                {
                    Diag.at(b).at(idx) = View(start_zero + idx, orbital);
                    ++idx;
                }
        }
        Teuchos::broadcast<int, Scalar>(* mpi_communicator, origin_owner, Diag.at(b).size(), Diag.at(b).data());
    }
    return Diag;
}

template<int dim, int degree, typename Scalar, class Node>
Scalar
BaseAlgebra<dim,degree,Scalar,Node>::trace(const Vector& A) const
{
    std::array<std::vector<Scalar>,2> Diag = diagonal(A);

    return std::accumulate(Diag[0].begin(), Diag[0].end(), Teuchos::ScalarTraits<Scalar>::zero())
                                * unit_cell(1).area / (unit_cell(0).area + unit_cell(1).area)
                                / static_cast<double>( dof_handler.n_orbitals(0) * dof_handler.n_cell_nodes() )
                    + std::accumulate(Diag[1].begin(), Diag[1].end(), Teuchos::ScalarTraits<Scalar>::zero())
                                * unit_cell(0).area / (unit_cell(0).area + unit_cell(1).area)
                                / static_cast<double>( dof_handler.n_orbitals(1) * dof_handler.n_cell_nodes() );
}

template<int dim, int degree, typename Scalar, class Node>
Scalar
BaseAlgebra<dim,degree,Scalar,Node>::dot(const Vector& A, const Vector& B) const
{
    size_t    
    n1 = dof_handler.n_orbitals(0),
    n2 = dof_handler.n_orbitals(1);

    Teuchos::Array<Scalar> dots_tpetra( n1+n2 );
    A.dot(B, dots_tpetra() );
    Scalar 
    s1 = Teuchos::ScalarTraits<Scalar>::zero(), 
    s2 = Teuchos::ScalarTraits<Scalar>::zero(),
    a1 = static_cast<Scalar>(unit_cell(1).area 
                            / (unit_cell(0).area + unit_cell(1).area)
                            / dof_handler.n_cell_nodes() ),
    a2 = static_cast<Scalar>(unit_cell(0).area 
                            / (unit_cell(0).area + unit_cell(1).area)
                            / dof_handler.n_cell_nodes() );


    for (size_t o1=0; o1 < n1; ++o1)
        s1 += dots_tpetra[o1];
    for (size_t o2=0; o2 < n2; ++o2)
        s2 += dots_tpetra[n1+o2];

    return a1 * s1 + a2 * s2;
}


template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::dot(
        const MultiVector& A, 
        const MultiVector& B, 
        Teuchos::ArrayView<Scalar> dots) const
{
    size_t    
    n1 = dof_handler.n_orbitals(0),
    n2 = dof_handler.n_orbitals(1);

    assert( A.getNumVectors() % (n1 + n2) == 0);
    assert( A.getNumVectors() / (n1 + n2) == static_cast<size_t>(dots.size()));
    assert( B.getNumVectors() / (n1 + n2) == static_cast<size_t>(dots.size()));

    Teuchos::Array<Scalar> dots_tpetra( n1+n2 );
    A.dot(B, dots_tpetra() );
    Scalar 
    s1 = Teuchos::ScalarTraits<Scalar>::zero(), 
    s2 = Teuchos::ScalarTraits<Scalar>::zero(),
    a1 = static_cast<Scalar>(unit_cell(1).area 
                            / (unit_cell(0).area + unit_cell(1).area)
                            / dof_handler.n_cell_nodes() ),
    a2 = static_cast<Scalar>(unit_cell(0).area 
                            / (unit_cell(0).area + unit_cell(1).area)
                            / dof_handler.n_cell_nodes() );

    for (size_t j=0; j < A.getNumVectors() / (n1 + n2); ++j)
    {
        for (size_t o1=0; o1 < n1; ++o1)
            s1 += dots_tpetra[o1];
        for (size_t o2=0; o2 < n2; ++o2)
            s2 += dots_tpetra[n1+o2];

        dots[j] = a1 * s1 + a2 * s2;
    }
}

// /* Const Views into the data in range block form */
// template<int dim, int degree, typename Scalar, class Node>
// std::array<const typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector, 2> 
// BaseAlgebra<dim,degree,Scalar,Node>::range_block_view_const(const MultiVector& A)
// {
//     return {{   * A.subView(dof_handler.column_range(0)), 
//                 * A.subView(dof_handler.column_range(1)) 
//             }};
// }

// /* Const Views into the data in domain block form */
// template<int dim, int degree, typename Scalar, class Node>
// std::array<const typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector, 2> 
// BaseAlgebra<dim,degree,Scalar,Node>::domain_block_view_const(const MultiVector& A)
// {
//     return {{   * A.offsetView(dof_handler.transpose_domain_map(0), 0),
//                 * A.offsetView(dof_handler.transpose_domain_map(1), dof_handler.transpose_domain_map(0)->getNodeNumElements())   
//             }};
// }

// /* Const Views into the data in fully decomposed block form */
// template<int dim, int degree, typename Scalar, class Node>
// std::array<std::array<const typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector, 2>, 2>
// BaseAlgebra<dim,degree,Scalar,Node>::block_view_const(const MultiVector& A)
// {
//     return
//         {{
//             {  * A.subView(dof_handler.column_range(0))->offsetView(dof_handler.transpose_domain_map(0), 0),
//                 * A.subView(dof_handler.column_range(0))->offsetView(dof_handler.transpose_domain_map(1), dof_handler.transpose_domain_map(0)->getNodeNumElements())   
//             },
//             {  * A.subView(dof_handler.column_range(1))->offsetView(dof_handler.transpose_domain_map(0), 0),
//                 * A.subView(dof_handler.column_range(1))->offsetView(dof_handler.transpose_domain_map(1), dof_handler.transpose_domain_map(0)->getNodeNumElements())   
//             }
//         }};
// }

//  Views into the data in range block form 
// template<int dim, int degree, typename Scalar, class Node>
// std::array<typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector, 2> 
// BaseAlgebra<dim,degree,Scalar,Node>::range_block_view(MultiVector& A)
// {
//     return {{   * A.subViewNonConst(dof_handler.column_range(0)), 
//                 * A.subViewNonConst(dof_handler.column_range(1)) 
//             }};
// }

// /* Views into the data in domain block form */
// template<int dim, int degree, typename Scalar, class Node>
// std::array<typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector, 2> 
// BaseAlgebra<dim,degree,Scalar,Node>::domain_block_view(MultiVector& A)
// {
//     return {{   * A.offsetViewNonConst(dof_handler.transpose_domain_map(0), 0),
//                 * A.offsetViewNonConst(dof_handler.transpose_domain_map(1), dof_handler.transpose_domain_map(0)->getNodeNumElements())   
//             }};
// }

// /* Views into the data in fully decomposed block form */
// template<int dim, int degree, typename Scalar, class Node>
// std::array<std::array<typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector, 2>, 2>
// BaseAlgebra<dim,degree,Scalar,Node>::block_view(MultiVector& A)
// {
//     return
//         {{
//             {  * A.subViewNonConst(dof_handler.column_range(0))->offsetViewNonConst(dof_handler.transpose_domain_map(0), 0),
//                 * A.subViewNonConst(dof_handler.column_range(0))->offsetViewNonConst(dof_handler.transpose_domain_map(1), dof_handler.transpose_domain_map(0)->getNodeNumElements())   
//             },
//             {  * A.subViewNonConst(dof_handler.column_range(1))->offsetViewNonConst(dof_handler.transpose_domain_map(0), 0),
//                 * A.subViewNonConst(dof_handler.column_range(1))->offsetViewNonConst(dof_handler.transpose_domain_map(1), dof_handler.transpose_domain_map(0)->getNodeNumElements())   
//             }
//         }};
// }




/**
 *  Implementation of some classes of operators using the BaseAlgebra data structures:   
 *      Range block operators (like the right product by the Hamiltonian),
 *      Adjoint-like operators (interpolation on the 4 blocks + permutation of orbitals),
 *      Liouvillian operators (combination of the previous ones approximating a commutator).
 */

 template<int dim, int degree, typename Scalar, class Node>
 BaseAlgebra<dim,degree,Scalar,Node>::RangeBlockOp::RangeBlockOp (
        std::array<RCP<Matrix>, 2>& A, 
        std::array<const size_t,2> n_orbitals):
    A(A),
    ColumnRange({{Teuchos::Range1D(0, n_orbitals[0]-1), 
                         Teuchos::Range1D(n_orbitals[0], n_orbitals[0] + n_orbitals[1]-1)}}),
    DomainMap(A[0]->getDomainMap()),
    RangeMap(A[0]->getRangeMap())
{}

 template<int dim, int degree, typename Scalar, class Node>
 void
 BaseAlgebra<dim,degree,Scalar,Node>::RangeBlockOp::apply (
        const MultiVector& X,
        MultiVector& Y, 
        Teuchos::ETransp mode,
        scalar_type alpha,
        scalar_type beta) const
{
    size_t N = ColumnRange[0].size() + ColumnRange[1].size();
    assert(X.getNumVectors() % N == 0);

    for (size_t j = 0; j < X.getNumVectors() / N ; ++j)
    {
        A[0]->apply(* X.subView(ColumnRange[0] + j*N), 
                       * Y.subViewNonConst(ColumnRange[0] + j*N), mode, alpha, beta);
        A[1]->apply(* X.subView(ColumnRange[1] + j*N), 
                       * Y.subViewNonConst(ColumnRange[1] + j*N), mode, alpha, beta);
    }
}

 template<int dim, int degree, typename Scalar, class Node>
 BaseAlgebra<dim,degree,Scalar,Node>::TransposeOp::TransposeOp ( 
        std::array<std::array<RCP<Matrix>, 2>, 2>& A, 
        std::array<size_t,2>                       n_orbitals,
        std::array<double, 2>                      unit_cell_areas,
        RCP<const typename Matrix::map_type>       domain_map, 
        RCP<const typename Matrix::map_type>       range_map):
    A(A),
    nOrbitals(n_orbitals),
    unitCellAreas(unit_cell_areas),
    ColumnRange({{Teuchos::Range1D(0, n_orbitals[0]-1), 
                     Teuchos::Range1D(n_orbitals[0], n_orbitals[0] + n_orbitals[1]-1)}}),
    DomainMap(domain_map),
    RangeMap(range_map)
{
    /* We allocate the helper multivectors */
    helper[0][0] = Tpetra::createMultiVector<scalar_type,local_ordinal_type, global_ordinal_type, node_type>
                            (A[0][0]->getRangeMap(), nOrbitals[0]); 
    helper[0][1] = Tpetra::createMultiVector<scalar_type,local_ordinal_type, global_ordinal_type, node_type>
                            (A[0][1]->getRangeMap(), nOrbitals[0]); 
    helper[1][0] = Tpetra::createMultiVector<scalar_type,local_ordinal_type, global_ordinal_type, node_type>
                            (A[1][0]->getRangeMap(), nOrbitals[1]);
    helper[1][1] = Tpetra::createMultiVector<scalar_type,local_ordinal_type, global_ordinal_type, node_type>
                            (A[1][1]->getRangeMap(), nOrbitals[1]); 
}

 template<int dim, int degree, typename Scalar, class Node>
 std::array<std::array<RCP<const typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector>, 2>, 2>
 BaseAlgebra<dim,degree,Scalar,Node>::TransposeOp::block_view_const(
        const MultiVector& X,
        const size_t j) const
{
    size_t N = nOrbitals[0] + nOrbitals[1];

    std::array<std::array<RCP<const typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector>, 2>, 2>
    blocks =
        {{
            {  X.offsetView(A[0][0]->getDomainMap(), 0)->subView(ColumnRange[0] + j*N),
                X.offsetView(A[0][1]->getDomainMap(), A[0][0]->getDomainMap()->getNodeNumElements()) ->  subView(ColumnRange[0] + j*N)
            },
            {  X.offsetView(A[1][0]->getDomainMap(), 0)->subView(ColumnRange[1] + j*N),
                X.offsetView(A[1][1]->getDomainMap(), A[1][0]->getDomainMap()->getNodeNumElements()) -> subView(ColumnRange[1] + j*N) 
            }
        }};
    return blocks;
}

 template<int dim, int degree, typename Scalar, class Node>
 std::array<std::array<RCP<typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector>, 2>, 2>
 BaseAlgebra<dim,degree,Scalar,Node>::TransposeOp::block_view(
        MultiVector& X,
        const size_t j) const
{
    size_t N = nOrbitals[0] + nOrbitals[1];
    std::array<std::array<RCP<typename BaseAlgebra<dim,degree,Scalar,Node>::MultiVector>, 2>, 2>
    blocks =
        {{
            {  X.offsetViewNonConst(A[0][0]->getDomainMap(), 0)
                    ->subViewNonConst(ColumnRange[0] + j*N),
                X.offsetViewNonConst(A[0][1]->getDomainMap(), A[0][0]
                    ->getDomainMap()->getNodeNumElements()) ->subViewNonConst(ColumnRange[0] + j*N)  
            },
            {  X.offsetViewNonConst(A[1][0]->getDomainMap(), 0)
                    ->subViewNonConst(ColumnRange[1] + j*N),
                X.offsetViewNonConst(A[1][1]->getDomainMap(), A[1][0]->getDomainMap()->getNodeNumElements())  
                    ->subViewNonConst(ColumnRange[1] + j*N) 
            }
        }};
    return blocks;
}




 template<int dim, int degree, typename Scalar, class Node>
 void
 BaseAlgebra<dim,degree,Scalar,Node>::TransposeOp::apply (
        const MultiVector& X,
        MultiVector& Y, 
        Teuchos::ETransp mode,
        scalar_type alpha,
        scalar_type beta) const
{
    /* Create appropriate four block views into the data */
    size_t N = nOrbitals[0] + nOrbitals[1];
    assert(X.getNumVectors() % N == 0);

    for (size_t j = 0; j < X.getNumVectors() / N ; ++j)
    {
        auto X_blocks  = block_view_const(X,j);
        auto Y_blocks  = block_view(Y,j);

        switch (mode)
        {
            case Teuchos::NO_TRANS:
                for (types::block_t range_block = 0; range_block < 2; ++range_block)
                    for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
                    {
                        /* First we apply our interpolant matrix */
                        A.at(range_block).at(domain_block)->apply (
                                    * X_blocks.at(range_block).at(domain_block), 
                                    * helper.at(range_block).at(domain_block), 
                                    mode, alpha);

                        /* Now we perform the 'transpose' of the inner/outer orbital indices and the complex conjugate */
                        typename MultiVector::dual_view_type::t_dev 
                        Y_View = Y_blocks.at(domain_block).at(range_block)->template getLocalView<Kokkos::Serial>();
                        typename MultiVector::dual_view_type::t_dev_const 
                        helperView_const = helper.at(range_block).at(domain_block)->template getLocalView<Kokkos::Serial>();

                        if (beta == Teuchos::ScalarTraits<scalar_type>::zero ())
                            Kokkos::parallel_for (
                                        helper.at(range_block).at(domain_block)->getLocalLength() / nOrbitals[domain_block], 
                                        KOKKOS_LAMBDA (const types::loc_t i) 
                                        {
                                            for (size_t o1 = 0; o1 < nOrbitals[domain_block]; ++o1)
                                                for (size_t o2 = 0; o2 < nOrbitals[range_block]; ++o2)
                                                    Y_View(o2 + nOrbitals[range_block] * i, o1) 
                                                            = helperView_const(o1 + nOrbitals[domain_block] * i, o2);
                                        });
                        else
                            Kokkos::parallel_for (
                                        helper.at(range_block).at(domain_block)->getLocalLength() / nOrbitals[domain_block], 
                                        KOKKOS_LAMBDA (const types::loc_t i) 
                                        {
                                            for (size_t o1 = 0; o1 < nOrbitals[domain_block]; ++o1)
                                                for (size_t o2 = 0; o2 < nOrbitals[range_block]; ++o2)
                                                    Y_View(o2 + nOrbitals[range_block] * i, o1) 
                                                            = helperView_const(o1 + nOrbitals[domain_block] * i, o2)
                                                             + beta * Y_View(o2 + nOrbitals[range_block] * i, o1);
                                        });
                    }
            break;

            case Teuchos::TRANS:
            case Teuchos::CONJ_TRANS:
                for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
                    for (types::block_t range_block = 0; range_block < 2; ++range_block)
                    {
                         /* First we perform the 'transpose' of the inner/outer orbital indices and the complex conjugate */
                        typename MultiVector::dual_view_type::t_dev_const 
                        X_View_const = X_blocks.at(domain_block).at(range_block)->template getLocalView<Kokkos::Serial>();
                        typename MultiVector::dual_view_type::t_dev
                        helperView = helper.at(range_block).at(domain_block)->template getLocalView<Kokkos::Serial>();

                        /* Here we have a rescaling due to the weighted scalar product behind the transpose */
                        Scalar r = unitCellAreas[domain_block] / unitCellAreas[range_block];

                        Kokkos::parallel_for (
                            helper.at(range_block).at(domain_block)->getLocalLength() / nOrbitals[domain_block], 
                            KOKKOS_LAMBDA (const types::loc_t i) 
                            {
                                for (size_t o2 = 0; o2 < nOrbitals[range_block]; ++o2)
                                    for (size_t o1 = 0; o1 < nOrbitals[domain_block]; ++o1)
                                        helperView(o1 + nOrbitals[domain_block] * i, o2) 
                                                = r * X_View_const(o2 + nOrbitals[range_block] * i, o1);
                            });
                        /* Next we apply our interpolant matrix */
                        A.at(range_block).at(domain_block)->apply (
                                    * helper.at(range_block).at(domain_block), 
                                    * Y_blocks.at(range_block).at(domain_block), 
                                    mode, alpha, beta);
                    }
        }
    }
}

template<int dim, int degree, typename Scalar, class Node>
BaseAlgebra<dim,degree,Scalar,Node>::DerivationOp::DerivationOp( 
        RCP<const DoFHandler<dim,degree,Node>>     dofHandler ):
    nodalPositions_ (   "Nodal Positions", 
                dofHandler->n_locally_owned_dofs() ),
    map_(dofHandler->locally_owned_dofs())
{
    nOrbitals_[0] = dofHandler->n_orbitals(0);
    nOrbitals_[1] = dofHandler->n_orbitals(1);
    
    double 
    a0 = dofHandler->unit_cell(0).area, 
    a1 = dofHandler->unit_cell(1).area;
    normalizationFactor_[0] = a1 / (a0 + a1) / dofHandler->n_cell_nodes();
    normalizationFactor_[1] = a0 / (a0 + a1) / dofHandler->n_cell_nodes();

    /* 
     * Fill up the array containing the local nodal positions for each lattice
     *  and unit cell DoF (not including local orbital DoF)
     */

    for (types::loc_t i = 0; i < dofHandler->n_locally_owned_points(0); ++i)
    {   
        /* Block 00 */
            dealii::Point<dim>
        vertexPosition = dofHandler->lattice(0).get_vertex_position(i);
            types::loc_t 
        ndep = dofHandler->n_dofs_each_point(0);

        for (size_t l = 0; l < dim; ++l)
        for (types::loc_t j = 0; j < ndep; ++j)
        {
            types::loc_t idx = i * ndep + j; 
            nodalPositions_ (idx,0,l) = vertexPosition[l];
        }

        /* Block 01 */
        vertexPosition = dofHandler->lattice(1).get_vertex_position(i);

        for (types::loc_t j = 0; j < dofHandler->n_cell_nodes(); ++j)
        {
            dealii::Point<dim>
            nodePosition = vertexPosition + dofHandler->unit_cell(1).get_node_position(i);

            for (size_t l = 0; l < dim; ++l)
            for (size_t k = 0; k < dofHandler->n_orbitals(1); ++k)
            {
                types::loc_t idx = (i * dofHandler->n_cell_nodes() + j) * dofHandler->n_orbitals(1) + k; 
                nodalPositions_ (idx,0,l) = nodePosition[l];
            }
        }
    }

    for (types::loc_t i = 0; i < dofHandler->n_locally_owned_points(0); ++i)
    {   
        /* Block 10 */
            dealii::Point<dim>
        vertexPosition = dofHandler->lattice(0).get_vertex_position(i);

        for (types::loc_t j = 0; j < dofHandler->n_cell_nodes(); ++j)
        {
            dealii::Point<dim>
            nodePosition = vertexPosition + dofHandler->unit_cell(0).get_node_position(i);

            for (size_t l = 0; l < dim; ++l)
            for (size_t k = 0; k < dofHandler->n_orbitals(0); ++k)
            {
                types::loc_t idx = (i * dofHandler->n_cell_nodes() + j) * dofHandler->n_orbitals(0) + k; 
                nodalPositions_ (idx,0,l) = nodePosition[l];
            }
        }

        /* Block 11 */
        vertexPosition = dofHandler->lattice(1).get_vertex_position(i);
            types::loc_t 
        ndep = dofHandler->n_dofs_each_point(1);

        for (size_t l = 0; l < dim; ++l)
        for (types::loc_t j = 0; j < ndep; ++j)
        {
            types::loc_t idx = i * ndep + j; 
            nodalPositions_ (idx,0,l) = vertexPosition[l];
        }
    }
}

 
template<int dim, int degree, typename Scalar, class Node>
std::array<Scalar,dim>
BaseAlgebra<dim,degree,Scalar,Node>::DerivationOp::weightedDot(
                const Vector& X,
                const Vector& Y
                        ) const
{
    std::array<Scalar,dim> local_dot, dot;
    size_t N = nOrbitals_[0] + nOrbitals_[1];
    auto dataX = X.getDualView().d_view;
    auto dataY = Y.getDualView().d_view;

    for (size_t l = 0; l < dim; ++l)
    {
        Scalar 
        s1 = Teuchos::ScalarTraits<Scalar>::zero(), 
        s2 = Teuchos::ScalarTraits<Scalar>::zero();

        for (size_t k = 0; k < nOrbitals_[0]; ++k)
        for (size_t i = 0; i < nodalPositions_.extent(0); ++i)
        {
            s1 += static_cast<Scalar>(nodalPositions_ (i,0,l) 
                                    * Teuchos::ScalarTraits<scalar_type>::conjugate(dataX (i,k) )
                                    * dataY (i,k)) ;
        }

        for (size_t k = nOrbitals_[0]; k < N; ++k)
        for (size_t i = 0; i < nodalPositions_.extent(0); ++i)
        {
            s2 += static_cast<Scalar>(nodalPositions_ (i,1,l) 
                                    * Teuchos::ScalarTraits<scalar_type>::conjugate(dataX (i,k) )
                                    * dataY (i,k)) ;
        }
        local_dot[l] = normalizationFactor_[0] * s1 + normalizationFactor_[1] * s2;
    }

    Teuchos::reduceAll<int,Scalar>(* map_->getComm(), Teuchos::REDUCE_SUM, dim, local_dot.data(), dot.data());
    return dot;
}


template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::DerivationOp::weightedDot(
                const MultiVector& X,
                const MultiVector& Y,
                Teuchos::ArrayView<std::array<Scalar,dim>>& dots
                        ) const
{
    size_t 
    N = nOrbitals_[0] + nOrbitals_[1],
    sizeX = X.getNumVectors(),
    sizeY = Y.getNumVectors();

    assert(sizeX == sizeY);
    assert(sizeX % N == 0);
    assert(sizeX / N == static_cast<size_t>(dots.size()));

    auto dataX = X.getDualView().d_view;
    auto dataY = Y.getDualView().d_view;

    for (size_t col = 0; col < sizeX/N; col++)
    {

        std::array<Scalar,dim> local_dot, dot;
        for (size_t l = 0; l < dim; ++l)
        {
            Scalar 
            s1 = Teuchos::ScalarTraits<Scalar>::zero(), 
            s2 = Teuchos::ScalarTraits<Scalar>::zero();

            for (size_t k = 0; k < nOrbitals_[0]; ++k)
            for (size_t i = 0; i < nodalPositions_.extent(0); ++i)
            {
                s1 += static_cast<Scalar>(nodalPositions_ (i,0,l) 
                            * Teuchos::ScalarTraits<scalar_type>::conjugate(dataX (i,k)) 
                            * dataY (i,k)) ;
            }

            for (size_t k = nOrbitals_[0]; k < nOrbitals_[0]+nOrbitals_[1]; ++k)
            for (size_t i = 0; i < nodalPositions_.extent(0); ++i)
            {
                s2 += static_cast<Scalar>(nodalPositions_ (i,1,l)
                            * Teuchos::ScalarTraits<scalar_type>::conjugate(dataX (i,k))
                            * dataY (i,k)) ;
            }
            local_dot[l] = normalizationFactor_[0] * s1 + normalizationFactor_[1] * s2;
        }
        
        Teuchos::reduceAll<int,Scalar>(* map_->getComm(), Teuchos::REDUCE_SUM, dim, local_dot.data(), dot.data());
        dots[col] = dot;
    }      
}

template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::DerivationOp::apply(
                const MultiVector&  X,
                MultiVector&        Y, 
                Teuchos::ETransp    mode,
                scalar_type         alpha,
                scalar_type         beta
                    ) const
{
    size_t N = nOrbitals_[0] + nOrbitals_[1];
    size_t sizeX = X.getNumVectors();
    size_t sizeY = Y.getNumVectors();

    assert( sizeY == dim * sizeX );
    assert( sizeX % N == 0 );

    auto dataX = X.getDualView().d_view;
    auto dataY = Y.getDualView().d_view;

    for (size_t l = 0; l < dim; l++)
    for (size_t col = 0; col < sizeX/N; col++)
    {
        for (size_t o1 = 0; o1 < nOrbitals_[0]; o1++)
            Kokkos::parallel_for (dataX.extent(0), 
                KOKKOS_LAMBDA(const types::loc_t i) {
                    dataY (i, o1 + col*N + l*sizeX) 
                        = alpha * nodalPositions_ (i,0,l) * dataX (i,o1+col*N) 
                            + beta * dataY (i, o1 + col*N + l*sizeX);
            });
            

        for (size_t o2 = nOrbitals_[0]; o2 < N; o2++)
            Kokkos::parallel_for (dataX.extent(0), 
                KOKKOS_LAMBDA(const types::loc_t i) {
                    dataY (i, o2 + col*N) 
                        = alpha * nodalPositions_ (i,1,l) * dataX (i,o2+col*N) 
                            + beta * dataY (i, o2 + col*N + l*sizeX);
            });
    }
}



template<int dim, int degree, typename Scalar, class Node>
BaseAlgebra<dim,degree,Scalar,Node>::LiouvillianOp::LiouvillianOp (   
        RCP<const TransposeOp>& A, RCP<const RangeBlockOp>& H,
        const scalar_type z, 
        const scalar_type s):
    A(A), H(H), s(s), z(z), DomainMap(A->getDomainMap()), RangeMap(A->getRangeMap())
 {
     T1 = Tpetra::createMultiVector<scalar_type,local_ordinal_type, global_ordinal_type, node_type>(DomainMap, 0);
     T2 = Tpetra::createMultiVector<scalar_type,local_ordinal_type, global_ordinal_type, node_type>(DomainMap, 0);
 }
 
template<int dim, int degree, typename Scalar, class Node>
void
BaseAlgebra<dim,degree,Scalar,Node>::LiouvillianOp::apply (
         const MultiVector& X,
         MultiVector& Y, 
         Teuchos::ETransp mode,
         scalar_type alpha,
         scalar_type beta) const
{
    if (T1.is_null() || T1->getMap() != Y.getMap() || T1->getNumVectors() != Y.getNumVectors())
        * T1 = MultiVector(Y.getMap(), Y.getNumVectors(), false);
    if (T2.is_null() || T2->getMap() != Y.getMap() || T2->getNumVectors() != Y.getNumVectors())
        * T2 = MultiVector(Y.getMap(), Y.getNumVectors(), false);

    if (mode == Teuchos::NO_TRANS)
    {
        A->apply(X, *T1, Teuchos::NO_TRANS);
        H->apply(*T1, *T2, Teuchos::TRANS);
        A->apply(*T2, *T1, Teuchos::TRANS);
        H->apply(X, *T1, Teuchos::NO_TRANS, 
                                -Teuchos::ScalarTraits<scalar_type>::one (), 
                                Teuchos::ScalarTraits<scalar_type>::one ());
        Y.update(alpha*s, *T1, alpha*z, X, beta);
    }
    if (mode == Teuchos::TRANS)
    {
        A->apply(X, *T1, Teuchos::NO_TRANS);
        H->apply(*T1, *T2, Teuchos::NO_TRANS);
        A->apply(*T2, *T1, Teuchos::TRANS);
        H->apply(X, *T1, Teuchos::TRANS, 
                                -Teuchos::ScalarTraits<scalar_type>::one (), 
                                Teuchos::ScalarTraits<scalar_type>::one ());
        Y.update(alpha*s, *T1, alpha*z, X, beta);
    }
    if (mode == Teuchos::CONJ_TRANS)
    {
        A->apply(X, *T1, Teuchos::NO_TRANS);
        H->apply(*T1, *T2, Teuchos::TRANS);
        A->apply(*T2, *T1, Teuchos::TRANS);
        H->apply(X, *T1, Teuchos::NO_TRANS, 
                                -Teuchos::ScalarTraits<scalar_type>::one (), 
                                Teuchos::ScalarTraits<scalar_type>::one ());
        Y.update(alpha*Teuchos::ScalarTraits<scalar_type>::conjugate(s), *T1, 
                 alpha*Teuchos::ScalarTraits<scalar_type>::conjugate(z), X, 
                 beta);
    }
}

/**
 * Explicit instantiations
 */
 template class BaseAlgebra<1,1,double>;
 template class BaseAlgebra<1,2,double>;
 template class BaseAlgebra<1,3,double>;
 template class BaseAlgebra<2,1,double>;
 template class BaseAlgebra<2,2,double>;
 template class BaseAlgebra<2,3,double>;

 template class BaseAlgebra<1,1,std::complex<double> >;
 template class BaseAlgebra<1,2,std::complex<double> >;
 template class BaseAlgebra<1,3,std::complex<double> >;
 template class BaseAlgebra<2,1,std::complex<double> >;
 template class BaseAlgebra<2,2,std::complex<double> >;
 template class BaseAlgebra<2,3,std::complex<double> >;

} /* End namespace Bilayer */
