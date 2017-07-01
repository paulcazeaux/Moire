/* 
* File:   bilayer/base_algebra.cpp
* Author: Paul Cazeaux
*
* Created on June 30, 2017, 7:00 PM
*/


#include "bilayer/base_algebra.h"

namespace Bilayer {

    template<int dim, int degree, typename Scalar>
    BaseAlgebra<dim,degree,Scalar>::BaseAlgebra(const Multilayer<dim, 2>& bilayer)
        :
        mpi_communicator(Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ()),
        dof_handler(bilayer),
        torus({{PeriodicTranslationUnit<dim, Scalar>(layer(0).n_orbitals, unit_cell(1).n_nodes_per_dim), 
                PeriodicTranslationUnit<dim, Scalar>(layer(1).n_orbitals, unit_cell(0).n_nodes_per_dim)}})
    {}


    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::base_setup()
    {
        dof_handler.initialize(mpi_communicator);
        Assert(dof_handler.locally_owned_dofs(0)->isContiguous(), dealii::ExcNotImplemented());
        Assert(dof_handler.locally_owned_dofs(1)->isContiguous(), dealii::ExcNotImplemented());
    }


    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::assemble_base_matrices()
    {
        this->assemble_adjoint_interpolant();
        this->assemble_hamiltonian_action();
    }    

    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::assemble_adjoint_interpolant()
    {
        for (types::block_t range_block = 0; range_block < 2; ++range_block)
            for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
            {
                /* First, we initialize the matrix with a static CrsGraph computed by the dof_handler object */
                adjoint_interpolant.at(range_block).at(domain_block) 
                    =  Teuchos::RCP<Matrix>(new Matrix(dof_handler.make_sparsity_pattern_adjoint_interpolant(range_block, domain_block)));

                std::vector<types::glob_t> globalRows;
                std::vector<Teuchos::Array<types::glob_t>> ColIndices;
                std::vector<Teuchos::Array<Scalar>> Values;

                for (types::loc_t n=0; n < dof_handler.n_locally_owned_points(range_block, domain_block); ++n)
                {
                    const PointData& this_point = dof_handler.locally_owned_point(range_block, domain_block, n);

                                            /* Block b <-> b */
                    if (range_block == domain_block)
                    {
                    /* First, the case of diagonal blocks */
                        size_t N = dof_handler.n_dofs_each_point(range_block, domain_block);
                        globalRows.resize(N);
                        ColIndices.resize(N);
                        Values.resize(N);
                        for (auto & cols : ColIndices) 
                            cols.resize(1);
                        for (auto & vals : Values)
                            vals.assign(1, 1.);

                        /* We simply exchange the point with its opposite */
                        std::array<types::loc_t, dim> on_grid = lattice(domain_block).get_vertex_grid_indices(this_point.lattice_index);
                        for (auto & c : on_grid)
                            c = -c;
                        types::loc_t opposite_lattice_index = lattice(domain_block).get_vertex_global_index(on_grid);

                        auto it_row = globalRows.begin();
                        auto it_col = ColIndices.begin();
                        for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(range_block, domain_block); ++cell_index)
                            for (types::loc_t orbital = 0; orbital < dof_handler.n_domain_orbitals(range_block, domain_block); orbital++)
                            {
                                *it_row = dof_handler.get_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index , orbital);
                                (*it_col).assign(1, dof_handler.get_block_dof_index(range_block, domain_block, opposite_lattice_index,   cell_index , orbital));
                                ++it_row;
                                ++it_col;
                            }
                    }
                    else
                    {
                        /* Now the case of extradiagonal blocks : we map the interpolation process */
                        const types::loc_t n_orbitals = dof_handler.n_domain_orbitals(domain_block, range_block);
                        std::vector<double> interpolation_weights;

                        size_t N = dof_handler.n_domain_orbitals(domain_block, range_block) * this_point.interpolated_nodes.size();
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
                        for (const auto & interp_point : this_point.interpolated_nodes)
                        {
                            auto [element_index, interp_range_block, interp_domain_block, interp_lattice_index, interp_cell_index] = interp_point;
                            for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                                it_row[orbital] = dof_handler.get_block_dof_index(interp_range_block, interp_domain_block, interp_lattice_index, interp_cell_index, orbital);

                            dealii::Point<dim> quadrature_point (- (lattice(domain_block).get_vertex_position(this_point.lattice_index)
                                                                    + lattice(range_block).get_vertex_position(interp_lattice_index)
                                                                    + unit_cell(range_block).get_node_position(interp_cell_index)    ));

                            unit_cell(domain_block).subcell_list[element_index].get_interpolation_weights(quadrature_point, interpolation_weights);
                            for (types::loc_t j = 0; j < Element<dim,degree>::dofs_per_cell; ++j)
                            {
                                types::loc_t cell_index = unit_cell(domain_block).subcell_list.at(element_index).unit_cell_dof_index_map.at(j);
                                if (unit_cell(domain_block).is_node_interior(cell_index))
                                {
                                    for (types::loc_t orbital = 0; orbital < n_orbitals; ++orbital)
                                    {
                                        it_cols[orbital].push_back(dof_handler.get_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index , orbital));
                                        it_vals[orbital].push_back(interpolation_weights.at(j));
                                    }
                                }
                                else // Boundary point!
                                {
                                    auto [source_range_block, source_domain_block, source_lattice_index, source_cell_index] 
                                                = this_point.boundary_lattice_points.at(cell_index - unit_cell(domain_block).n_nodes);
                                    /* Last check to see if the boundary point actually exists on the grid. Maybe some extrapolation would be useful? */
                                    if (source_lattice_index != types::invalid_local_index) 
                                        for (types::loc_t orbital = 0; orbital < n_orbitals; orbital++)
                                        {
                                            it_cols[orbital].push_back(dof_handler.get_block_dof_index(range_block, domain_block, source_lattice_index, source_cell_index , orbital));
                                            it_vals[orbital].push_back(interpolation_weights.at(j));
                                        }
                                }
                                it_row += n_orbitals;
                                it_cols += n_orbitals;
                                it_vals += n_orbitals;
                            }
                        }
                    }

                    for (types::loc_t i = 0; i < globalRows.size(); ++i)
                        adjoint_interpolant.at(range_block).at(domain_block)->replaceGlobalValues(globalRows.at(i), ColIndices.at(i), Values.at(i));
                }
                adjoint_interpolant.at(range_block).at(domain_block)->fillComplete ();
            }

        /* Finally, we allocate the helper multivectors */
        helper.at(0).at(0) = MultiVector(dof_handler.transpose_range_map(0,0), dof_handler.n_range_orbitals(0,0));
        helper.at(0).at(1) = MultiVector(dof_handler.transpose_range_map(0,1), dof_handler.n_range_orbitals(0,1));
        helper.at(1).at(0) = MultiVector(dof_handler.transpose_range_map(1,0), dof_handler.n_range_orbitals(1,0));
        helper.at(1).at(1) = MultiVector(dof_handler.transpose_range_map(1,1), dof_handler.n_range_orbitals(1,1));
    }


    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::assemble_hamiltonian_action()
    {
        for (types::block_t range_block = 0; range_block < 2; ++range_block)
        {
            hamiltonian_action.at(range_block) =  Teuchos::RCP<Matrix>(new Matrix(dof_handler.make_sparsity_pattern_hamiltonian_action(range_block)) );

            std::vector<types::glob_t> globalRows;
            std::vector<Teuchos::Array<types::glob_t>> ColIndices;
            std::vector<Teuchos::Array<Scalar>> Values;

            for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
                for (types::loc_t n=0; n < dof_handler.n_locally_owned_points(range_block, domain_block); ++n)
                {
                    const PointData& this_point = dof_handler.locally_owned_point(range_block, domain_block, n);

                    dealii::Point<dim> 
                    this_point_position = lattice(domain_block).get_vertex_position(this_point.lattice_index);
                    std::array<types::loc_t, dim> 
                    this_point_grid_indices = lattice(domain_block).get_vertex_grid_indices(this_point.lattice_index);

                            /* Block b <-> b */
                    globalRows.clear();
                    for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(range_block, domain_block); ++cell_index)
                        for (types::loc_t orbital = 0; orbital <  dof_handler.n_domain_orbitals(range_block, domain_block); orbital++)
                            globalRows.push_back(dof_handler.get_dof_index(range_block, domain_block, this_point.lattice_index, cell_index, orbital));

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
                        grid_vector = lattice(0).get_vertex_grid_indices(neighbor_lattice_index);
                        for (size_t j=0; j<dim; ++j)
                            grid_vector[j] -= this_point_grid_indices[j];

                        auto it_cols = ColIndices.begin();
                        auto it_vals = Values.begin();
                        for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(range_block, domain_block); ++cell_index)
                            for (types::loc_t orbital = 0; orbital <  dof_handler.n_domain_orbitals(range_block, domain_block); orbital++)
                            {
                                for (types::loc_t orbital_middle = 0; orbital_middle <  dof_handler.n_domain_orbitals(range_block, domain_block); orbital_middle++)
                                {
                                    it_cols->push_back(dof_handler.get_dof_index(range_block, domain_block, neighbor_lattice_index, cell_index, orbital_middle));
                                    it_vals->push_back(dof_handler.intralayer_term(orbital_middle, orbital, grid_vector, domain_block ));
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
                        for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(range_block, domain_block); ++cell_index)
                        {
                            /* Note: the unit cell node displacement follows the point which is in the interlayer block */
                            dealii::Tensor<1,dim> arrow_vector = lattice(other_domain_block).get_vertex_position(neighbor_lattice_index)
                                                                + (other_domain_block != range_block ? 1. : -1.) 
                                                                    * unit_cell(1-range_block).get_node_position(cell_index)
                                                                        - this_point_position;
                            
                            for (types::loc_t orbital = 0; orbital <  dof_handler.n_domain_orbitals(range_block, domain_block); orbital++)
                            {
                                for (types::loc_t orbital_middle = 0; orbital_middle < dof_handler.n_domain_orbitals(range_block, other_domain_block); orbital_middle++)
                                {
                                    it_cols[orbital].push_back(dof_handler.get_dof_index(range_block, other_domain_block, neighbor_lattice_index, cell_index, orbital_middle));
                                    it_vals[orbital].push_back(dof_handler.interlayer_term(orbital_middle, orbital, arrow_vector, other_domain_block, domain_block ));
                                }
                            }
                            it_cols += dof_handler.n_domain_orbitals(range_block, other_domain_block); 
                            it_vals += dof_handler.n_domain_orbitals(range_block, other_domain_block);
                        }
                    }

                    for (size_t i = 0; i < globalRows.size(); ++i)
                        hamiltonian_action.at(range_block)->replaceGlobalValues(globalRows.at(i), ColIndices.at(i), Values.at(i));
                }
            hamiltonian_action.at(range_block)->fillComplete ();
        }
    }
    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::hamiltonian_rproduct(const std::array<MultiVector, 2>  A, std::array<MultiVector, 2> & B)
    {
        for (types::block_t block = 0; block < 2; ++block)
            hamiltonian_action.at(block)->apply(A.at(block), B.at(block), Teuchos::NO_TRANS);
    }

    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::adjoint(const std::array<MultiVector, 2>  A, std::array<MultiVector, 2> & tA)
    {
        /* Check that the vectors have the same data distribution */
        for (types::block_t range_block = 0; range_block < 2; ++range_block)
        {
            Assert( A.at(range_block).getMap()->isSameAs(* (dof_handler.locally_owned_dofs(range_block) )), dealii::ExcInternalError() );
            Assert( tA.at(range_block).getMap()->isSameAs(* (dof_handler.locally_owned_dofs(range_block) )), dealii::ExcInternalError() );
        }

        /* first, we decompose A and tA into their four relevant blocks */
        std::array<std::array<const MultiVector, 2>, 2> A_blocks = 
            {{
                {{  * A.at(0).offsetView(dof_handler.transpose_domain_map(0, 0), 0),
                    * A.at(0).offsetView(dof_handler.transpose_domain_map(0, 1),
                                                                            dof_handler.transpose_domain_map(0, 0)->getNodeNumElements())   }},
                {{  * A.at(1).offsetView(dof_handler.transpose_domain_map(1, 0), 0),
                    * A.at(1).offsetView(dof_handler.transpose_domain_map(1, 1),
                                                                            dof_handler.transpose_domain_map(1, 0)->getNodeNumElements())   }}
            }};
        std::array<std::array<MultiVector, 2>, 2> tA_blocks = 
            {{
                {{  * tA.at(0).offsetView(dof_handler.transpose_domain_map(0, 0), 0),
                    * tA.at(0).offsetView(dof_handler.transpose_domain_map(0, 1),
                                                                            dof_handler.transpose_domain_map(0, 0)->getNodeNumElements())   }},
                {{  * tA.at(1).offsetView(dof_handler.transpose_domain_map(1, 0), 0),
                    * tA.at(1).offsetView(dof_handler.transpose_domain_map(1, 1),
                                                                            dof_handler.transpose_domain_map(1, 0)->getNodeNumElements())   }}
            }};



        /* First we deal with the FFT-based translation inside each unit cell, in the diagonal blocks */
        for (types::block_t b = 0; b < 2; ++b)
        {
            const types::loc_t n_dofs = dof_handler.n_dofs_each_point(b, b);

            adjoint_interpolant.at(b).at(b)->apply (A_blocks.at(b).at(b), helper.at(b).at(b));

            typename MultiVector::dual_view_type::t_dev
            helperView = helper.at(b).at(b).template getLocalView<Kokkos::Serial>();
            Kokkos::View<Scalar *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
            fftView = torus.at(b).view ();

            /* We use the FFT now to translate in the unit cells for each lattice point! */
            for (types::loc_t n=0; n<dof_handler.n_locally_owned_points(b, b); ++n)
            {
                const types::loc_t n_orbitals = dof_handler.n_range_orbitals(b,b);
                const PointData& this_point = dof_handler.locally_owned_point(b, b, n);

                types::loc_t
                start = dof_handler.get_block_dof_index(b, b, this_point.lattice_index, 0, 0),
                end = start + n_dofs;
                dealii::Tensor<1,dim>
                    vector = - unit_cell(1-b).inverse_basis * lattice(b).get_vertex_position(this_point.lattice_index);

                for (types::loc_t j=0; j<n_orbitals; ++j)
                {
                    /* Copy the data into FFTW-allocated memory */
                    auto pointView = Kokkos::subview(helperView, std::make_pair(start, end), j);
                    Kokkos::deep_copy(fftView, pointView);
                    /* Forward FFT */
                    torus.at(b).translate (vector);
                    /* Copy back into helper Kokkos array */
                    Kokkos::deep_copy(pointView, fftView);
                }
            }
            /* Next we deal with the extradiagonal block interpolation*/
            adjoint_interpolant.at(b).at(1-b)->apply (A_blocks.at(b).at(1-b), helper.at(b).at(1-b));

            /* Now we perform the 'transpose' of the inner/outer orbital indices and the complex conjugate */
            const types::loc_t
            n_orbitals = dof_handler.n_range_orbitals(b,b);
            typename MultiVector::dual_view_type::t_dev 
            tA_View = tA_blocks.at(b).at(b).template getLocalView<Kokkos::Serial>();
            typename MultiVector::dual_view_type::t_dev_const 
            helperView_const = helperView;

            Kokkos::parallel_for (dof_handler.n_cell_nodes(b,b) * dof_handler.n_locally_owned_points(b,b), KOKKOS_LAMBDA (const types::loc_t i) {
                    for (types::loc_t o1 = 0; o1 < n_orbitals; ++o1)
                        for (types::loc_t o2 = 0; o2 < n_orbitals; ++o2)
                            tA_View(o2 + n_orbitals * i, o1) = numbers::conjugate<Scalar>( helperView_const(o1 + n_orbitals * i, o2) );
                });   

            const types::loc_t
            n_orbitals_1 = dof_handler.n_domain_orbitals(b, 1-b),
            n_orbitals_2 = dof_handler.n_range_orbitals(b, 1-b);
            tA_View = tA_blocks.at(1-b).at(b).template getLocalView<Kokkos::Serial>();
            helperView_const = helper.at(b).at(b-1).template getLocalView<Kokkos::Serial>();

            Kokkos::parallel_for (dof_handler.n_cell_nodes(1-b,b) * dof_handler.n_locally_owned_points(1-b,b), KOKKOS_LAMBDA (const types::loc_t i) {
                    for (types::loc_t o1 = 0; o1 < n_orbitals_1; ++o1)
                        for (types::loc_t o2 = 0; o2 < n_orbitals_2; ++o2)
                            tA_View(o2 + n_orbitals_2 * i, o1) = numbers::conjugate<Scalar>( helperView_const(o1 + n_orbitals_1 * i, o2) );
                });  
        }
    }


    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::linear_combination(const Scalar alpha, const std::array<MultiVector, 2> A,
                                                const Scalar beta, std::array<MultiVector, 2> & B)
    {
        for (types::block_t block = 0; block < 2; ++block)
            B.at(block).update(alpha, A.at(block), beta);
    }

    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::create_identity(std::array<MultiVector, 2>& Id)
    {
        assert(Id.at(0).getMap()->isSameAs(* (dof_handler.locally_owned_dofs(0)) ) 
                    && Id.at(1).getMap()->isSameAs(* (dof_handler.locally_owned_dofs(1)) ) );
        

        std::array<types::loc_t, dim> lattice_indices_0;
        for (size_t i=0; i<dim; ++i)
            lattice_indices_0[i] = 0;
        /* Block 0,0 */
        for (types::block_t b = 0; b < 2; ++b)
        {
            Id.at(b).putScalar(0.);

            types::loc_t lattice_index_0 = lattice(b).get_vertex_global_index(lattice_indices_0);
            if (dof_handler.is_locally_owned_point(b,b,lattice_index_0))
                for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(b,b); ++cell_index)
                    for (types::loc_t orbital = 0; orbital < dof_handler.n_domain_orbitals(b,b); ++orbital)
                        Id.at(b).replaceGlobalValue(dof_handler.get_dof_index(b, b, lattice_index_0, cell_index, orbital), orbital, 1.);
        }
    }


    template<int dim, int degree, typename Scalar>
    std::array<std::vector<Scalar>,2>
    BaseAlgebra<dim,degree,Scalar>::diagonal(const std::array<MultiVector, 2> A)
    {
        std::array<std::vector<Scalar>,2> Diag;

        for (types::block_t b = 0; b<2; ++b)
        {
            Diag.at(b).resize(dof_handler.n_cell_nodes(b,b) * dof_handler.n_domain_orbitals(b,b), 0.0);

            typename MultiVector::dual_view_type::t_dev_const 
            View = A.at(b). template getLocalView<Kokkos::Serial>();

            std::array<types::loc_t, dim> lattice_indices_0;
            for (size_t i=0; i<dim; ++i)
                lattice_indices_0[i] = 0;
            /* Add diagonal values on the current process */
            types::loc_t lattice_index_0 = lattice(b).get_vertex_global_index(lattice_indices_0);

            int origin_owner = dof_handler.point_owner(b, b, lattice_index_0);

            if (dof_handler.my_pid == origin_owner)
            {
                types::loc_t start_zero = dof_handler.locally_owned_dofs(b)->
                                                            getLocalElement( 
                                                            dof_handler.get_dof_index(b, b, lattice_index_0, 0, 0) );
                for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(b,b); ++cell_index)
                    for (types::loc_t orbital = 0; orbital < dof_handler.n_domain_orbitals(b,b); ++orbital)
                    {
                        size_t idx = cell_index * dof_handler.n_domain_orbitals(b,b) + orbital;
                        Diag.at(b).at(idx) = static_cast<Scalar>(View(start_zero + idx, orbital));
                    }
            }
            Teuchos::broadcast<int, Scalar>(* mpi_communicator, origin_owner, Diag.at(b).size(), Diag.at(b).data());
        }
        return Diag;
    }

    template<int dim, int degree, typename Scalar>
    Scalar
    BaseAlgebra<dim,degree,Scalar>::trace(const std::array<MultiVector, 2> A)
    {
        std::array<std::vector<Scalar>,2> Diag = diagonal(A);

        return std::accumulate(Diag.at(0).begin(), Diag.at(0).end(), static_cast<Scalar>(0.0))
                                    * unit_cell(1).area / (unit_cell(0).area + unit_cell(1).area)
                                    / static_cast<double>( dof_handler.n_domain_orbitals(0,0) * dof_handler.n_cell_nodes(0,0) )
                        + std::accumulate(Diag.at(1).begin(), Diag.at(1).end(), static_cast<Scalar>(0.0))
                                    * unit_cell(0).area / (unit_cell(0).area + unit_cell(1).area)
                                    / static_cast<double>( dof_handler.n_domain_orbitals(1,1) * dof_handler.n_cell_nodes(1,1) );
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