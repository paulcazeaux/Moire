/* 
* File:   bilayer/base_algebra.h
* Author: Paul Cazeaux
*
* Created on May 12, 2017, 9:00 AM
*/



#ifndef moire__bilayer_base_algebra_h
#define moire__bilayer_base_algebra_h

#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <fstream>

#include <complex>
#include "fftw3.h"

#include <Tpetra_DefaultPlatform.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_CrsGraph_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>

#include "deal.II/base/exceptions.h"
#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"
#include "deal.II/base/conditional_ostream.h"
#include "deal.II/base/timer.h"

#include "tools/types.h"
#include "tools/numbers.h"
#include "bilayer/dof_handler.h"



namespace Bilayer {
    
    /**
    * A class encapsulating the basic operations in a C* algebra equipped with a discretized Hamiltonian.
    * It is intended as a base class creating the necessary operations for use in further computations 
    * (density of states, conductivity, etc.)
    */
    template <int dim, int degree, typename Scalar = std::complex<double> >
    class BaseAlgebra
    {
    public:
        /**
         *  Public typedefs.
         * These types correspond to the main Trilinos containers
         * used in the discretization.
         */
        typedef typename Tpetra::MultiVector<Scalar,types::loc_t, types::glob_t>  MultiVector;
        typedef typename Tpetra::CrsMatrix<Scalar, types::loc_t, types::glob_t>   Matrix;

        /**
         *  Default constructor.
         * Takes in a particular instance of a Multilayer object with
         * the desired geometrical and material parameters.
         *
         * Initializes 
         * - the underling MPI communicator to MPI_COMM_WORLD,
         * - the dof_handler object in charge of managing the interplay
         *     between geometry and degrees of freedom,
         * - raw input/output memory arrays used during the DFT 
         *     computations  with the special aligned allocation 
         *     fftw_malloc as recommended by the FFTW library,
         * - the FFTW plans for computation of said Fourier transforms.
         */
        BaseAlgebra(const Multilayer<dim, 2>& bilayer);

        /**
         *  Default destructor. 
         * Releases the raw memory allocated above for the FFTW arrays
         * as well as the allocated FFTW plans.
         */
        ~BaseAlgebra();

    protected:
        void                base_setup();
        void                assemble_base_matrices();
        void                assemble_hamiltonian_action();
        void                assemble_adjoint_interpolant();


        /* update (wrapper around the multivector update function): A = alpha * A + beta * B. */
        void                linear_combination(const Scalar alpha, const std::array<MultiVector, 2> A,
                                                const Scalar beta, std::array<MultiVector, 2> & B);
        /* Assemble identity observable */
        void                create_identity(std::array<MultiVector, 2>& Id);
        /* Application of the hamiltonian action, representing the right-product in the C* algebra */
        void                hamiltonian_rproduct(const std::array<MultiVector, 2> A, std::array<MultiVector, 2> & B, Scalar scaling = 1.0, Scalar shift = 0.0);
        /* Adjoint operation on an observable */
        void                adjoint(const std::array<MultiVector, 2> A, std::array<MultiVector, 2>& tA);
        /* Trace of an observable, returned on root process (pid = 0, returns 0 on other processes) */
        Scalar              trace(const std::array<MultiVector, 2> A);

        /* MPI communication environment and utilities */
        Teuchos::RCP<const Teuchos::Comm<int> >             mpi_communicator;

        /* DoF Handler object and local indices range */
        DoFHandler<dim,degree>                              dof_handler;

        /* Matrices representing the sparse linear action of the two main operations, acting on blocks */
        std::array<std::array<Teuchos::RCP<Matrix>, 2>, 2>  adjoint_interpolant;
        std::array<Teuchos::RCP<Matrix>, 2>                 hamiltonian_action;

        /* Data structures allocated for additional local computations in the adjoint operation */
        std::array<fftw_plan, 2>                            fplan, bplan;
        std::array<Scalar * , 2>                            data_in, data_out;
        std::array<std::array<MultiVector, 2>, 2>           helper;

        /* Convenience functions */
        const LayerData<dim>&       layer(const int & idx)      const { return dof_handler.layer(idx); }
        const Lattice<dim>&         lattice(const int & idx)    const { return dof_handler.lattice(idx); };
        const UnitCell<dim,degree>& unit_cell(const int & idx)  const { return dof_handler.unit_cell(idx); };
    };



    template<int dim, int degree, typename Scalar>
    BaseAlgebra<dim,degree,Scalar>::BaseAlgebra(const Multilayer<dim, 2>& bilayer)
        :
        mpi_communicator(Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ()),
        dof_handler(bilayer)
    {
        /* Allocate auxiliary arrays for FFT computations */
        data_in[0]   = (Scalar *) fftw_malloc(sizeof(Scalar) * dof_handler.n_dofs_each_point(0,0));
        data_out[0]  = (Scalar *) fftw_malloc(sizeof(Scalar) * dof_handler.n_dofs_each_point(0,0));
        data_in[1]   = (Scalar *) fftw_malloc(sizeof(Scalar) * dof_handler.n_dofs_each_point(1,1));
        data_out[1]  = (Scalar *) fftw_malloc(sizeof(Scalar) * dof_handler.n_dofs_each_point(1,1));
        fplan[0] = fftw_plan_many_dft(dim, unit_cell(1).n_nodes_per_dim.data(), layer(0).n_orbitals, 
                                                reinterpret_cast<fftw_complex *>(data_in[0]),  NULL, layer(0).n_orbitals, 1,
                                                reinterpret_cast<fftw_complex *>(data_out[0]), NULL, layer(0).n_orbitals, 1, FFTW_FORWARD, FFTW_MEASURE);
        bplan[0] = fftw_plan_many_dft(dim, unit_cell(1).n_nodes_per_dim.data(), layer(0).n_orbitals * layer(0).n_orbitals, 
                                                reinterpret_cast<fftw_complex *>(data_out[0]), NULL, layer(0).n_orbitals, 1,
                                                reinterpret_cast<fftw_complex *>(data_in[0]),  NULL, layer(0).n_orbitals, 1, FFTW_BACKWARD, FFTW_MEASURE);
        fplan[1] = fftw_plan_many_dft(dim, unit_cell(0).n_nodes_per_dim.data(), layer(1).n_orbitals * layer(1).n_orbitals, 
                                                reinterpret_cast<fftw_complex *>(data_in[1]),  NULL, layer(1).n_orbitals, 1,
                                                reinterpret_cast<fftw_complex *>(data_out[1]), NULL, layer(1).n_orbitals, 1, FFTW_FORWARD, FFTW_MEASURE);
        bplan[1] = fftw_plan_many_dft(dim, unit_cell(0).n_nodes_per_dim.data(), layer(1).n_orbitals * layer(1).n_orbitals, 
                                                reinterpret_cast<fftw_complex *>(data_out[1]), NULL, layer(1).n_orbitals, 1,
                                                reinterpret_cast<fftw_complex *>(data_in[1]),  NULL, layer(1).n_orbitals, 1, FFTW_BACKWARD, FFTW_MEASURE);
    }

    template<int dim, int degree, typename Scalar>
    BaseAlgebra<dim,degree,Scalar>::~BaseAlgebra()
    {
        fftw_destroy_plan(fplan[0]);
        fftw_destroy_plan(bplan[0]);
        fftw_destroy_plan(fplan[1]);
        fftw_destroy_plan(bplan[1]);

        fftw_free(data_in[0]);
        fftw_free(data_out[0]);
        fftw_free(data_in[1]);
        fftw_free(data_out[1]);
    }


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
            /* If critical, we can avoid using the helper array when the number of orbitals is the same for both layers,
             * At the cost of computing the hermitian transpose in-place
             *
        if !(dof_handler.transpose_domain_map(0,1) == dof_handler.transpose_range_map(1,0) 
            && dof_handler.transpose_domain_map(1,0) == dof_handler.transpose_range_map(0,1))
        {
        }
        */
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
                            
                            if ( arrow_vector.norm() < dof_handler.inter_search_radius)
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
    BaseAlgebra<dim,degree,Scalar>::hamiltonian_rproduct(const std::array<MultiVector, 2>  A, std::array<MultiVector, 2> & B, Scalar scaling, Scalar shift)
    {
        for (types::block_t block = 0; block < 2; ++block)
            hamiltonian_action.at(block)->apply(A.at(block), B.at(block), Teuchos::NO_TRANS, scaling, shift);
    }

    template<int dim, int degree, typename Scalar>
    void
    BaseAlgebra<dim,degree,Scalar>::adjoint(const std::array<MultiVector, 2>  A, std::array<MultiVector, 2> & tA)
    {
        /* first, we decompose A and tA into their four relevant blocks */
        std::array<std::array<Teuchos::RCP<const MultiVector>, 2>, 2> A_blocks;
        std::array<std::array<Teuchos::RCP<MultiVector>, 2>, 2> tA_blocks;

        for (types::block_t range_block = 0; range_block < 2; ++range_block)
        {
            /* Check that the vectors have the same data distribution */
            Assert( A.at(range_block).getMap()->isSameAs(* (dof_handler.locally_owned_dofs(range_block) )), dealii::ExcInternalError() );
            Assert( tA.at(range_block).getMap()->isSameAs(* (dof_handler.locally_owned_dofs(range_block) )), dealii::ExcInternalError() );

            A_blocks.at(range_block).at(0) = A.at(range_block).offsetView(dof_handler.transpose_domain_map(range_block, 0), 0);
            A_blocks.at(range_block).at(1) = A.at(range_block).offsetView(dof_handler.transpose_domain_map(range_block, 1),
                                                                            dof_handler.transpose_domain_map(range_block, 0)->getLocalLength());
            tA_blocks.at(range_block).at(0) = tA.at(range_block).offsetViewNonConst(dof_handler.transpose_domain_map(range_block, 0), 0);
            tA_blocks.at(range_block).at(1) = tA.at(range_block).offsetViewNonConst(dof_handler.transpose_domain_map(range_block, 1),
                                                                            dof_handler.transpose_domain_map(range_block, 0)->getLocalLength());

        }
        /* First we deal with the FFT-based translation inside each unit cell, in the diagonal blocks */
        for (types::block_t b = 0; b < 2; ++b)
        {
            const types::loc_t n_dofs = dof_handler.n_dofs_each_point(b, b);

            adjoint_interpolant.at(b).at(b)->apply (A_blocks.at(b).at(b), helper.at(b).at(b));

            typename MultiVector::dual_view_type
            helperView = helper.at(b).at(b).template getLocalView();
            Kokkos::View<Scalar *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
            fftView (data_in.at(b), n_dofs);

            /* We use the FFT now to translate in the unit cells for each lattice point! */
            for (types::loc_t n=0; n<dof_handler.n_locally_owned_points(b, b); ++n)
            {
                const types::loc_t n_orbitals = dof_handler.n_range_orbitals(b,b);
                const PointData& this_point = dof_handler.locally_owned_point(b, b, n);

                types::loc_t
                start = dof_handler.get_block_dof_index(b, b, this_point.lattice_index, 0, 0),
                end = start + n_dofs;
                dealii::Tensor<1,dim>
                    this_point_position = unit_cell(1-b).inverse_basis * lattice(b).get_vertex_position(this_point.lattice_index);

                for (types::loc_t j=0; j<n_orbitals; ++j)
                {
                    /* Copy the data into FFTW-allocated memory */
                    auto pointView = Kokkos::subview(helperView, std::make_pair(start, end), j);
                    Kokkos::deep_copy(fftView, pointView);

                    /* Forward FFT */
                    fftw_execute(fplan.at(b));

                    /* Phase shift */
                    dealii::Tensor<1,dim> indices;
                    types::loc_t stride = unit_cell(1-b).n_nodes_per_dim[0];
                    for (types::loc_t unrolled_index = 0; unrolled_index < dof_handler.n_cell_nodes(b,b); ++unrolled_index)
                    {
                        indices[0] = unrolled_index % stride;
                        if (dim == 2)
                            indices[1] = unrolled_index / stride;

                        Scalar phase = std::polar(1./(double) n_dofs, 
                                                    -2 * numbers::PI * dealii::scalar_product(this_point_position,  indices));
                        for (types::loc_t j=0; j<n_orbitals; ++j)
                            data_out.at(b)[n_orbitals * unrolled_index + j] *= phase;
                    }

                    /* Backward FFT */
                    fftw_execute(bplan.at(b));
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
                            tA_View(o2 + n_orbitals * i, o1) = Kokkos::conj( helperView_const(o1 + n_orbitals * i, o2) );
                });   

            const types::loc_t
            n_orbitals_1 = dof_handler.n_domain_orbitals(b, 1-b),
            n_orbitals_2 = dof_handler.n_range_orbitals(b, 1-b);
            tA_View = tA_blocks.at(1-b).at(b).template getLocalView<Kokkos::Serial>();
            helperView_const = helper.at(b).at(b-1).getLocalView();

            Kokkos::parallel_for (dof_handler.n_cell_nodes(1-b,b) * dof_handler.n_locally_owned_points(1-b,b), KOKKOS_LAMBDA (const types::loc_t i) {
                    for (types::loc_t o1 = 0; o1 < n_orbitals_1; ++o1)
                        for (types::loc_t o2 = 0; o2 < n_orbitals_2; ++o2)
                            tA_View(o2 + n_orbitals_2 * i, o1) = Kokkos::conj( helperView_const(o1 + n_orbitals_1 * i, o2) );
                });  
                /* Possible code for the case layer(0).n_orbitals == layer(1).n_orbitals : in-place transpose
            if (* dof_handler.transpose_range_map(b,1-b) == * dof_handler.transpose_domain_map(1-b,b)) 
            {
                adjoint_interpolant.at(b).at(1-b)->apply (A_blocks.at(b).at(1-b), tA_blocks.at(1-b).at(b));

                const types::loc_t n_orbitals = layer(b).n_orbitals;
                Kokkos::View<Scalar *, Kokkos::Serial> blockView = tA_blocks.at(1-b).at(b).getLocalView<Kokkos::Serial>();

                Kokkos::parallel_for (dof_handler.n_cell_nodes(1-b,b) * dof_handler.n_locally_owned_points(1-b,b), KOKKOS_LAMBDA (const types::loc_t i) {
                        for (types::loc_t o1 = 0; o1 < n_orbitals; ++o1)
                        {
                            for (types::loc_t o2 = 0; o2 < o1; ++o2)
                            {
                                const Scalar tmp = Kokkos::conj( blockView(o2 + n_orbitals * i, o1) );
                                blockView(o2 + n_orbitals * i, o1) = Kokkos::conj( blockView(o1 + n_orbitals * i, o2) );
                                blockView(o1 + n_orbitals * i, o2) = tmp;
                            }
                            blockView(o1 + n_orbitals * i, o1) = Kokkos::conj( blockView(o1 + n_orbitals * i, o1) );
                        }
                            
                    }); 
            }
            else
            {
                // Case layer(0).n_orbitals != layer(1).n_orbitals : as before

            }*/
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
    Scalar
    BaseAlgebra<dim,degree,Scalar>::trace(const std::array<MultiVector, 2> A)
    {
        std::array<Scalar,2> LocTrace = {{0, 0}};

        for (types::block_t b = 0; b<2; ++b)
        {
            typename MultiVector::dual_view_type::t_dev_const 
            View = A.at(b). template getLocalView<Kokkos::Serial>();

            std::array<types::loc_t, dim> lattice_indices_0;
            for (size_t i=0; i<dim; ++i)
                lattice_indices_0[i] = 0;
            /* Add diagonal values on the current process */
            types::loc_t lattice_index_0 = lattice(b).get_vertex_global_index(lattice_indices_0);
            if (dof_handler.is_locally_owned_point(b,b,lattice_index_0))
            {
                types::loc_t start_zero = dof_handler.locally_owned_dofs(b)->
                                                            getLocalElement( 
                                                            dof_handler.get_dof_index(b, b, lattice_index_0, 0, 0) );
                for (types::loc_t cell_index = 0; cell_index < dof_handler.n_cell_nodes(b,b); ++cell_index)
                    for (types::loc_t orbital = 0; orbital < dof_handler.n_domain_orbitals(b,b); ++orbital)
                        LocTrace.at(b) += Scalar(View(start_zero + cell_index * dof_handler.n_domain_orbitals(b,b) + orbital, orbital));
            }
            LocTrace.at(b) *= unit_cell(1-b).area / static_cast<double>( dof_handler.n_cell_nodes(b,b) );
        }

        Scalar 
        loc_trace = (LocTrace[0] + LocTrace[1]) / (unit_cell(0).area + unit_cell(1).area),
        result = 0.0;
        
        Teuchos::reduce<int, Scalar>(&loc_trace, &result, 1, Teuchos::REDUCE_SUM, 0, * mpi_communicator);
        return result;
    }   

}/* End namespace Bilayer */
#endif
