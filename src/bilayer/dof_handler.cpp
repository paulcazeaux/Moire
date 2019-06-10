/* 
* File:   bilayer/dof_handler.cpp
* Author: Paul Cazeaux
*
* Created on June 30, 2017, 7:00 PM
*/

#include "bilayer/dof_handler.h"

namespace Bilayer {
    template<int dim, int degree, class Node>
    DoFHandler<dim,degree,Node>::DoFHandler(const Multilayer<dim, 2>& bilayer)
        :
        Multilayer<dim, 2>(bilayer)
    {
        n_cell_nodes_ = UnitCell<dim,degree>::compute_n_nodes(bilayer.refinement_level);
        for (size_t i = 0; i<2; ++i)
        {
            lattices_[i]   = std::make_unique<Lattice<dim>>(layer(i).lattice_basis, bilayer.cutoff_radius);
            unit_cells_[i] = std::make_unique<UnitCell<dim,degree>>(layer(i).lattice_basis, bilayer.refinement_level);
            assert(unit_cells_[i]->n_nodes == n_cell_nodes_);
        }
    }


    template<int dim, int degree, class Node>
    void
    DoFHandler<dim,degree,Node>::initialize(Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator)
    {
        this->coarse_setup(mpi_communicator);
        this->distribute_dofs(mpi_communicator);
    }

    template<int dim, int degree, class Node>
    types::MemUsage
    DoFHandler<dim,degree,Node>::memory_consumption() const
    {
        types::MemUsage memory = {0, 0, 0, 0};
        memory.Static += sizeof(this);
        memory.Static += 5 * sizeof(types::loc_t) + sizeof(lattices_) + sizeof(unit_cells_);
        memory.InitArrays += sizeof(partition_indices_) + partition_indices_.capacity() * sizeof(types::subdomain_id);
        memory.Static += sizeof(reordered_indices_) + reordered_indices_.at(0).capacity() * sizeof(types::loc_t)
                                                    + reordered_indices_.at(1).capacity() * sizeof(types::loc_t);
        memory.Static += sizeof(original_indices_) + original_indices_.at(0).capacity() * sizeof(types::loc_t)
                                                    + original_indices_.at(1).capacity() * sizeof(types::loc_t);
        memory.Static += sizeof(locally_owned_points_partition_) + locally_owned_points_partition_.at(0).capacity() * sizeof(types::loc_t)
                                                    + locally_owned_points_partition_.at(1).capacity() * sizeof(types::loc_t);
        memory.Static += sizeof(lattice_point_dof_partition_) + sizeof(types::glob_t) * (lattice_point_dof_partition_[0].capacity()
                                                                                            + lattice_point_dof_partition_[1].capacity());
        typedef std::tuple<types::loc_t, types::block_t, types::block_t, types::loc_t, types::loc_t, std::vector<double>> interp_t;
        memory.InitArrays += sizeof(lattice_points_) + sizeof(PointData) * (lattice_points_[0].capacity()
                                                                        + lattice_points_[1].capacity())
                                + std::accumulate(lattice_points_[0].begin(), lattice_points_[0].end(), 0, 
                                                [] (types::loc_t a, PointData b) {return a + sizeof(interp_t) * b.inter_interpolating_nodes.capacity()
                                                                                           + sizeof(interp_t) * b.intra_interpolating_nodes.capacity(); })
                                + std::accumulate(lattice_points_[1].begin(), lattice_points_[1].end(), 0, 
                                                [] (types::loc_t a, PointData b) {return a + sizeof(interp_t) * b.inter_interpolating_nodes.capacity()
                                                                                           + sizeof(interp_t) * b.intra_interpolating_nodes.capacity(); });
        memory.Static += sizeof(owned_dofs_) + sizeof(transpose_domain_maps_) + sizeof(transpose_range_maps_)
                                + 10 * (sizeof(Teuchos::RCP<const Map>) + sizeof(Map));
        return memory;
    }


    template<int dim, int degree, class Node>
    void
    DoFHandler<dim,degree,Node>::coarse_setup(Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator)
    {
        my_pid = mpi_communicator->getRank ();
        n_procs = mpi_communicator->getSize ();
        partition_indices_.resize(lattice(0).n_vertices + lattice(1).n_vertices);
        if (n_procs > 1)
        {
            if (!my_pid) // Use Parmetis in the future?
                this->make_coarse_partition(partition_indices_);
            /* Broadcast the result of this operation */
            Teuchos::broadcast<int, types::subdomain_id>(* mpi_communicator, 0, partition_indices_.size(), partition_indices_.data());
        }
        else
            std::fill(partition_indices_.begin(), partition_indices_.end(), 0);
        
        /* Compute the reordering of the indices into contiguous range for each processor */
        for (types::block_t domain_block = 0; domain_block < 2; ++domain_block){
            reordered_indices_.at(domain_block).resize(lattice(domain_block).n_vertices);
            original_indices_.at(domain_block).resize(lattice(domain_block).n_vertices);
            locally_owned_points_partition_.at(domain_block).resize(n_procs+1);

            types::loc_t next_free_index = 0;
            for (types::subdomain_id m = 0; m<n_procs; ++m)
            {
                locally_owned_points_partition_.at(domain_block).at(m) = next_free_index;

                types::loc_t offset = (domain_block == 1 ? lattice(0).n_vertices : 0);
                for (types::loc_t i = 0; i<lattice(domain_block).n_vertices; ++i)
                    if (partition_indices_.at(i+offset) == m)
                    {
                        reordered_indices_.at(domain_block).at(i) = next_free_index;
                        original_indices_.at(domain_block).at(next_free_index) = i;
                        ++next_free_index;
                    }
            }

            locally_owned_points_partition_.at(domain_block).at(n_procs) = next_free_index;
            n_locally_owned_points_.at(domain_block) = locally_owned_points_partition_.at(domain_block).at(my_pid+1)
                                                        - locally_owned_points_partition_.at(domain_block).at(my_pid);
        }
    }

    template<int dim, int degree, class Node>
    void
    DoFHandler<dim,degree,Node>::make_coarse_partition(std::vector<types::subdomain_id>& partition_indices)
    {
        /* Produce a sparsity pattern in CSR format and pass it to the METIS partitioner */
        std::vector<idx_t> int_rowstart(1);
            int_rowstart.reserve(lattice(0).n_vertices + lattice(1).n_vertices);
        std::vector<idx_t> int_colnums;
            int_colnums.reserve(50 * (lattice(0).n_vertices + lattice(1).n_vertices));
            /*******************/
            /* Rows in block 0 */
            /*******************/
        for (types::loc_t m = 0; m<lattice(0).n_vertices; ++m)
        {   /**
             * Start with entries corresponding to the right-product by the Hamiltonian.
             */
            Teuchos::Array<types::glob_t> col_indices;
                /* Block 0 <-> 0 */
            std::vector<types::loc_t>   
            neighbors = lattice(0).list_neighborhood_indices(
                                        lattice(0).get_vertex_position(m), layer(0).intra_search_radius);
            for (auto & mm: neighbors)
                col_indices.push_back(mm);

                /* Block 1 -> 0 */
            neighbors = lattice(1).list_neighborhood_indices(
                                        lattice(0).get_vertex_position(m), 
                                        this->inter_search_radius + std::max(unit_cell(0).bounding_radius,
                                                                                unit_cell(1).bounding_radius));
            for (auto & pp: neighbors)
                col_indices.push_back(pp+lattice(0).n_vertices);

            /**
             * Next, we add the terms corresponding to the transpose operation.
             * Within this coarsened framework we neglect boundary terms linking two neighboring cells.
             */

                    /* Block 0 <-> 0 */
            std::array<types::loc_t, dim> on_grid = lattice(0).get_vertex_grid_indices(m);
            for (auto & c : on_grid)
                c = -c;
            col_indices.push_back(lattice(0).get_vertex_global_index(on_grid));

                    /* Block 1 -> 0 */
            neighbors = lattice(1).list_neighborhood_indices(
                                        -lattice(0).get_vertex_position(m), 
                                        unit_cell(0).bounding_radius+unit_cell(1).bounding_radius);
            for (auto & qq: neighbors) 
                col_indices.push_back(qq + lattice(0).n_vertices);

            /* Fill the sparsity pattern row */
            std::sort(col_indices.begin(), col_indices.end());
            for (const auto & col: col_indices)
                int_colnums.push_back(col);
            int_rowstart.push_back(int_colnums.size());
         }

            /*******************/
            /* Rows in block 1 */
            /*******************/
        for (types::loc_t n = 0; n < lattice(1).n_vertices; ++n)
        {   /**
             * Start with entries corresponding to the right-product by the Hamiltonian.
             */
            Teuchos::Array<types::glob_t> col_indices;
                /* Block 0 -> 1 */
            std::vector<types::loc_t>   
            neighbors = lattice(0).list_neighborhood_indices(
                                        lattice(1).get_vertex_position(n), 
                                        this->inter_search_radius + std::max(unit_cell(0).bounding_radius,
                                                                                unit_cell(1).bounding_radius));
            for (auto & qq: neighbors) 
                col_indices.push_back(qq);
                /* Block 1 <-> 1 */
            neighbors = lattice(1).list_neighborhood_indices(
                                        lattice(1).get_vertex_position(n), layer(1).intra_search_radius );
            for (auto & nn: neighbors) 
                col_indices.push_back(nn+lattice(0).n_vertices);

        /**
             * Next, we add the terms corresponding to the transpose operation.
             * Within this coarsened framework we neglect boundary terms linking two neighboring cells.
             */

                    /* Block 0 -> 1 */  
            neighbors = lattice(0).list_neighborhood_indices(
                                        -lattice(1).get_vertex_position(n), 
                                        unit_cell(0).bounding_radius+unit_cell(1).bounding_radius);
            for (auto & qq: neighbors) 
                col_indices.push_back(qq);

                    /* Block 1 <-> 1 */
            std::array<types::loc_t, dim> on_grid = lattice(1).get_vertex_grid_indices(n);
            for (auto & c : on_grid)
                c = -c;
            col_indices.push_back(lattice(1).get_vertex_global_index(on_grid) + lattice(0).n_vertices);

            /* Fill the sparsity pattern row */
            std::sort(col_indices.begin(), col_indices.end());
            for (const auto & col: col_indices)
                int_colnums.push_back(col);
            int_rowstart.push_back(int_colnums.size());
        }

        idx_t
        n = static_cast<idx_t>(lattice(0).n_vertices + lattice(1).n_vertices),
        ncon = 1,
        nparts  = static_cast<idx_t>(n_procs), 
        dummy;                                    

        /* Set Metis options */
        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions (options);
        options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
        options[METIS_OPTION_MINCONN] = 1;

          /* Call Metis */
        std::vector<idx_t> int_partition_indices (lattice(0).n_vertices + lattice(1).n_vertices);
        METIS_PartGraphKway(&n, &ncon, &int_rowstart[0], &int_colnums[0],
                                     nullptr, nullptr, nullptr,
                                     &nparts,nullptr,nullptr,&options[0],
                                     &dummy,&int_partition_indices[0]);

        std::copy (int_partition_indices.begin(),
                   int_partition_indices.end(),
                   partition_indices.begin());
    }

    template<int dim, int degree, class Node>
    void
    DoFHandler<dim,degree,Node>::distribute_dofs(Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator)
    {
        /* Initialize basic #dofs information for each block separately and each lattice point */
        owned_dofs_ = Tpetra::createContigMapWithNode<typename Map::local_ordinal_type, typename Map::global_ordinal_type, Node>(  
                                                            n_dofs_each_point(0) * n_lattice_points(0) + n_dofs_each_point(1) * n_lattice_points(1), 
                                                            n_dofs_each_point(0) * n_locally_owned_points(0) + n_dofs_each_point(1) * n_locally_owned_points(1), 
                                                            mpi_communicator);

        for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
            transpose_domain_maps_.at(domain_block) 
                                = Tpetra::createContigMapWithNode<typename Map::local_ordinal_type, typename Map::global_ordinal_type, Node>(  
                                                        n_dofs_each_point(domain_block) * n_lattice_points(domain_block), 
                                                        n_dofs_each_point(domain_block) * n_locally_owned_points(domain_block),
                                                        mpi_communicator);
        /**
         * Then, each processor assigns degree of freedom ranges for all lattice points
         * (even those it doesn't own)
         */
        lattice_point_dof_partition_.at(0).resize( n_lattice_points(0));
        lattice_point_dof_partition_.at(1).resize( n_lattice_points(1));
        
        types::glob_t next_free_dof = 0;
            /* Process by process, first lattice points of the first layer, then the second layer */
        for (int m = 0; m < n_procs; ++m)
            for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
                for (types::loc_t n = locally_owned_points_partition_.at(domain_block).at(m); 
                                    n < locally_owned_points_partition_.at(domain_block).at(m+1); ++n)
                {
                    lattice_point_dof_partition_.at(domain_block).at(n) = next_free_dof;
                    next_free_dof += n_dofs_each_point(domain_block);
                }
        /** Finally we create the Tpetra range maps for the transpose interpolant. 
         *  The tricky part is that due to the multivector format, domain/range orbitals 
         *  have to be permuted in a second step after interpolation!
         */

        transpose_range_maps_.at(0).at(0) = transpose_domain_maps_.at(0);
        transpose_range_maps_.at(0).at(1) = Tpetra::createContigMapWithNode<typename Map::local_ordinal_type, typename Map::global_ordinal_type, types::DefaultNode>(
                                                                (transpose_domain_maps_.at(0)->getGlobalNumElements () / n_orbitals(0)) * n_orbitals(1),
                                                                (transpose_domain_maps_.at(0)->getNodeNumElements () / n_orbitals(0)) * n_orbitals(1),
                                                                mpi_communicator);
        transpose_range_maps_.at(1).at(0) = Tpetra::createContigMapWithNode<typename Map::local_ordinal_type, typename Map::global_ordinal_type, types::DefaultNode>(
                                                                (transpose_domain_maps_.at(1)->getGlobalNumElements () / n_orbitals(1)) * n_orbitals(0),
                                                                (transpose_domain_maps_.at(1)->getNodeNumElements () / n_orbitals(1)) * n_orbitals(0),
                                                                mpi_communicator);
        transpose_range_maps_.at(1).at(1) = transpose_domain_maps_.at(1);

        /**
         * Now we know which global dof range is owned by each lattice point system-wide. 
         * We explore our local lattice points and fill in the PointData structure
         */
        types::loc_t interp_element_index;
        std::vector<double> interp_weights (Element<dim,degree>::dofs_per_cell);



        for (types::block_t block = 0; block < 2; ++block)
            for (types::loc_t n = locally_owned_points_partition_.at(block).at(my_pid); n < locally_owned_points_partition_.at(block).at(my_pid+1); ++n)
            {
                types::loc_t  lattice_index = original_indices_.at(block).at(n);

                /* Basic info goes into PointData structure */
                    lattice_points_.at(block).emplace_back(block, lattice_index);
                    PointData& this_point = lattice_points_.at(block).back();
                    

                /* Now we find all the interpolating points, for both intra- and inter-layer blocks of the transpose action matrix */

                /** First the intra-layer interpolation.
                 * Spatial DOF: points are exchanged with their opposites x -> -x 
                 */
                std::array<types::loc_t, dim> 
                on_grid = lattice(block).get_vertex_grid_indices(lattice_index);
                for (auto & c : on_grid)
                    c = -c;
                types::loc_t 
                interp_lattice_index = lattice(block).get_vertex_global_index(on_grid);

                /* Configuration space DOF: need to map the translation ω -> ω - x modulo the unit cell */
                this_point.intra_interpolating_nodes .reserve(n_cell_nodes());
                for (types::loc_t cell_index = 0; cell_index < n_cell_nodes(); ++cell_index)
                {
                    dealii::Point<dim> 
                    interpolation_point (  unit_cell(block).get_node_position(cell_index) 
                                        - lattice(block).get_vertex_position(lattice_index)
                                        );

                    std::tie(interp_element_index, interp_weights) = unit_cell(block).interpolate(interpolation_point);
                    this_point.intra_interpolating_nodes.push_back(
                                    std::make_tuple(cell_index, block, block, interp_lattice_index, interp_element_index, interp_weights));
                }

                /* Second the inter-layer interpolation. */
                const types::block_t other_block = (block == 0 ? 1 : 0);

                /* For each node, find the lattice point and corresponding element on the other side */
                this_point.inter_interpolating_nodes .reserve(n_cell_nodes());
                for (types::loc_t cell_index = 0; cell_index < n_cell_nodes(); ++cell_index)
                {
                    const dealii::Point<dim> X = - (lattice(block).get_vertex_position(lattice_index) 
                                                + unit_cell(block).get_node_position(cell_index));
                    const types::loc_t interp_lattice_index = lattice(other_block).round_to_global_index(X);

                    if (interp_lattice_index != types::invalid_local_index)
                    {
                    /* Determine the element on which we will interpolate the value at the current point */
                    std::tie(interp_element_index, interp_weights) = unit_cell(other_block).interpolate(X);
                    this_point.inter_interpolating_nodes.push_back(
                            std::make_tuple(cell_index, block, other_block, interp_lattice_index, interp_element_index, interp_weights));
                    }
                }
            }
    }


    template<int dim, int degree, class Node>
    Teuchos::RCP<const typename DoFHandler<dim,degree,Node>::SparsityPattern>
    DoFHandler<dim,degree,Node>::make_sparsity_pattern_hamiltonian_action(const types::block_t range_block) const
    {
        Teuchos::RCP<SparsityPattern> sparsity_pattern = Tpetra::createCrsGraph(owned_dofs_);
        std::vector<types::glob_t> globalRows;
        std::vector<Teuchos::Array<types::glob_t>> ColIndices;

        /* Each range block has a specific unit cell attached */

        for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
            for (types::loc_t n=0; n < n_locally_owned_points(domain_block); ++n)
            {
                const PointData& this_point = locally_owned_point(domain_block, n);
                assert(this_point.domain_block == domain_block);

                dealii::Point<dim> this_point_position = lattice(domain_block).get_vertex_position(this_point.lattice_index);
                std::array<types::loc_t, dim> 
                this_point_grid_indices = lattice(domain_block).get_vertex_grid_indices(this_point.lattice_index);

                        /* Block b <-> b */
                globalRows.clear();
                for (types::loc_t cell_index = 0; cell_index < n_cell_nodes(); ++cell_index)
                    for (size_t orbital = 0; orbital <  n_orbitals(domain_block); orbital++)
                        globalRows.push_back(get_dof_index(domain_block, this_point.lattice_index, cell_index, orbital));

                ColIndices.resize(globalRows.size());
                for (auto & cols : ColIndices) 
                    cols.clear();

                std::vector<types::loc_t>   
                neighbors = lattice(domain_block).list_neighborhood_indices(this_point_position, layer(domain_block).intra_search_radius);
                for (auto neighbor_lattice_index : neighbors)
                {
                    std::array<types::loc_t,dim> 
                    grid_vector = lattice(domain_block).get_vertex_grid_indices(neighbor_lattice_index);
                    for (size_t j=0; j<dim; ++j)
                        grid_vector[j] = this_point_grid_indices[j] - grid_vector[j];

                    auto it_cols = ColIndices.begin();
                    for (types::loc_t cell_index = 0; cell_index < n_cell_nodes(); ++cell_index)
                        for (size_t orbital = 0; orbital <  n_orbitals(domain_block); orbital++)
                        {
                            for (size_t orbital_middle = 0; orbital_middle <  n_orbitals(domain_block); orbital_middle++)
                                if (this->is_intralayer_term_nonzero(orbital_middle, orbital, grid_vector, domain_block ))
                                    it_cols->push_back(get_dof_index(domain_block, neighbor_lattice_index, cell_index, orbital_middle));
                            it_cols++; 
                        }
                }
                

                        /* Block a -> b, a != b */
                types::block_t other_domain_block = 1-this_point.domain_block;
                neighbors = lattice(other_domain_block).list_neighborhood_indices( this_point_position, 
                                            this->inter_search_radius + unit_cell(1-range_block).bounding_radius);
                        
                for (auto neighbor_lattice_index : neighbors)
                {
                    auto it_cols = ColIndices.begin();
                    for (types::loc_t cell_index = 0; cell_index < n_cell_nodes(); ++cell_index)
                    {
                        /* Note: the unit cell node displacement follows the point which is in the interlayer block */
                        dealii::Tensor<1,dim> arrow_vector = this_point_position 
                                                                + (domain_block != range_block ? 1. : -1.) 
                                                                    * unit_cell(1-range_block).get_node_position(cell_index)
                                                                - lattice(other_domain_block).get_vertex_position(neighbor_lattice_index);
                        
                        for (size_t orbital = 0; orbital <  n_orbitals(domain_block); orbital++)
                            for (size_t orbital_middle = 0; orbital_middle < n_orbitals(other_domain_block); orbital_middle++)
                                if (this->is_interlayer_term_nonzero(orbital_middle, orbital, arrow_vector, other_domain_block, domain_block ))
                                    it_cols[orbital].push_back(get_dof_index(other_domain_block, neighbor_lattice_index, cell_index, orbital_middle));
                        it_cols += n_orbitals(domain_block); 
                    }
                }

                for (size_t i = 0; i < globalRows.size(); ++i)
                    sparsity_pattern->insertGlobalIndices(globalRows.at(i), ColIndices.at(i));
            }
        sparsity_pattern->fillComplete ();
        return sparsity_pattern.getConst ();
    }

    template<int dim, int degree, class Node>
    Teuchos::RCP<const typename DoFHandler<dim,degree,Node>::SparsityPattern>
    DoFHandler<dim,degree,Node>::make_sparsity_pattern_transpose_interpolant(const types::block_t range_block, const types::block_t domain_block) const
    {
        Teuchos::RCP<SparsityPattern> sparsity_pattern = Tpetra::createCrsGraph(transpose_range_maps_.at(range_block).at(domain_block));

        std::vector<types::glob_t> globalRows;
        std::vector<Teuchos::Array<types::glob_t>> ColIndices;

        /* First, the case of diagonal blocks */
        if (range_block == domain_block)
        {
            for (const PointData& this_point : lattice_points_.at(range_block) )
            { 
                assert(this_point.domain_block == domain_block);

                size_t N = n_orbitals(domain_block) * this_point.intra_interpolating_nodes .size();
                globalRows.resize(N);
                ColIndices.resize(N);
                for (auto & cols : ColIndices) 
                    cols.clear();

                auto it_row = globalRows.begin();
                auto it_cols = ColIndices.begin();
                for (const auto & interp_point : this_point.intra_interpolating_nodes)
                {
                    const auto [cell_index, interp_range_block, interp_domain_block, interp_lattice_index, interp_element_index, interp_weights] = interp_point;
                    assert(interp_domain_block == domain_block);
                    assert(interp_range_block == range_block);

                    /* Store row index for each orbital */
                    for (size_t orbital = 0; orbital < n_orbitals(domain_block); ++orbital)
                        it_row[orbital] = get_transpose_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index, orbital);

                    /* Iterate through the nodes of the interpolating element */
                    for (types::loc_t j = 0; j < Element<dim,degree>::dofs_per_cell; ++j)
                    {
                        types::loc_t interp_cell_index = unit_cell(domain_block).subcell_list [interp_element_index].unit_cell_dof_index_map.at(j);
                    
                        /* Store column indices for each orbital */
                        if (unit_cell(domain_block).is_node_interior(interp_cell_index))
                            for (size_t orbital = 0; orbital < n_orbitals(domain_block); orbital++)
                                it_cols[orbital].push_back(get_block_dof_index(domain_block, interp_lattice_index, interp_cell_index, orbital));
                        else // Boundary point!
                        {
                            /* Periodic wrap */
                            auto [offset_interp_cell_index, offset_indices] = unit_cell(domain_block).map_boundary_point_interior(interp_cell_index);
                            for (size_t orbital = 0; orbital < n_orbitals(domain_block); orbital++)
                                it_cols[orbital].push_back(get_block_dof_index(domain_block, interp_lattice_index, offset_interp_cell_index, orbital));
                        }
                    }
                    it_row  += n_orbitals(domain_block);
                    it_cols += n_orbitals(domain_block);
                }
                for (size_t i = 0; i < N; ++i)
                    sparsity_pattern->insertGlobalIndices(globalRows.at(i), ColIndices.at(i));
            }
        }
        /* Now the case of extradiagonal blocks */
        else
        {
            for (const PointData& this_point : lattice_points_.at(range_block) )
            {  
                assert(this_point.domain_block == range_block);

                size_t N = n_orbitals(domain_block) * this_point.inter_interpolating_nodes .size();
                globalRows.resize(N);
                ColIndices.resize(N);
                for (auto & cols : ColIndices) 
                    cols.clear();

                auto it_row = globalRows.begin();
                auto it_cols = ColIndices.begin();
                for (const auto & interp_point : this_point.inter_interpolating_nodes)
                {
                    const auto [cell_index, interp_range_block, interp_domain_block, interp_lattice_index, interp_element_index, interp_weights] = interp_point;
                    assert(interp_domain_block == domain_block);
                    assert(interp_range_block == range_block);

                    /* Store row index for each orbital */
                    for (size_t orbital = 0; orbital < n_orbitals(domain_block); ++orbital)
                        it_row[orbital] = get_transpose_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index, orbital);

                    /* Iterate through the nodes of the interpolating element */
                    for (types::loc_t j = 0; j < Element<dim,degree>::dofs_per_cell; ++j)
                    {
                        types::loc_t interp_cell_index = unit_cell(domain_block).subcell_list [interp_element_index].unit_cell_dof_index_map.at(j);
                        
                        /* Store column indices for each orbital */
                        if (unit_cell(domain_block).is_node_interior(interp_cell_index))
                            for (size_t orbital = 0; orbital < n_orbitals(domain_block); orbital++)
                                it_cols[orbital].push_back(get_block_dof_index(domain_block, interp_lattice_index, interp_cell_index, orbital));
                        else // Boundary point!
                        {
                            auto [offset_interp_cell_index, offset] = unit_cell(domain_block).map_boundary_point_interior(interp_cell_index);
                            const types::loc_t offset_interp_lattice_index = lattice(domain_block).offset_global_index(interp_lattice_index, offset);

                            /* Check that this offset point exists in our cutout */
                            if (offset_interp_lattice_index != types::invalid_local_index)
                                for (size_t orbital = 0; orbital < n_orbitals(domain_block); orbital++)
                                    it_cols[orbital].push_back(get_block_dof_index(domain_block, offset_interp_lattice_index, offset_interp_cell_index, orbital));
                        }
                    }
                    it_row  += n_orbitals(domain_block);
                    it_cols += n_orbitals(domain_block);
                }
                for (size_t i = 0; i < N; ++i)
                    sparsity_pattern->insertGlobalIndices(globalRows.at(i), ColIndices.at(i));
            }
        }
        sparsity_pattern->fillComplete ( transpose_domain_maps_.at(domain_block),
                                        transpose_range_maps_.at(range_block).at(domain_block) );
        return sparsity_pattern.getConst ();
    }

    /* Interface accessors */

    template<int dim, int degree, class Node>
    types::loc_t 
    DoFHandler<dim,degree,Node>::n_lattice_points(const types::block_t block) const
        { return lattice(block).n_vertices; }

    template<int dim, int degree, class Node>
    types::loc_t 
    DoFHandler<dim,degree,Node>::n_locally_owned_points(const types::block_t block) const
        { return n_locally_owned_points_.at(block); }

    template<int dim, int degree, class Node>
    types::loc_t 
    DoFHandler<dim,degree,Node>::n_dofs_each_point(const types::block_t block) const
        { return n_cell_nodes() * layer(block).n_orbitals; }

    template<int dim, int degree, class Node>
    types::loc_t 
    DoFHandler<dim,degree,Node>::n_cell_nodes() const
        { return n_cell_nodes_; }

    template<int dim, int degree, class Node>
    size_t 
    DoFHandler<dim,degree,Node>::n_orbitals(const types::block_t block) const
        { return layer(block).n_orbitals; }

    template<int dim, int degree, class Node>
    Teuchos::Range1D
    DoFHandler<dim,degree,Node>::column_range(  const types::block_t block) const
        {
            switch (block) {
            case 0:
                return Teuchos::Range1D(0, layer(0).n_orbitals-1);
            case 1:
                return Teuchos::Range1D(layer(0).n_orbitals, layer(0).n_orbitals+layer(1).n_orbitals-1);
            default: 
                throw std::logic_error("Looking for block number " + std::to_string(block) + "in Bilayer::DofHandler::column_range!");
            }
        }


    template<int dim, int degree, class Node>
    const PointData &
    DoFHandler<dim,degree,Node>::locally_owned_point(const types::block_t block,
                                                const types::loc_t local_index) const
    {
        /* Bounds checking */
        assert(block < 2);
        assert(local_index < n_locally_owned_points(block));
        return lattice_points_.at(block).at(local_index); 
    }

    template<int dim, int degree, class Node>
    bool
    DoFHandler<dim,degree,Node>::is_locally_owned_point(const types::block_t block,
                                                   const types::loc_t lattice_index) const
    {
        /* Bounds checking */
        assert(block < 2);
        if (lattice_index == types::invalid_local_index)
            return false;
        else {
            assert(lattice_index < n_lattice_points(block) );

            types::loc_t idx = reordered_indices_.at(block).at(lattice_index);

            return (idx >= locally_owned_points_partition_.at(block).at(my_pid) 
                    && idx < locally_owned_points_partition_.at(block).at(my_pid+1));
        }
    }

    template<int dim, int degree, class Node>
    types::subdomain_id
    DoFHandler<dim,degree,Node>::point_owner( const types::block_t block,
                                         const types::loc_t lattice_index) const
    {
        assert(block < 2);
        if (lattice_index == types::invalid_local_index)
            return types::invalid_id;
        else
        {
            assert(lattice_index < n_lattice_points(block) );
            types::loc_t idx = lattice_index + (block == 1 ? lattice(0).n_vertices : 0);
            return partition_indices_.at(idx);
        }
    }

    template<int dim, int degree, class Node>
    std::tuple<types::block_t, types::loc_t, types::loc_t, types::loc_t>
    DoFHandler<dim,degree,Node>::get_dof_info( const types::glob_t block_dof_index) const
    {
        assert( block_dof_index < n_dofs());
        types::block_t block;

        types::glob_t point_index = std::upper_bound(lattice_point_dof_partition_.at(0).begin(), lattice_point_dof_partition_.at(0).end(), block_dof_index) 
                                                - lattice_point_dof_partition_.at(0).begin() - 1;
        if (block_dof_index - lattice_point_dof_partition_.at(0)[point_index] < n_dofs_each_point(0))
        {
            block = 0;
        }
        else
        {
            block = 1;
            point_index = std::upper_bound(lattice_point_dof_partition_.at(1).begin(), lattice_point_dof_partition_.at(1).end(), block_dof_index) - lattice_point_dof_partition_.at(1).begin() - 1;
        }

        assert(block_dof_index >= lattice_point_dof_partition_.at(block)[point_index]);
        assert(block_dof_index - lattice_point_dof_partition_.at(block)[point_index] < n_dofs_each_point(block));

        types::loc_t lattice_index = original_indices_.at(block).at(point_index);
        types::loc_t cell_index = (block_dof_index - lattice_point_dof_partition_.at(block)[point_index]) / n_orbitals(block);
        size_t orbital = (block_dof_index - lattice_point_dof_partition_.at(block)[point_index]) % n_orbitals(block);

        return std::make_tuple(block, lattice_index, cell_index, orbital);
    }


    template<int dim, int degree, class Node>
    std::tuple<types::loc_t, types::loc_t, types::loc_t>
    DoFHandler<dim,degree,Node>::get_dof_info(const types::block_t block, const types::glob_t block_dof_index) const
    {
        assert( block < 2 );
        assert( block_dof_index < n_dofs());

        const std::vector<types::glob_t>& partition = lattice_point_dof_partition_.at(block);
        auto lower = std::upper_bound(partition.begin(), partition.end(), block_dof_index) - 1;

        types::loc_t lattice_index = original_indices_.at(block).at(static_cast<types::loc_t>(lower - partition.begin()));
        types::loc_t cell_index = (block_dof_index - *lower) / n_orbitals(block);
        size_t orbital = (block_dof_index - *lower) % n_orbitals(block);

        return std::make_tuple(lattice_index, cell_index, orbital);
    }

    template<int dim, int degree, class Node>
    types::glob_t
    DoFHandler<dim,degree,Node>::get_dof_index(const types::block_t block,
                                          const types::loc_t lattice_index, const types::loc_t cell_index, 
                                          const size_t orbital) const
    {
        /* Bounds checking */
        assert( block  < 2   );
        assert( lattice_index < n_lattice_points(block)  );
        assert( cell_index    < n_cell_nodes() );
        assert( orbital       < n_orbitals(block) );

        types::loc_t idx = reordered_indices_.at(block).at(lattice_index);

        return lattice_point_dof_partition_.at(block).at(idx) 
                    + cell_index * n_orbitals(block)
                    + orbital;
    }


    template<int dim, int degree, class Node>
    types::glob_t
    DoFHandler<dim,degree,Node>::get_block_dof_index(const types::block_t block,
                                                const types::loc_t lattice_index, const types::loc_t cell_index, 
                                                const size_t orbital) const
    {
        /* Bounds checking */
        assert( block  < 2   );
        assert( lattice_index < n_lattice_points(block)  );
        assert( cell_index    < n_cell_nodes() );
        assert( orbital       < n_orbitals(block) );

        return ( reordered_indices_.at(block).at(lattice_index) * n_cell_nodes()
                    + cell_index) * n_orbitals(block)
                    + orbital;
    }


    template<int dim, int degree, class Node>
    types::glob_t
    DoFHandler<dim,degree,Node>::get_transpose_block_dof_index(const types::block_t range_block,  const types::block_t domain_block,
                                                const types::loc_t lattice_index, const types::loc_t cell_index, 
                                                const size_t orbital) const
    {
        /* Bounds checking */
        assert( range_block   < 2   );
        assert( domain_block  < 2   );
        assert( lattice_index < n_lattice_points(range_block)  );
        assert( cell_index    < n_cell_nodes() );
        assert( orbital       < n_orbitals(domain_block) );

        return ( reordered_indices_.at(range_block).at(lattice_index) * n_cell_nodes()
                    + cell_index) * n_orbitals(domain_block)
                    + orbital;
    }


    template<int dim, int degree, class Node>
    types::glob_t
    DoFHandler<dim,degree,Node>::n_dofs() const
    {
        return owned_dofs_->getGlobalNumElements();
    }


    template<int dim, int degree, class Node>
    types::glob_t
    DoFHandler<dim,degree,Node>::n_locally_owned_dofs()  const
    {
        return owned_dofs_->getNodeNumElements();
    }


    template<int dim, int degree, class Node>
    Teuchos::RCP<const typename DoFHandler<dim,degree,Node>::Map>
    DoFHandler<dim,degree,Node>::locally_owned_dofs() const
    {
        return owned_dofs_;
    }


    template<int dim, int degree, class Node>
    Teuchos::RCP<const typename DoFHandler<dim,degree,Node>::Map>
    DoFHandler<dim,degree,Node>::transpose_domain_map( const types::block_t domain_block ) const
    {
        return transpose_domain_maps_.at(domain_block);
    }


    template<int dim, int degree, class Node>
    Teuchos::RCP<const typename DoFHandler<dim,degree,Node>::Map>
    DoFHandler<dim,degree,Node>::transpose_range_map(  const types::block_t range_block, const types::block_t domain_block  ) const
    {
        return transpose_range_maps_.at(range_block).at(domain_block);
    }


    /**
     * Explicit instantiations
     */
     template class DoFHandler<1,1,types::DefaultNode>;
     template class DoFHandler<1,2,types::DefaultNode>;
     template class DoFHandler<1,3,types::DefaultNode>;
     template class DoFHandler<2,1,types::DefaultNode>;
     template class DoFHandler<2,2,types::DefaultNode>;
     template class DoFHandler<2,3,types::DefaultNode>;

} /* End namespace Bilayer */
