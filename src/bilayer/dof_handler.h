/* 
* File:   bilayer/dof_handler.h
* Author: Paul Cazeaux
*
* Created on April 24, 2017, 12:15 PM
*/



#ifndef moire__bilayer_dofhandler_h
#define moire__bilayer_dofhandler_h

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <metis.h>

#include <Teuchos_GlobalMPISession.hpp>
#include <Tpetra_DefaultPlatform.hpp>
#include <Tpetra_Map_decl.hpp>
#include <Tpetra_CrsGraph_decl.hpp>

#include "deal.II/base/exceptions.h"
#include "deal.II/base/point.h"
#include "deal.II/base/tensor.h"

#include "tools/types.h"
#include "parameters/multilayer.h"
#include "geometry/lattice.h"
#include "geometry/unit_cell.h"
#include "bilayer/point_data.h"



namespace Bilayer {

    /**
    * A class encapsulating the discretized underlying bilayer structure:
    * assembly of distributed meshes and lattice points for a bilayer groupoid.
    * It is in particular responsible for building a sparsity pattern 
    * and handling partitioning of the degrees of freedom by metis.
    */
    template <int dim, int degree>
    class DoFHandler : public Multilayer<dim, 2>
    {


    static_assert( (dim == 1 || dim == 2), "UnitCell dimension must be 1 or 2!\n");

    public:
        typedef typename Tpetra::Map<types::loc_t, types::glob_t>       Map;
        typedef typename Tpetra::CrsGraph<types::loc_t, types::glob_t>  SparsityPattern;


        /**
         * Default constructor.
         */
        DoFHandler(const Multilayer<dim, 2>& bilayer);

        /**
         * Initialization routine.
         */
        void               initialize(Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator);

        /**
         *  Determine an estimate of the current memory usage
         * by this class, including most data fields, such 
         * as local point data stored on the local node. 
         * The global usage should then be reduced across 
         * MPI nodes if necessary.
         */
        types::MemUsage    memory_consumption() const;

        /**
         * MPI context information 
         */
        int    my_pid;
        int    n_procs;

        /**
         * Construction of the sparsity patterns for the two main operations: right-multiply by the Hamiltonian, and adjoint.
         * We assume that the dynamic pattern has been initialized to the right size and row index set, obtained from this class
         * either as locally_owned_dofs() which is enough for the right multiply operator, or the larger locally_relevant_dofs()
         * for the adjoint operator, which needs a further communication step to make sure all processors have all the relevant 
         * entries of the sparsity pattern.
         */
        Teuchos::RCP<const SparsityPattern>    make_sparsity_pattern_hamiltonian_action(   const types::block_t range_block) const;
        Teuchos::RCP<const SparsityPattern>    make_sparsity_pattern_adjoint_interpolant(  const types::block_t range_block, 
                                                                                           const types::block_t domain_block) const;
        
        /**
         * Utility functions 
         */
        const LayerData<dim>&          layer(const int & idx)      const { return *(this->layer_data.at(idx)); }
        const Lattice<dim>&            lattice(const int & idx)    const { return *(lattices_.at(idx)); };
        const UnitCell<dim,degree>&    unit_cell(const int & idx)  const { return *(unit_cells_.at(idx)); };

        /**
         * Block_wise information about the number of lattice points, unit cell nodes, domain and range orbitals. 
         */
        types::loc_t    n_lattice_points(       const types::block_t range_block, const types::block_t domain_block) const;
        types::loc_t    n_locally_owned_points( const types::block_t range_block, const types::block_t domain_block) const;
        types::loc_t    n_dofs_each_point(      const types::block_t range_block, const types::block_t domain_block) const;
        types::loc_t    n_cell_nodes(           const types::block_t range_block, const types::block_t domain_block) const;
        types::loc_t    n_domain_orbitals(      const types::block_t range_block, const types::block_t domain_block) const;
        types::loc_t    n_range_orbitals(       const types::block_t range_block, const types::block_t domain_block) const;
  
        /**
         * Accessors to traverse the lattice points level of discretization.
         */
        const PointData &    locally_owned_point(    const types::block_t range_block,  const types::block_t domain_block, 
                                                     const types::loc_t lattice_index) const;
        bool                 is_locally_owned_point( const types::block_t range_block,  const types::block_t domain_block, 
                                                     const types::loc_t lattice_index) const;

        /**
         * Some basic information about the DoF repartition, available after their distribution.
         */
        types::glob_t               n_dofs(               const types::block_t range_block ) const;
        types::glob_t               n_locally_owned_dofs( const types::block_t range_block  )  const;
        Teuchos::RCP<const Map>     locally_owned_dofs(   const types::block_t range_block  ) const;
        Teuchos::RCP<const Map>     transpose_domain_map( const types::block_t range_block, const types::block_t domain_block) const;
        Teuchos::RCP<const Map>     transpose_range_map(  const types::block_t range_block, const types::block_t domain_block) const;


        /**
         * Accessor to go from global degrees of freedom to geometric (domain block, lattice, cell and orbital indices 
         */
        std::tuple<types::loc_t, types::loc_t, types::loc_t>
        get_dof_info(  const types::block_t range_block, const types::block_t domain_block, 
                       const types::glob_t block_dof_index) const;

        /**
         * Accessors to go from geometric indices to global degrees of freedom 
         */
        types::glob_t   get_dof_index(  const types::block_t range_block, const types::block_t domain_block,
                                        const types::loc_t lattice_index, const types::loc_t cell_index, 
                                        const types::loc_t orbital) const;

        types::glob_t   get_block_dof_index(  const types::block_t range_block, const types::block_t domain_block,
                                              const types::loc_t lattice_index, const types::loc_t cell_index, 
                                              const types::loc_t orbital) const;


    private:
        /**
         * The base geometry, available on every processor 
         */
        std::array<std::unique_ptr<Lattice<dim>>, 2>            lattices_;
        std::array<std::unique_ptr<UnitCell<dim,degree>>, 2>    unit_cells_;

        /** 
         * First, we organize our coarse degrees of freedom, i.e. lattice points for both lattices.
         * The dofs corresponding to each lattice point for each term are then
         * organized as two MultiVectors according to the following structure.
         *
         *          < range orbitals of layer 1 >         
         *  [   Lattice 1   (intralayer terms of layer 1)   ]   -> block (0,0)
         *  [   Lattice 2   (interlayer terms 2->1)         ]   -> block (0,1)
         *
         *          < range orbitals of layer 2 >
         *  [   Lattice 1   (interlayer terms 1->2)         ]   -> block (1,0)
         *  [   Lattice 2   (intralayer terms of layer 2)   ]   -> block (1,1)
         *
         */

        /**
         * Total number of lattice points: 
         */
        types::loc_t    n_lattice_points_;

        /**
         * The subdomain ID for each lattice point, determined by the setup() function
         * and known to all processors, in the original numbering 
         */
        std::vector<types::subdomain_id>    partition_indices_;

        /**
         * Re-ordering of corresponding lattice point indices, known to all processors 
         */
        std::array<std::vector<types::loc_t>, 2>    reordered_indices_;
        std::array<std::vector<types::loc_t>, 2>    original_indices_;


        /**
         * Corresponding slice owned by each processor (contiguous in the reordered set of indices) 
         */
        std::array<std::vector<types::loc_t>, 2>    locally_owned_points_partition_;
        std::array<types::loc_t, 2>                 n_locally_owned_points_;

        /**
         * Tool to construct the above global partition and reordering of lattice points 
         */
        void    make_coarse_partition( std::vector<types::subdomain_id>& partition_indices );
        void    coarse_setup( Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator );
        void    distribute_dofs( Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator );

        /**
         *  Now we turn to the matter of enumerating local, fine degrees of freedom,
         *  including cell node and orbital indices, for our array of two MultiVectors.
         *
         *  Total number of degrees of freedom for each point is 
         *                              layer(0).n_orbitals^2               * unit_cell(0).n_nodes      [block 0,0],
         *                  layer(0).n_orbitals   *   layer(1).n_orbitals   * unit_cell(1).n_nodes      [block 0,1];
         *
         *                  layer(1).n_orbitals   *   layer(0).n_orbitals   * unit_cell(0).n_nodes      [block 1,0],
         *                              layer(1).n_orbitals^2               * unit_cell(1).n_nodes      [block 1,1].
         */

        std::array<std::array<std::vector<types::glob_t>, 2>, 2>    lattice_point_dof_partition_;
        std::array<std::array<std::vector<PointData>, 2>, 2>        lattice_points_;

        /**
         * Information on the overall degrees of freedom for each range block, grouping both row blocks 
         */
        std::array<Teuchos::RCP<const Map>, 2>    owned_dofs_;

        /**
         * Information about the column blocks individual data distribution, for the transpose interpolant 
         */
        std::array<std::array<Teuchos::RCP<const Map>, 2>, 2>   transpose_domain_maps_;
        std::array<std::array<Teuchos::RCP<const Map>, 2>, 2>   transpose_range_maps_;
    };




    template<int dim, int degree>
    DoFHandler<dim,degree>::DoFHandler(const Multilayer<dim, 2>& bilayer)
        :
        Multilayer<dim, 2>(bilayer)
    {
        for (size_t i = 0; i<2; ++i)
        {
            lattices_[i]   = std::make_unique<Lattice<dim>>(layer(i).lattice_basis, bilayer.cutoff_radius);
            unit_cells_[i] = std::make_unique<UnitCell<dim,degree>>(layer(i).lattice_basis, bilayer.refinement_level);
        }
        n_lattice_points_ = lattice(0).n_vertices + lattice(1).n_vertices;
    }


    template<int dim, int degree>
    void
    DoFHandler<dim,degree>::initialize(Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator)
    {
        this->coarse_setup(mpi_communicator);
        this->distribute_dofs(mpi_communicator);
    }

    template<int dim, int degree>
    types::MemUsage
    DoFHandler<dim,degree>::memory_consumption() const
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
        memory.Static += sizeof(lattice_point_dof_partition_) + sizeof(types::glob_t) * (lattice_point_dof_partition_[0][0].capacity()
                                                                                            + lattice_point_dof_partition_[1][0].capacity()
                                                                                            + lattice_point_dof_partition_[0][1].capacity()
                                                                                            + lattice_point_dof_partition_[1][1].capacity());
        typedef std::tuple<types::block_t, types::block_t, types::loc_t, types::loc_t> boundary_t;
        typedef std::tuple<types::loc_t, types::block_t, types::block_t, types::loc_t, types::loc_t> interp_t;
        memory.InitArrays += sizeof(lattice_points_) + sizeof(PointData) * (lattice_points_[0][0].capacity()
                                                                        + lattice_points_[1][0].capacity()
                                                                        + lattice_points_[0][1].capacity()
                                                                        + lattice_points_[1][1].capacity())
                                + std::accumulate(lattice_points_[0][0].begin(), lattice_points_[0][0].end(), 0, 
                                                [] (types::loc_t a, PointData b) {return a + sizeof(boundary_t) * b.boundary_lattice_points.capacity() 
                                                                                            + sizeof(interp_t) * b.interpolated_nodes.capacity(); })
                                + std::accumulate(lattice_points_[1][0].begin(), lattice_points_[1][0].end(), 0, 
                                                [] (types::loc_t a, PointData b) {return a + sizeof(boundary_t) * b.boundary_lattice_points.capacity() 
                                                                                            + sizeof(interp_t) * b.interpolated_nodes.capacity(); })
                                + std::accumulate(lattice_points_[0][1].begin(), lattice_points_[0][1].end(), 0, 
                                                [] (types::loc_t a, PointData b) {return a + sizeof(boundary_t) * b.boundary_lattice_points.capacity() 
                                                                                            + sizeof(interp_t) * b.interpolated_nodes.capacity(); })
                                + std::accumulate(lattice_points_[1][1].begin(), lattice_points_[1][1].end(), 0, 
                                                [] (types::loc_t a, PointData b) {return a + sizeof(boundary_t) * b.boundary_lattice_points.capacity() 
                                                                                            + sizeof(interp_t) * b.interpolated_nodes.capacity(); });
        memory.Static += sizeof(owned_dofs_) + sizeof(transpose_domain_maps_) + sizeof(transpose_range_maps_)
                                + 10 * (sizeof(Teuchos::RCP<const Map>) + sizeof(Map));
        return memory;
    }


    template<int dim, int degree>
    void
    DoFHandler<dim,degree>::coarse_setup(Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator)
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

    template<int dim, int degree>
    void
    DoFHandler<dim,degree>::make_coarse_partition(std::vector<types::subdomain_id>& partition_indices)
    {
        /* Produce a sparsity pattern in CSR format and pass it to the METIS partitioner */
        std::vector<idx_t> int_rowstart(1);
            int_rowstart.reserve(lattice(0).n_vertices + lattice(1).n_vertices);
        std::vector<idx_t> int_colnums;
            int_colnums.reserve(50 * (lattice(0).n_vertices + lattice(1).n_vertices));

            /*******************/
            /* Rows in block 0 */
            /*******************/

        Teuchos::Array<types::glob_t> col_indices;
        for (types::loc_t m = 0; m<lattice(0).n_vertices; ++m)
        {   /**
             * Start with entries corresponding to the right-product by the Hamiltonian.
             */
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
             * Next, we add the terms corresponding to the adjoint operation.
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
             * Next, we add the terms corresponding to the adjoint operation.
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
        int ierr = 
        METIS_PartGraphKway(&n, &ncon, &int_rowstart[0], &int_colnums[0],
                                     nullptr, nullptr, nullptr,
                                     &nparts,nullptr,nullptr,&options[0],
                                     &dummy,&int_partition_indices[0]);

        std::copy (int_partition_indices.begin(),
                   int_partition_indices.end(),
                   partition_indices.begin());
    }

    template<int dim, int degree>
    void
    DoFHandler<dim,degree>::distribute_dofs(Teuchos::RCP<const Teuchos::Comm<int> > mpi_communicator)
    {
        /* Initialize basic #dofs information for each block separately and each lattice point */
        for (types::block_t range_block = 0; range_block < 2; ++range_block)
        {
            owned_dofs_.at(range_block) = Tpetra::createContigMap<Map::local_ordinal_type, Map::global_ordinal_type>(  
                                                            n_dofs_each_point(range_block, 0) * n_lattice_points(range_block, 0)
                                                                + n_dofs_each_point(range_block, 1) * n_lattice_points(range_block, 1), 
                                                            n_dofs_each_point(range_block, 0) * n_locally_owned_points(range_block, 0)
                                                                + n_dofs_each_point(range_block, 1) * n_locally_owned_points(range_block, 1), 
                                                            mpi_communicator);

            for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
                transpose_domain_maps_.at(range_block).at(domain_block) 
                                    = Tpetra::createContigMap<Map::local_ordinal_type, Map::global_ordinal_type>(  
                                                            n_dofs_each_point(range_block, domain_block) * n_lattice_points(range_block, domain_block), 
                                                            n_dofs_each_point(range_block, domain_block) * n_locally_owned_points(range_block, domain_block),
                                                            mpi_communicator);
            /**
             * Then, each processor assigns degree of freedom ranges for all lattice points
             * (even those it doesn't own)
             */
            auto & dof_partition = lattice_point_dof_partition_.at(range_block);
            dof_partition.at(0).resize( n_lattice_points(range_block, 0));
            dof_partition.at(1).resize( n_lattice_points(range_block, 1));
            
            types::glob_t next_free_dof = 0;
                /* Process by process, first lattice points of the first layer, then the second layer */
            for (int m = 0; m < n_procs; ++m)
                for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
                    for (types::loc_t n = locally_owned_points_partition_.at(domain_block).at(m); 
                                        n < locally_owned_points_partition_.at(domain_block).at(m+1); ++n)
                    {
                        dof_partition.at(domain_block).at(n) = next_free_dof;
                        next_free_dof += n_dofs_each_point(range_block, domain_block);
                    }
        }
        /** Finally we create the Tpetra range maps for the transpose interpolant. 
         *  The tricky part is that due to the multivector format, domain/range orbitals 
         *  have to be permuted in a second step after interpolation!
         */

        transpose_range_maps_.at(0).at(0) = transpose_domain_maps_.at(0).at(0);
        transpose_range_maps_.at(0).at(1) = Tpetra::createContigMap<Map::local_ordinal_type, Map::global_ordinal_type>(
                                                                (transpose_domain_maps_.at(1).at(0)->getGlobalNumElements () / n_domain_orbitals(1, 0)) * n_range_orbitals(1, 0),
                                                                (transpose_domain_maps_.at(1).at(0)->getNodeNumElements () / n_domain_orbitals(1, 0)) * n_range_orbitals(1, 0),
                                                                mpi_communicator);
        transpose_range_maps_.at(1).at(0) = Tpetra::createContigMap<Map::local_ordinal_type, Map::global_ordinal_type>(
                                                                (transpose_domain_maps_.at(0).at(1)->getGlobalNumElements () / n_domain_orbitals(0, 1)) * n_range_orbitals(0, 1),
                                                                (transpose_domain_maps_.at(0).at(1)->getNodeNumElements () / n_domain_orbitals(0, 1)) * n_range_orbitals(0, 1),
                                                                mpi_communicator);
        transpose_range_maps_.at(1).at(1) = transpose_domain_maps_.at(1).at(1);

        /**
         * Now we know which global dof range is owned by each lattice point system-wide. 
         * We explore our local lattice points and fill in the PointData structure
         */
        for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
            for (types::loc_t n = locally_owned_points_partition_.at(domain_block).at(my_pid); n < locally_owned_points_partition_.at(domain_block).at(my_pid+1); ++n)
            {
                types::loc_t  this_lattice_index = original_indices_.at(domain_block).at(n);

                /* Basic info goes into PointData structure */
                for (types::block_t range_block = 0; range_block<2; ++range_block)
                {
                    lattice_points_.at(range_block).at(domain_block).emplace_back(range_block, domain_block, this_lattice_index);
                    
                    /* Case of inter-layer blocks: boundary points and interpolation points require more data */
                    if (range_block != domain_block)
                    {
                        PointData& this_point = lattice_points_.at(range_block).at(domain_block).back();

                        const auto& this_cell    = unit_cell(1-range_block);
                        const auto& this_lattice = lattice(domain_block);
                        std::array<types::loc_t,dim> 
                        this_vertex_indices = this_lattice.get_vertex_grid_indices(this_lattice_index);

                    /* We now deal with the technicalities of boundary points */
                        this_point.boundary_lattice_points .reserve(this_cell.n_boundary_nodes);

                        for (types::loc_t p = 0; p < this_cell.n_boundary_nodes; ++p)
                        {
                            /* Which cell does this boundary point belong to? */
                            auto [cell_index, lattice_indices] = this_cell.map_boundary_point_interior(p);
                            for (size_t i=0; i<dim; ++i)
                                lattice_indices[i] += this_vertex_indices[i];

                            /* Find out what is the corresponding lattice point index */
                            types::loc_t new_lattice_index = this_lattice.get_vertex_global_index(lattice_indices);
                            /* Check that this point exists in our cutout */
                            if (new_lattice_index != types::invalid_local_index)
                            {
                                /* Add the point to the vector of identified boundary points */
                                this_point.boundary_lattice_points .push_back(
                                    std::make_tuple(range_block, domain_block, new_lattice_index, cell_index));
                            }
                            else // The point is out of bounds
                                this_point.boundary_lattice_points .push_back(
                                    std::make_tuple(range_block, domain_block,
                                    types::invalid_local_index, types::invalid_local_index));
                        }

                    /* We now find and add the interpolation points from the other grid */

                        const auto this_point_position  = this_lattice.get_vertex_position(this_lattice_index);
                        const auto& other_cell          = unit_cell(range_block);
                        const auto& other_lattice       = lattice(range_block);

                        size_t estimate_size = other_cell.n_nodes 
                                    * std::ceil(dealii::determinant(this_cell.basis)/dealii::determinant(other_cell.basis));
                        this_point.interpolated_nodes .reserve(estimate_size);

                        /* Select and iterate through the relevant neighbors of our current point in the other lattice */
                        std::vector<types::loc_t>   
                        neighbors = other_lattice.list_neighborhood_indices( -this_point_position, 
                                            other_cell.bounding_radius+this_cell.bounding_radius);
                        for (types::loc_t lattice_index : neighbors)
                        {
                            const dealii::Point<dim> relative_points_position = - (other_lattice.get_vertex_position(lattice_index) + this_point_position);
                            
                            /* Iterate through the grid points in the corresponding unit cell and test invidually if they are relevant */
                            for (types::loc_t cell_index = 0; cell_index < other_cell.n_nodes; ++cell_index)
                            {
                                types::loc_t element_index = this_cell.find_element( relative_points_position - other_cell.get_node_position(cell_index));
                                /* Test whether the point is in the cell and add to the vector of identified interpolated points */
                                if (element_index != types::invalid_local_index) 
                                    this_point.interpolated_nodes.push_back(
                                            std::make_tuple(element_index, domain_block, range_block, lattice_index, cell_index));
                            }
                        }
                    }
                }
            }
    }


    template<int dim, int degree>
    Teuchos::RCP<const typename DoFHandler<dim,degree>::SparsityPattern>
    DoFHandler<dim,degree>::make_sparsity_pattern_hamiltonian_action(const types::block_t range_block) const
    {
        Teuchos::RCP<SparsityPattern> sparsity_pattern = Tpetra::createCrsGraph(owned_dofs_.at(range_block));
        Teuchos::Array<types::glob_t> ColIndices;

        /* Each range block has a specific unit cell attached */
        const auto& cell = unit_cell(1-range_block);

        for (types::block_t domain_block = 0; domain_block < 2; ++domain_block)
            for (types::loc_t n=0; n < n_locally_owned_points(range_block, domain_block); ++n)
            {
                const PointData& this_point = locally_owned_point(range_block, domain_block, n);
                assert(this_point.range_block == range_block);
                assert(this_point.domain_block == domain_block);

                dealii::Point<dim> this_point_position = lattice(domain_block).get_vertex_position(this_point.lattice_index);

                        /* Block b <-> b */
                std::vector<types::loc_t>   
                neighbors = lattice(domain_block).list_neighborhood_indices(this_point_position, layer(domain_block).intra_search_radius);
                
                for (types::loc_t cell_index = 0; cell_index < n_cell_nodes(range_block, domain_block); ++cell_index)
                {
                    ColIndices.clear();
                    for (auto neighbor_lattice_index : neighbors)
                        for (types::loc_t orbital_middle = 0; orbital_middle <  n_domain_orbitals(range_block, domain_block); orbital_middle++)
                            ColIndices.push_back(
                                get_dof_index(range_block, domain_block, neighbor_lattice_index, cell_index, orbital_middle));
                    for (types::loc_t orbital = 0; orbital <  n_domain_orbitals(range_block, domain_block); orbital++)
                        sparsity_pattern->insertGlobalIndices(
                            get_dof_index(range_block, domain_block, this_point.lattice_index, cell_index, orbital), ColIndices);
                }

                        /* Block a -> b, a != b */
                types::block_t other_domain_block = 1-this_point.domain_block;
                neighbors = lattice(other_domain_block).list_neighborhood_indices( this_point_position, 
                                            this->inter_search_radius + cell.bounding_radius);
                        
                for (auto neighbor_lattice_index : neighbors)
                    for (types::loc_t cell_index = 0; cell_index < n_cell_nodes(range_block, domain_block); ++cell_index)
                    {
                        /* Note: the unit cell node displacement follows the point which is in the interlayer block */
                        dealii::Tensor<1,dim> arrow_vector = lattice(other_domain_block).get_vertex_position(neighbor_lattice_index)
                                                            + (other_domain_block != range_block ? 1. : -1.) 
                                                                * cell.get_node_position(cell_index)
                                                                    - this_point_position;
                        
                        if ( arrow_vector.norm() < this->inter_search_radius)
                        {
                            ColIndices.clear();
                            for (types::loc_t orbital_middle = 0; orbital_middle < n_domain_orbitals(range_block, other_domain_block); orbital_middle++)
                                ColIndices.push_back(
                                    get_dof_index(range_block, other_domain_block, neighbor_lattice_index, cell_index, orbital_middle));
                            for (types::loc_t orbital = 0; orbital < n_domain_orbitals(range_block, domain_block); orbital++)
                                sparsity_pattern->insertGlobalIndices(
                                    get_dof_index(range_block, domain_block, this_point.lattice_index, cell_index, orbital), ColIndices);
                        }
                    }
            }
        sparsity_pattern->fillComplete ();
        return sparsity_pattern.getConst ();
    }

    template<int dim, int degree>
    Teuchos::RCP<const typename DoFHandler<dim,degree>::SparsityPattern>
    DoFHandler<dim,degree>::make_sparsity_pattern_adjoint_interpolant(const types::block_t range_block, const types::block_t domain_block) const
    {
        Teuchos::RCP<SparsityPattern> sparsity_pattern = Tpetra::createCrsGraph(transpose_range_maps_.at(range_block).at(domain_block));

        for (const PointData& this_point : lattice_points_.at(range_block).at(domain_block) )
        {   
            assert(this_point.range_block == range_block);
            assert(this_point.domain_block == domain_block);
            if (range_block == domain_block)
            {
            /* First, the case of diagonal blocks */
                const types::glob_t n_orbitals = n_domain_orbitals(range_block, domain_block); 
                const types::glob_t n_nodes    = n_cell_nodes(range_block, domain_block);
                /* We simply exchange the point with its opposite */
                std::array<types::loc_t, dim> on_grid = lattice(domain_block).get_vertex_grid_indices(this_point.lattice_index);
                for (auto & c : on_grid)
                    c = -c;
                types::loc_t opposite_lattice_index = lattice(domain_block).get_vertex_global_index(on_grid);

                for (types::loc_t cell_index = 0; cell_index < n_nodes; ++cell_index)
                    for (types::loc_t orbital = 0; orbital < n_domain_orbitals(range_block, domain_block); orbital++)
                    {
                        types::glob_t 
                        row = get_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index , orbital),
                        col = get_block_dof_index(range_block, domain_block, opposite_lattice_index,   cell_index , orbital);
                        sparsity_pattern->insertGlobalIndices(row, 1, &col);
                    }
            }
            else
            {
                /* Now the case of extradiagonal blocks : we map the interpolation process */            
                const auto& cell = unit_cell(1-range_block);

                const types::glob_t n_orbitals        = n_domain_orbitals(range_block, domain_block);  
                const types::glob_t interp_n_nodes    = n_cell_nodes(domain_block, range_block);

                for (const auto & interp_point : this_point.interpolated_nodes)
                {
                    auto [element_index, interp_range_block, interp_domain_block, interp_lattice_index, interp_cell_index] = interp_point;
                    assert(interp_range_block == domain_block);
                    assert(interp_domain_block == range_block);

                    for (types::loc_t cell_index : cell.subcell_list [element_index].unit_cell_dof_index_map)
                    {
                        if (cell.is_node_interior(cell_index))
                        {
                            for (types::loc_t orbital = 0; orbital < n_orbitals; orbital++)
                            {
                                types::glob_t 
                                row = orbital + n_orbitals * (interp_cell_index + interp_n_nodes * reordered_indices_.at(interp_domain_block).at(interp_lattice_index) ),
                                col = get_block_dof_index(range_block, domain_block, this_point.lattice_index, cell_index , orbital);
                                sparsity_pattern->insertGlobalIndices(row, 1, &col);
                            }
                        }
                        else // Boundary point!
                        {
                            auto [source_range_block, source_domain_block, source_lattice_index, source_cell_index] 
                                        = this_point.boundary_lattice_points[cell_index - cell.n_nodes];
                            assert(source_range_block == range_block );
                            assert(source_domain_block == domain_block );

                            /* Last check to see if the boundary point actually exists on the grid. Maybe some extrapolation would be useful? */
                            if (source_lattice_index != types::invalid_local_index) 
                                for (types::loc_t orbital = 0; orbital < n_orbitals; orbital++)
                                {
                                    types::glob_t 
                                    row = orbital + n_orbitals * (interp_cell_index + interp_n_nodes * reordered_indices_.at(interp_domain_block).at(interp_lattice_index) ),
                                    col = get_block_dof_index(range_block, domain_block, source_lattice_index, source_cell_index , orbital);
                                    sparsity_pattern->insertGlobalIndices(row, 1, &col);
                                }
                        }
                    }
                }
            }
        }
        sparsity_pattern->fillComplete ( transpose_domain_maps_.at(range_block).at(domain_block),
                                        transpose_range_maps_.at(range_block).at(domain_block) );
        return sparsity_pattern.getConst ();
    }

    /* Interface accessors */

    template<int dim, int degree>
    types::loc_t 
    DoFHandler<dim,degree>::n_lattice_points(const types::block_t range_block, const types::block_t domain_block) const
        { return lattice(domain_block).n_vertices; }

    template<int dim, int degree>
    types::loc_t 
    DoFHandler<dim,degree>::n_locally_owned_points(const types::block_t range_block, const types::block_t domain_block) const
        { return n_locally_owned_points_.at(domain_block); }

    template<int dim, int degree>
    types::loc_t 
    DoFHandler<dim,degree>::n_dofs_each_point(const types::block_t range_block, const types::block_t domain_block) const
    {
        return unit_cell(1-range_block).n_nodes * layer(domain_block).n_orbitals;
    }

    template<int dim, int degree>
    types::loc_t 
    DoFHandler<dim,degree>::n_cell_nodes(const types::block_t range_block, const types::block_t domain_block) const
        { return unit_cell(1-range_block).n_nodes; }

    template<int dim, int degree>
    types::loc_t 
    DoFHandler<dim,degree>::n_domain_orbitals(const types::block_t range_block, const types::block_t domain_block) const
        { return layer(domain_block).n_orbitals; }

    template<int dim, int degree>
    types::loc_t 
    DoFHandler<dim,degree>::n_range_orbitals(const types::block_t range_block, const types::block_t domain_block) const
        { return layer(range_block).n_orbitals; }


    template<int dim, int degree>
    const PointData &
    DoFHandler<dim,degree>::locally_owned_point(const types::block_t range_block, 
                                                const types::block_t domain_block,
                                                const types::loc_t lattice_index) const
    {
        /* Bounds checking */
        assert(range_block < 2);
        assert(domain_block < 2);
        assert(lattice_index < n_lattice_points(range_block, domain_block));

        types::loc_t idx = reordered_indices_.at(domain_block).at(lattice_index);

        assert( (idx >= locally_owned_points_partition_.at(domain_block).at(my_pid) 
                    && idx < locally_owned_points_partition_.at(domain_block).at(my_pid+1)) );
        return lattice_points_.at(range_block).at(domain_block).at(idx - locally_owned_points_partition_.at(domain_block).at(my_pid)); 
    }

    template<int dim, int degree>
    bool
    DoFHandler<dim,degree>::is_locally_owned_point(const types::block_t range_block, 
                                                    const types::block_t domain_block,
                                                   const types::loc_t lattice_index) const
    {
        /* Bounds checking */
        assert(range_block < 2);
        assert(domain_block < 2);
        if (lattice_index == types::invalid_local_index)
            return false;
        else {
            assert(lattice_index < n_lattice_points(range_block, domain_block) );

            types::loc_t idx = reordered_indices_.at(domain_block).at(lattice_index);

            return (idx >= locally_owned_points_partition_.at(domain_block).at(my_pid) 
                    && idx < locally_owned_points_partition_.at(domain_block).at(my_pid+1));
        }
    }


    template<int dim, int degree>
    std::tuple<types::loc_t, types::loc_t, types::loc_t>
    DoFHandler<dim,degree>::get_dof_info(const types::block_t range_block, const types::block_t domain_block, const types::glob_t block_dof_index) const
    {
        assert( range_block  < 2 );
        assert( domain_block < 2 );
        assert( block_dof_index < n_dofs(range_block));

        const std::vector<types::glob_t>& partition = lattice_point_dof_partition_.at(range_block).at(domain_block);
        auto lower = std::lower_bound(partition.begin(), partition.end(), block_dof_index);

        types::loc_t lattice_index = original_indices_.at(domain_block).at(static_cast<types::loc_t>(lower - partition.begin()));
        types::loc_t cell_index = (block_dof_index - *lower) / n_domain_orbitals(range_block, domain_block);
        types::loc_t orbital = (block_dof_index - *lower) % n_domain_orbitals(range_block, domain_block);

        return std::make_tuple(lattice_index, cell_index, orbital);
    }

    template<int dim, int degree>
    types::glob_t
    DoFHandler<dim,degree>::get_dof_index(const types::block_t range_block,  const types::block_t domain_block,
                                          const types::loc_t lattice_index, const types::loc_t cell_index, 
                                          const types::loc_t orbital) const
    {
        /* Bounds checking */
        assert( range_block   < 2   );
        assert( domain_block  < 2   );
        assert( lattice_index < n_lattice_points(range_block, domain_block)  );
        assert( cell_index    < n_cell_nodes(range_block, domain_block) );
        assert( orbital       < n_domain_orbitals(range_block, domain_block) );

        types::loc_t idx = reordered_indices_.at(domain_block).at(lattice_index);

        return lattice_point_dof_partition_.at(range_block).at(domain_block).at(idx) 
                    + cell_index * n_domain_orbitals(range_block, domain_block)
                    + orbital;
    }


    template<int dim, int degree>
    types::glob_t
    DoFHandler<dim,degree>::get_block_dof_index(const types::block_t range_block,  const types::block_t domain_block,
                                                const types::loc_t lattice_index, const types::loc_t cell_index, 
                                                const types::loc_t orbital) const
    {
        /* Bounds checking */
        assert( range_block   < 2   );
        assert( domain_block  < 2   );
        assert( lattice_index < n_lattice_points(range_block, domain_block)  );
        assert( cell_index    < n_cell_nodes(range_block, domain_block) );
        assert( orbital       < n_domain_orbitals(range_block, domain_block) );

        types::loc_t idx = reordered_indices_.at(domain_block).at(lattice_index);

        return ( reordered_indices_.at(domain_block).at(lattice_index) * n_dofs_each_point(range_block, domain_block)
                    + cell_index) * n_domain_orbitals(range_block, domain_block)
                    + orbital;
    }


    template<int dim, int degree>
    types::glob_t
    DoFHandler<dim,degree>::n_dofs(  const types::block_t range_block   ) const
    {
        return owned_dofs_.at(range_block)->getGlobalNumElements();
    }


    template<int dim, int degree>
    types::glob_t
    DoFHandler<dim,degree>::n_locally_owned_dofs(  const types::block_t range_block   )  const
    {
        return owned_dofs_.at(range_block)->getNodeNumElements();
    }


    template<int dim, int degree>
    Teuchos::RCP<const typename DoFHandler<dim,degree>::Map>
    DoFHandler<dim,degree>::locally_owned_dofs(  const types::block_t range_block   ) const
    {
        return owned_dofs_.at(range_block);
    }


    template<int dim, int degree>
    Teuchos::RCP<const typename DoFHandler<dim,degree>::Map>
    DoFHandler<dim,degree>::transpose_domain_map(  const types::block_t range_block, const types::block_t domain_block  ) const
    {
        return transpose_domain_maps_.at(range_block).at(domain_block);
    }


    template<int dim, int degree>
    Teuchos::RCP<const typename DoFHandler<dim,degree>::Map>
    DoFHandler<dim,degree>::transpose_range_map(  const types::block_t range_block, const types::block_t domain_block  ) const
    {
        return transpose_range_maps_.at(range_block).at(domain_block);
    }


} /* End namespace Bilayer */

#endif
