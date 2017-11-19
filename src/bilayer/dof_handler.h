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
        typedef typename Tpetra::Map<types::loc_t, types::glob_t, Kokkos::Compat::KokkosSerialWrapperNode>       Map;
        typedef typename Tpetra::CrsGraph<types::loc_t, types::glob_t, Kokkos::Compat::KokkosSerialWrapperNode>  SparsityPattern;


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


        bool is_locally_owned_point( const types::block_t range_block,  const types::block_t domain_block, 
                                                     const types::loc_t lattice_index) const;
        types::subdomain_id point_owner( const types::block_t range_block,  const types::block_t domain_block, 
                                                     const types::loc_t lattice_index) const;
  
        /**
         * Accessors to traverse the lattice points level of discretization.
         */
        const PointData &    locally_owned_point(    const types::block_t range_block,  const types::block_t domain_block, 
                                                     const types::loc_t local_index) const;

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
        std::tuple<types::block_t, types::loc_t, types::loc_t, types::loc_t>
        get_dof_info(  const types::block_t range_block, 
                       const types::glob_t block_dof_index) const;


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


        types::glob_t   get_transpose_block_dof_index(  const types::block_t range_block, const types::block_t domain_block,
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

} /* End namespace Bilayer */

#endif
