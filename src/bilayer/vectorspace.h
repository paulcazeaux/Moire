/* 
* File:   bilayer/vectorspace.h
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on July 31, 2018, 10:00 PM
*/


#ifndef moire__bilayer_vector_space_h
#define moire__bilayer_vector_space_h

#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <fstream>

#include <complex>
#include "RTOpPack_Types.hpp"

#include <Thyra_TpetraThyraWrappers.hpp>

#include <Tpetra_DefaultPlatform.hpp>
#include <Kokkos_Core.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Range1D.hpp>
#include <Teuchos_ConstNonconstObjectContainer.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_Operator.hpp>


#include "tools/types.h"
#include "tools/numbers.h"
#include "bilayer/dof_handler.h"
#include "bilayer/vector.h"
#include "bilayer/multivector.h"
#include "bilayer/scalarprod.h"



namespace Bilayer {
    using Thyra::EConj;
    using Thyra::EViewType;
    using Thyra::EStrideType;
    using Thyra::EOpTransp;


    /** Concrete implementation of an SPMD vector space for Bilayers.
    */
    template <int dim, int degree, typename Scalar, class Node = Kokkos::Compat::KokkosSerialWrapperNode>
    class VectorSpace : public Thyra::SpmdVectorSpaceDefaultBase<Scalar>
    {
    public:

        typedef typename BaseAlgebra<dim,degree,Scalar,Node>::Vector::map_type map_type;
        typedef VectorSpace<dim,degree,Scalar,Node> this_t;

        /** Constructors and initializers */

        /** Create with weak ownership to self. */
        static Teuchos::RCP<VectorSpace<dim,degree,Scalar,Node> > create();

        /** Initialize a space. */
        void initialize(
            const Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> >& baseAlgebra
        );


        /** Public overridden from Thyra::VectorSpaceBase */
        /** Returns true if all the elements in range are in this process.
        */
        bool hasInCoreView(
            const Teuchos::Range1D& range, const EViewType viewType, const EStrideType strideType
        ) const;
        Teuchos::RCP< const Thyra::VectorSpaceBase<Scalar> > clone() const;

    protected:

    /** Protected overridden from Thyra::VectorSpaceBase */

        Teuchos::RCP<Thyra::VectorBase<Scalar> >          createMember() const;
        Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >     createMembers(int numMembers) const;


    public:

    /** Public overridden from SpmdVectorSpaceDefaultBase */

        Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal> > 
            getComm() const;
        Teuchos::Ordinal 
            localSubDim() const;
        Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> > 
            getAlgebra() const;
        Teuchos::RCP<const map_type>
            getOrbVecMap() const;
        Teuchos::RCP<const map_type>
            getVecMap() const;
        Teuchos::RCP<const Thyra::TpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node> >
            getOrbTpetraVectorSpace() const;
        Teuchos::RCP<const Thyra::TpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node> >
            getTpetraVectorSpace() const;
        size_t
            getNumOrbitals() const;

        Teuchos::RCP<const Thyra::TpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node> >
            createDomainVectorSpace(size_t numElements) const;


    private: 

    /* Private data members */
        Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> > 
            baseAlgebra_;
        Teuchos::RCP<const map_type>        
            orbVecMap_, vecMap_;
        Teuchos::RCP<const Thyra::TpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node> >
            orbTpetraVectorSpace_, tpetraVectorSpace_;
        Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal> >     
            comm_;
        Teuchos::RCP<this_t>                                     
            weakSelfPtr_;

    /* Private member functions */

        VectorSpace();

    }; // end class VectorSpace


    /** Nonmember constructor that creates a serial vector space.
    */
    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<VectorSpace<dim,degree,Scalar,Node> >
    vectorSpace(
        const Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> >& baseAlgebra
    )
    {
        Teuchos::RCP<VectorSpace<dim,degree,Scalar,Node> > vs =
        VectorSpace<dim,degree,Scalar,Node>::create();
        vs->initialize(baseAlgebra);
        return vs;
    }

} /* End namespace Bilayer */
#endif
