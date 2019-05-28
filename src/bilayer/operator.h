/* 
* File:   bilayer/operator.h
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on February 24, 2019, 9:00AM
*/


#ifndef moire__bilayer_operator_h
#define moire__bilayer_operator_h

#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <fstream>

#include <complex>
#include "RTOpPack_Types.hpp"

#include <Thyra_TpetraThyraWrappers.hpp>

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
    template <int dim, int degree, typename Scalar, class Node>
    class Operator : public Thyra::LinearOpDefaultBase<Scalar>
    {
    public:

        typedef typename BaseAlgebra<dim,degree,Scalar,Node>::Operator op_type;
        typedef Operator<dim,degree,Scalar,Node> this_t;

        /** Constructors and initializers */
        Operator();


        /** Initialize a space. */
        void initialize(
            const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
            const Teuchos::RCP<op_type> &Op
        );

        void constInitialize(
            const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
            const Teuchos::RCP<const op_type> &Op
        );

        Teuchos::RCP< const Thyra::VectorSpaceBase< Scalar > >  range() const;
        Teuchos::RCP< const Thyra::VectorSpaceBase< Scalar > >  domain() const;

        bool    opSupportedImpl (Thyra::EOpTransp M_trans) const;

        void    
        applyImpl ( const Thyra::EOpTransp M_trans, 
                    const Thyra::MultiVectorBase< Scalar > &X_in, 
                    const Teuchos::Ptr< Thyra::MultiVectorBase< Scalar > > &Y_inout, 
                    const Scalar alpha, const Scalar beta) const;
    private: 

    /* Private data members */
        Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> >
            vectorSpace_;
        Teuchos::ConstNonconstObjectContainer<op_type>
            operator_;

    }; // end class VectorSpace


    /** Nonmember constructor that creates a serial vector space.
    */
    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::LinearOpBase<Scalar>>
    createOperator(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<typename Operator<dim,degree,Scalar,Node>::op_type >& Op
    )
    {
        Teuchos::RCP<Operator<dim,degree,Scalar,Node> > op =
        Teuchos::rcp(new Operator<dim,degree,Scalar,Node>);
        op->initialize(vectorSpace, Op);
        return op;
    }

    /** Nonmember constructor that creates a serial vector space.
    */
    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::LinearOpBase<Scalar> >
    createConstOperator(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<const typename Operator<dim,degree,Scalar,Node>::op_type >& Op
    )
    {
        Teuchos::RCP<Operator<dim,degree,Scalar,Node> > op =
        Teuchos::rcp(new Operator<dim,degree,Scalar,Node>);
        op->constInitialize(vectorSpace, Op);
        return op;
    }

} /* End namespace Bilayer */
#endif
