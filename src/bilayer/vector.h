/* 
* File:   bilayer/vector.h
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on July 31, 2018, 10:00 PM
*/


#ifndef moire__bilayer_vector_h
#define moire__bilayer_vector_h

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
#include <Teuchos_PtrDecl.hpp>
#include <Teuchos_Range1D.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_ConstNonconstObjectContainer.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_Operator.hpp>


#include "tools/types.h"
#include "tools/numbers.h"
#include "bilayer/dof_handler.h"
#include "bilayer/multivector.h"
#include "bilayer/vectorspace.h"
#include "bilayer/scalarprod.h"



namespace Bilayer {
    using Thyra::EConj;
    using Thyra::EViewType;
    using Thyra::EStrideType;
    using Thyra::EOpTransp;

    template <int dim, int degree, typename Scalar, class Node> class VectorSpace;

    template <int dim, int degree, typename Scalar, class Node>
    class Vector : virtual public Thyra::SpmdVectorDefaultBase<Scalar>
    {
    public:
    /** Public types for Tpetra vectors, multivectors and their Thyra adapters*/
        typedef typename Tpetra::Vector<Scalar,types::loc_t,types::glob_t,Node> TV;
        typedef typename Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> TMV;
        typedef typename Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> tTV;
        typedef typename Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> tTMV;


    /** Constructors/initializers */

        /** Construct to uninitialized. */
        Vector();

        /** Initialize. */
        void initialize(
            const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
            const Teuchos::RCP<TMV>  &orbVector
        );

        /** Initialize. */
        void constInitialize(
            const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
            const Teuchos::RCP<const TMV> &orbVector
        );

        /** Get the embedded non-const Thyra::TpetraVector. */
        Teuchos::RCP<tTV> getThyraVector();

        /** Get the embedded const Thyra::TpetraVector. */
        Teuchos::RCP<const tTV> getConstThyraVector() const;

        /** Get the embedded non-const Thyra::TpetraMultiVector. */
        Teuchos::RCP<tTMV> getThyraOrbVector();

        /** Get the embedded const Thyra::TpetraMultiVector. */
        Teuchos::RCP<const tTMV> getConstThyraOrbVector() const;

        /** Overridden from Thyra::VectorDefaultBase */
        Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > domain() const;

        // Should these Impl functions also be protected???

        /** Overridden from Thyra::SpmdMultiVectorBase */
        Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<Scalar> > spmdSpaceImpl() const;

        /** Overridden from Thyra::SpmdVectorBase */
        void getNonconstLocalVectorDataImpl(const Teuchos::Ptr<Teuchos::ArrayRCP<Scalar> > &localValues);
        void getLocalVectorDataImpl(const Teuchos::Ptr<Teuchos::ArrayRCP<const Scalar> > &localValues) const;

    protected:

        /** Overridden protected functions from VectorBase */
        virtual void randomizeImpl(Scalar l, Scalar u);
        virtual void absImpl(const Thyra::VectorBase<Scalar>& x);
        virtual void reciprocalImpl(const Thyra::VectorBase<Scalar>& x);
        virtual void eleWiseScaleImpl(const Thyra::VectorBase<Scalar>& x);

        virtual typename Teuchos::ScalarTraits<Scalar>::magnitudeType norm2WeightedImpl(const Thyra::VectorBase<Scalar>& x) const;

        virtual void applyOpImpl(
            const RTOpPack::RTOpT<Scalar> &op,
            const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::VectorBase<Scalar> > > &vecs,
            const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::VectorBase<Scalar> > > &targ_vecs,
            const Teuchos::Ptr<RTOpPack::ReductTarget> &reduct_obj,
            const Teuchos::Ordinal global_offset
        ) const;

        void acquireDetachedVectorViewImpl(
            const Teuchos::Range1D& rng,
            RTOpPack::ConstSubVectorView<Scalar>* sub_vec
        ) const;

        void acquireNonconstDetachedVectorViewImpl(
            const Teuchos::Range1D& rng,
            RTOpPack::SubVectorView<Scalar>* sub_vec
        );

        void commitNonconstDetachedVectorViewImpl(
            RTOpPack::SubVectorView<Scalar>* sub_vec
        );

        /** Overridden protected functions from MultiVectorBase */

        virtual void assignImpl(Scalar alpha);
        virtual void assignMultiVecImpl(const Thyra::MultiVectorBase<Scalar>& mv);
        virtual void scaleImpl(Scalar alpha);
        virtual void updateImpl(
            Scalar alpha,
            const Thyra::MultiVectorBase<Scalar>& mv
        );

        virtual void linearCombinationImpl(
            const Teuchos::ArrayView<const Scalar>& alpha,
            const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > >& mv,
            const Scalar& beta
        );

        virtual void dotsImpl(
            const Thyra::MultiVectorBase<Scalar>& mv,
            const Teuchos::ArrayView<Scalar>& prods
        ) const;

        virtual void norms1Impl(
            const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
        ) const;

        virtual void norms2Impl(
            const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
        ) const;

        virtual void normsInfImpl(
            const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
        ) const;

        /** Overridden protected functions from LinearOpBase */

        void applyImpl(
            const EOpTransp M_trans,
            const Thyra::MultiVectorBase<Scalar> &X,
            const Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > &Y,
            const Scalar alpha,
            const Scalar beta
        ) const;

    private:
    /* Private data members */

        Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> >
            vectorSpace_;

        /* The following view the same data in two different layouts. */
        Teuchos::ConstNonconstObjectContainer<tTMV> 
            orbVector_;
        Teuchos::ConstNonconstObjectContainer<tTV> 
            vector_;

    /* Private member functions */

        // Non-throwing Thyra::TpetraVector or MultiVector extraction methods.
        // Return null if casting failed.
        Teuchos::RCP<tTV>
        getVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& mv) const;

        Teuchos::RCP<const tTV>
        getConstVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const;

        Teuchos::RCP<tTMV>
        getMultiVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& mv) const;

        Teuchos::RCP<const tTMV>
        getConstMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const;
    };


    // Nonmember constructor for Vector.
    template <int dim, int degree, typename Scalar, class Node>
    inline
    Teuchos::RCP<Thyra::VectorBase<Scalar> >
    createVector(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<typename Vector<dim,degree,Scalar,Node>::TMV> &orbVector
    )
    {
        Teuchos::RCP<Vector<dim,degree,Scalar,Node> > v =
        Teuchos::rcp(new Vector<dim,degree,Scalar,Node>);
        v->initialize(vectorSpace, orbVector);
        return v;
    }


    //Nonmember const constructor for Vector.
    template <int dim, int degree, typename Scalar, class Node>
    inline
    Teuchos::RCP<const Thyra::VectorBase<Scalar> >
    createConstVector(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<const typename Vector<dim,degree,Scalar,Node>::TMV> &orbVector
    )
    {
        Teuchos::RCP<Vector<dim,degree,Scalar,Node> > v =
        Teuchos::rcp(new Vector<dim,degree,Scalar,Node>);
        v->constInitialize(vectorSpace, orbVector);
        return v;
    }

} /* End namespace Bilayer */
#endif
