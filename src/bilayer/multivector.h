/* 
* File:   bilayer/multivector.h
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on July 31, 2018, 10:00 PM
*/


#ifndef moire__bilayer_multivector_h
#define moire__bilayer_multivector_h

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
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_ConstNonconstObjectContainer.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_Operator.hpp>

#include "tools/types.h"
#include "tools/numbers.h"
#include "bilayer/dof_handler.h"
#include "bilayer/vector.h"
#include "bilayer/vectorspace.h"
#include "bilayer/scalarprod.h"



namespace Bilayer {
    using Thyra::EConj;
    using Thyra::EViewType;
    using Thyra::EStrideType;
    using Thyra::EOpTransp;

    template <int dim, int degree, typename Scalar, class Node> class VectorSpace;


    /* Concrete implementation of Thyra::MultiVector in terms of Bilayer::MultiVector. */

    template <int dim, int degree, typename Scalar, class Node = Kokkos::Compat::KokkosSerialWrapperNode>
    class MultiVector : virtual public Thyra::SpmdMultiVectorDefaultBase<Scalar>
    {
        public:
        /** Public types for Tpetra vectors, multivectors and their Thyra adapters*/
            typedef typename Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> TMV;
            typedef typename Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> tTMV;

            /// Construct to uninitialized
            MultiVector();

            /** Initialize.
            */
            void initialize(
                const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
                const Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
                const Teuchos::RCP<TMV> &tpetraMultiVector
            );

            /** Initialize.
            */
            void constInitialize(
                const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
                const Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
                const Teuchos::RCP<const TMV> &tpetraMultiVector
            );

            /** Get the embedded non-const Thyra::TpetraMultiVector. */
            Teuchos::RCP<tTMV> getThyraMultiVector();

            /** Get the embedded const Thyra::TpetraMultiVector. */
            Teuchos::RCP<const tTMV> getConstThyraMultiVector() const;

            /** Get the embedded non-const Thyra::TpetraMultiVector. */
            Teuchos::RCP<tTMV> getThyraOrbMultiVector();

            /** Get the embedded const Thyra::TpetraMultiVector. */
            Teuchos::RCP<const tTMV> getConstThyraOrbMultiVector() const;


            /** Overridden public functions form MultiVectorAdapterBase */
            Teuchos::RCP< const Thyra::ScalarProdVectorSpaceBase<Scalar> >
            domainScalarProdVecSpc() const;

        protected:

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

            Teuchos::RCP<const Thyra::VectorBase<Scalar> > colImpl(Teuchos::Ordinal j) const;
            Teuchos::RCP<Thyra::VectorBase<Scalar> > nonconstColImpl(Teuchos::Ordinal j);

            Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >
            contigSubViewImpl(const Teuchos::Range1D& colRng) const;

            Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >
            nonconstContigSubViewImpl(const Teuchos::Range1D& colRng);

            Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >
            nonContigSubViewImpl(const Teuchos::ArrayView<const int>& cols_in) const;

            Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >
            nonconstNonContigSubViewImpl(const Teuchos::ArrayView<const int>& cols_in);

            virtual void mvMultiReductApplyOpImpl(
                const RTOpPack::RTOpT<Scalar> &primary_op,
                const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > > &multi_vecs,
                const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > > &targ_multi_vecs,
                const Teuchos::ArrayView<const Teuchos::Ptr<RTOpPack::ReductTarget> > &reduct_objs,
                const Teuchos::Ordinal primary_global_offset
            ) const;

            void acquireDetachedMultiVectorViewImpl(
                const Teuchos::Range1D &rowRng,
                const Teuchos::Range1D &colRng,
                RTOpPack::ConstSubMultiVectorView<Scalar>* sub_mv
            ) const;

            void acquireNonconstDetachedMultiVectorViewImpl(
                const Teuchos::Range1D &rowRng,
                const Teuchos::Range1D &colRng,
                RTOpPack::SubMultiVectorView<Scalar>* sub_mv
            );

            void commitNonconstDetachedMultiVectorViewImpl(
                RTOpPack::SubMultiVectorView<Scalar>* sub_mv
            );


            /** Overridden protected functions from SpmdMultiVectorBase */
            Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<Scalar> > spmdSpaceImpl() const;

            void getNonconstLocalMultiVectorDataImpl(
                const Teuchos::Ptr<Teuchos::ArrayRCP<Scalar> > &localValues, const Teuchos::Ptr<Teuchos::Ordinal> &leadingDim
            );
            void getLocalMultiVectorDataImpl(
                const Teuchos::Ptr<Teuchos::ArrayRCP<const Scalar> > &localValues, const Teuchos::Ptr<Teuchos::Ordinal> &leadingDim
            ) const;

            /** Overridden protected functions from MultiVectorAdapterBase */
            virtual void euclideanApply(
                const EOpTransp M_trans,
                const Thyra::MultiVectorBase<Scalar> &X,
                const Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > &Y,
                const Scalar alpha,
                const Scalar beta
            ) const;

        private:

            // ///////////////////////////////////////
            // Private data members

            Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> >
                vectorSpace_;
            Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > 
                domainSpace_;
            /* The following view the same data in two different layouts. */
            Teuchos::ConstNonconstObjectContainer<tTMV> 
                orbMultiVector_;
            Teuchos::ConstNonconstObjectContainer<tTMV> 
                multiVector_;

            // ////////////////////////////////////
            // Private member functions

            // Non-throwing TpetraMultiVector extraction methods.
            // Return null if casting failed.
            Teuchos::RCP<tTMV>
            getMultiVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& mv) const;

            Teuchos::RCP<const tTMV>
            getConstMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const;

            Teuchos::RCP<tTMV>
            getOrbMultiVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& mv) const;

            Teuchos::RCP<const tTMV>
            getConstOrbMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const;
    };


    // Nonmember constructor for MultiVector.
    template <int dim, int degree, typename Scalar, class Node>
    inline
    Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >
    createMultiVector(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> > 
        &tpetraMultiVector
    )
    {
        size_t nVectors;
        if (tpetraMultiVector->getGlobalLength() == vectorSpace->getVecMap()->getGlobalNumElements())
            nVectors = tpetraMultiVector->getNumVectors();
        else if (tpetraMultiVector->getGlobalLength() == vectorSpace->getOrbVecMap()->getGlobalNumElements())
            nVectors = tpetraMultiVector->getNumVectors() / vectorSpace->getNumOrbitals();
        else 
            throw dealii::ExcInternalError();

        Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar>> 
        domainSpace = vectorSpace->createDomainVectorSpace(nVectors);

        Teuchos::RCP<MultiVector<dim,degree,Scalar,Node> > tmv =
        Teuchos::rcp(new MultiVector<dim,degree,Scalar,Node>);
        tmv->initialize(vectorSpace, domainSpace, tpetraMultiVector);
        return tmv;
    }


    // Nonmember const constructor for MultiVector.
    template <int dim, int degree, typename Scalar, class Node>
    inline
    Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >
    createConstMultiVector(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<const Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> > 
        &tpetraMultiVector
    )
    {
        size_t nVectors;
        if (tpetraMultiVector->getGlobalLength() == vectorSpace->getVecMap()->getGlobalNumElements())
            nVectors = tpetraMultiVector->getNumVectors();
        else if (tpetraMultiVector->getGlobalLength() == vectorSpace->getOrbVecMap()->getGlobalNumElements())
            nVectors = tpetraMultiVector->getNumVectors() / vectorSpace->getNumOrbitals();
        else 
            throw dealii::ExcInternalError();

        Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar>> 
        domainSpace = vectorSpace->createDomainVectorSpace(nVectors) ;
          
        Teuchos::RCP<MultiVector<dim,degree,Scalar,Node> > tmv =
        Teuchos::rcp(new MultiVector<dim,degree,Scalar,Node>);
        tmv->constInitialize(vectorSpace, domainSpace, tpetraMultiVector);
        return tmv;
    }

} /* End namespace Bilayer */
#endif
