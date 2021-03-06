/* 
* File:   bilayer/multivector.cpp
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on August 2, 2018, 9:00 AM
*/

#include <bilayer/multivector.h>



namespace Bilayer {


    // Constructors/initializers/accessors


    template <int dim, int degree, typename Scalar, class Node>
    MultiVector<dim,degree,Scalar,Node>::MultiVector() 
    {}


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::initialize(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
        const Teuchos::RCP<Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> > &inputMVector
    )
    {
        #ifdef TEUCHOS_DEBUG
            TEUCHOS_ASSERT(Teuchos::nonnull(vectorSpace));
            TEUCHOS_ASSERT(Teuchos::nonnull(inputMVector));
        #endif
        vectorSpace_ = vectorSpace;
        domainSpace_ = domainSpace;
        size_t 
        N = vectorSpace_->getNumOrbitals(),
        n_rows = vectorSpace_->getOrbVecMap()->getNodeNumElements();

        orbMultiVector_.initialize(Teuchos::rcp<tTMV>(new tTMV()));
        multiVector_.initialize(Teuchos::rcp<tTMV>(new tTMV()));
        if (inputMVector->getMap()->isSameAs(* vectorSpace->getVecMap()))
        {
            size_t n = inputMVector->getNumVectors();

            typename TMV::dual_view_type::t_host
                localViewHost (inputMVector->getLocalViewHost().data(), n_rows, N*n); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (inputMVector->getLocalViewDevice().data(), n_rows, N*n);
            typename TMV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbMultiVector_.getNonconstObj()->initialize(
                    vectorSpace_->getOrbTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(N*n),
                    Teuchos::rcp(new TMV (vectorSpace_->getOrbVecMap(), localDualView))
                                );
            multiVector_.getNonconstObj()->initialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(n),
                    inputMVector);
        }
        else if (inputMVector->getNumVectors() % N == 0 
                && inputMVector->getMap()->isSameAs(* vectorSpace->getOrbVecMap()))
        {
            size_t n = inputMVector->getNumVectors() / N;

            typename TMV::dual_view_type::t_host
                localViewHost (inputMVector->getLocalViewHost().data(), n_rows*N, n); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (inputMVector->getLocalViewDevice().data(), n_rows*N, n);
            typename TMV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbMultiVector_.getNonconstObj()->initialize(
                    vectorSpace_->getOrbTpetraVectorSpace(),
                    vectorSpace_->createDomainVectorSpace(N),
                    inputMVector);
            multiVector_.getNonconstObj()->initialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(n*N),
                    Teuchos::rcp(new TMV (vectorSpace_->getVecMap(), localDualView))
                            );
        }
        else
            throw std::logic_error("Failed to initialize Bilayer::MultiVector!");

        this->updateSpmdSpace();
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::constInitialize(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
        const Teuchos::RCP<const Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node> > &inputMVector
    )
    {
        #ifdef TEUCHOS_DEBUG
            TEUCHOS_ASSERT(Teuchos::nonnull(vectorSpace));
            TEUCHOS_ASSERT(Teuchos::nonnull(inputMVector));
        #endif
        vectorSpace_ = vectorSpace;
        domainSpace_ = domainSpace;
        size_t 
        N = vectorSpace_->getNumOrbitals(),
        n_rows = vectorSpace_->getOrbVecMap()->getNodeNumElements();

        Teuchos::RCP<tTMV> orbMultiVector (new tTMV());
        Teuchos::RCP<tTMV> multiVector (new tTMV());
        if (inputMVector->getMap()->isSameAs(* vectorSpace->getVecMap()))
        {
            size_t n = inputMVector->getNumVectors();

            typename TMV::dual_view_type::t_host
                localViewHost (inputMVector->getLocalViewHost().data(), n_rows, N*n); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (inputMVector->getLocalViewDevice().data(), n_rows, N*n);
            typename TMV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbMultiVector->constInitialize(
                    vectorSpace_->getOrbTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(N*n),
                    Teuchos::rcp(new TMV (vectorSpace_->getOrbVecMap(), localDualView))
                                );
            multiVector->constInitialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(n),
                    inputMVector);
        }
        else if (inputMVector->getNumVectors() % N == 0 
                && inputMVector->getMap()->isSameAs(* vectorSpace->getOrbVecMap()))
        {
            size_t n = inputMVector->getNumVectors() / N;

            typename TMV::dual_view_type::t_host
                localViewHost (inputMVector->getLocalViewHost().data(), n_rows*N, n); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (inputMVector->getLocalViewDevice().data(), n_rows*N, n);
            typename TMV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbMultiVector->constInitialize(
                    vectorSpace_->getOrbTpetraVectorSpace(),
                    vectorSpace_->createDomainVectorSpace(N),
                    inputMVector);
            multiVector->constInitialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(n*N),
                    Teuchos::rcp(new TMV (vectorSpace_->getVecMap(), localDualView))
                            );
        }
        else
            throw std::logic_error("Failed to constInitialize Bilayer::MultiVector!");

        orbMultiVector_.initialize(orbMultiVector);
        multiVector_.initialize(multiVector);
        this->updateSpmdSpace();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::getThyraMultiVector()
    {
        return multiVector_.getNonconstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::getConstThyraMultiVector() const
    {
        return multiVector_.getConstObj();
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::getThyraOrbMultiVector()
    {
        return orbMultiVector_.getNonconstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::getConstThyraOrbMultiVector() const
    {
        return orbMultiVector_.getConstObj();
    }


    // Overridden public functions form MultiVectorAdapterBase


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP< const Thyra::ScalarProdVectorSpaceBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::domainScalarProdVecSpc() const
    {
        return domainSpace_;
    }


    // Overridden protected functions from MultiVectorBase


    template <int dim, int degree, typename Scalar, class Node>
    void
    MultiVector<dim,degree,Scalar,Node>::assignImpl(Scalar alpha)
    {
        multiVector_.getNonconstObj()->assign(alpha);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::
    assignMultiVecImpl(const Thyra::MultiVectorBase<Scalar>& mv)
    {
        auto tmv = this->getConstMultiVector(Teuchos::rcpFromRef(mv));

        if (Teuchos::nonnull(tmv)) 
            multiVector_.getNonconstObj()->assign(*tmv);
        else 
            throw std::logic_error("Failed to cast argument Thyra::MultiVectorBase to Bilayer::MultiVector in assignMultiVecImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void
    MultiVector<dim,degree,Scalar,Node>::scaleImpl(Scalar alpha)
    {
        multiVector_.getNonconstObj()->scale(alpha);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::updateImpl(
    Scalar alpha,
        const Thyra::MultiVectorBase<Scalar>& mv
    )
    {
        auto tmv = this->getConstMultiVector(Teuchos::rcpFromRef(mv));

        // If the cast succeeded, call Tpetra directly.
        if (Teuchos::nonnull(tmv)) 
        {
            multiVector_.getNonconstObj()->update(alpha, *tmv);
        } 
        else 
            throw std::logic_error("Failed to cast argument Thyra::MultiVectorBase to Bilayer::MultiVector in Bilayer::MultiVector::updateImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::linearCombinationImpl(
        const Teuchos::ArrayView<const Scalar>& alpha,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > >& mv,
        const Scalar& beta
    )
    {
        Teuchos::Array<Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > > tmvs (mv.size());
        Teuchos::RCP<const tTMV> tmv;
        for (int i = 0; i < mv.size(); ++i) 
        {
        tmv = this->getConstMultiVector(Teuchos::rcpFromPtr(mv[i]));
        if (nonnull(tmv))
            tmvs[i] = tmv.ptr();
        else 
            throw std::logic_error("Failed to cast one or more Thyra::MultiVectorBase to Bilayer::MultiVector in Bilayer::MultiVector::linearCombinationImpl!");
        }
        // Teuchos::Array<const Teuchos::Ptr<const tTMV> > tmvs_const (tmvs);

        multiVector_.getNonconstObj()->linear_combination(alpha, tmvs(), beta);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::dotsImpl(
        const Thyra::MultiVectorBase<Scalar>& mv,
        const Teuchos::ArrayView<Scalar>& prods
    ) const
    {
        auto tmv = this->getConstMultiVector(Teuchos::rcpFromRef(mv));

        if (Teuchos::nonnull(tmv))
            multiVector_.getConstObj()->dots(* tmv, prods);
        else
            throw std::logic_error("Failed to cast argument Thyra::MultiVectorBase to Bilayer::MultiVector in Bilayer::MultiVector::dotsImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::norms1Impl(
        const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
    ) const
    {
        multiVector_.getConstObj()->norms_1(norms);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::norms2Impl(
        const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
    ) const
    {
        multiVector_.getConstObj()->norms_2(norms);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::normsInfImpl(
        const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
    ) const
    {
        multiVector_.getConstObj()->norms_inf(norms);
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::VectorBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::colImpl(Teuchos::Ordinal j) const
    {
        typedef typename Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> tTV;
        const RCP<const tTV> tCol = Teuchos::rcp_dynamic_cast<const tTV>
             (this->getConstThyraMultiVector()->col(j));
        assert(Teuchos::nonnull(tCol));
   
        return createConstVector(vectorSpace_, tCol->getConstTpetraVector());
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::VectorBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::nonconstColImpl(Teuchos::Ordinal j)
    {
        typedef typename Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> tTV;
        const RCP<tTV> tCol = Teuchos::rcp_dynamic_cast<tTV>
             (this->getThyraMultiVector()->col(j));
        assert(Teuchos::nonnull(tCol));
   
        return createVector(vectorSpace_, tCol->getTpetraVector());    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::contigSubViewImpl(
        const Teuchos::Range1D& colRng
    ) const
    {
        size_t N = vectorSpace_->getNumOrbitals();
        Teuchos::Range1D OrbColRng (N*colRng.lbound(), N*colRng.ubound()+N-1);

        const RCP<const tTMV> tView = Teuchos::rcp_dynamic_cast<const tTMV>
            (this->getConstThyraMultiVector()->subView(colRng));
        const RCP<const tTMV> tOrbView = Teuchos::rcp_dynamic_cast<const tTMV>
            (this->getConstThyraOrbMultiVector()->subView(OrbColRng));
   
        assert(Teuchos::nonnull(tView) && Teuchos::nonnull(tOrbView));
   
        const RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
            vectorSpace_->createDomainVectorSpace(colRng.size());

        return Teuchos::rcp(new MultiVector<dim,degree,Scalar,Node>(
                        vectorSpace_,viewDomainSpace,tView,tOrbView));
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::nonconstContigSubViewImpl(
        const Teuchos::Range1D& colRng
    )
    {
        size_t N = vectorSpace_->getNumOrbitals();
        Teuchos::Range1D OrbColRng (N*colRng.lbound(), N*colRng.ubound()+N-1);

        const RCP<tTMV> tView = Teuchos::rcp_dynamic_cast<tTMV>
            (this->getThyraMultiVector()->subView(colRng));
        const RCP<tTMV> tOrbView = Teuchos::rcp_dynamic_cast<tTMV>
            (this->getThyraOrbMultiVector()->subView(OrbColRng));
   
        assert(Teuchos::nonnull(tView) && Teuchos::nonnull(tOrbView));
   
        const RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
            vectorSpace_->createDomainVectorSpace(colRng.size());

        return Teuchos::rcp(new MultiVector<dim,degree,Scalar,Node>(
                        vectorSpace_,viewDomainSpace,tView,tOrbView));
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::nonContigSubViewImpl(
        const Teuchos::ArrayView<const int>& colRng
    ) const
    {
        size_t N = vectorSpace_->getNumOrbitals();
        Teuchos::Array<int> OrbColRng (N * colRng.size());
        for (int i=0; i<colRng.size(); i++)
            for (size_t o=0; o<N; ++o)
                OrbColRng[N*i+o] = N*colRng[i]+o;

        const RCP<const tTMV> tView = Teuchos::rcp_dynamic_cast<const tTMV>
            (this->getConstThyraMultiVector()->subView(colRng));
        const RCP<const tTMV> tOrbView = Teuchos::rcp_dynamic_cast<const tTMV>
            (this->getConstThyraOrbMultiVector()->subView(OrbColRng));

        assert(Teuchos::nonnull(tView) && Teuchos::nonnull(tOrbView));
   
        const RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
            vectorSpace_->createDomainVectorSpace(colRng.size());

        return Teuchos::rcp(new MultiVector<dim,degree,Scalar,Node>(
                        vectorSpace_,viewDomainSpace,tView,tOrbView));
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::nonconstNonContigSubViewImpl(
        const Teuchos::ArrayView<const int>& colRng
    )
    {
        size_t N = vectorSpace_->getNumOrbitals();
        Teuchos::Array<int> OrbColRng (N * colRng.size());
        for (int i=0; i<colRng.size(); i++)
            for (size_t o=0; o<N; ++o)
                OrbColRng[N*i+o] = N*colRng[i]+o;

        const RCP<tTMV> tView = Teuchos::rcp_dynamic_cast<tTMV>
            (this->getThyraMultiVector()->subView(colRng));
        const RCP<tTMV> tOrbView = Teuchos::rcp_dynamic_cast<tTMV>
            (this->getThyraOrbMultiVector()->subView(OrbColRng));
   
        assert(Teuchos::nonnull(tView) && Teuchos::nonnull(tOrbView));
   
        const RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > viewDomainSpace =
            vectorSpace_->createDomainVectorSpace(colRng.size());

        return Teuchos::rcp(new MultiVector<dim,degree,Scalar,Node>(
                        vectorSpace_,viewDomainSpace,tView,tOrbView));
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::
    mvMultiReductApplyOpImpl(
        const RTOpPack::RTOpT<Scalar> &primary_op,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > > &multi_vecs,
        const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > > &targ_multi_vecs,
        const Teuchos::ArrayView<const Teuchos::Ptr<RTOpPack::ReductTarget> > &reduct_objs,
        const Teuchos::Ordinal primary_global_offset
    ) const
    {

        Teuchos::Array<Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > > tmulti_vecs (multi_vecs.size());
        Teuchos::Array<Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > > ttarg_multi_vecs (targ_multi_vecs.size());
        
        for (int i = 0; i < multi_vecs.size(); ++i) 
        {
            Teuchos::RCP<const tTMV> tmv 
                = this->getConstMultiVector(Teuchos::rcpFromPtr(multi_vecs[i]));
            if (nonnull(tmv))
                tmulti_vecs[i] = tmv.ptr();
            else 
                throw std::logic_error("Failed to cast one or more Thyra::MultiVectorBase to Bilayer::MultiVector in Bilayer::MultiVector::mvMultiReductApplyOpImpl!");
        }

        Teuchos::RCP<tTMV> tmv;
        for (int i = 0; i < multi_vecs.size(); ++i) 
        {
            Teuchos::RCP<tTMV> tmv 
                = this->getMultiVector(Teuchos::rcpFromPtr(targ_multi_vecs[i]));
            if (nonnull(tmv))
                ttarg_multi_vecs[i] = tmv.ptr();
            else 
                throw std::logic_error("Failed to cast one or more target Thyra::MultiVectorBase to Bilayer::MultiVector in Bilayer::MultiVector::mvMultiReductApplyOpImpl!");
        }

        multiVector_.getConstObj()->applyOp(
            primary_op, tmulti_vecs, ttarg_multi_vecs, reduct_objs, primary_global_offset);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::
    acquireDetachedMultiVectorViewImpl(
        const Teuchos::Range1D &rowRng,
        const Teuchos::Range1D &colRng,
        RTOpPack::ConstSubMultiVectorView<Scalar>* sub_mv
    ) const
    {
        multiVector_.getConstObj()->acquireDetachedView(rowRng, colRng, sub_mv);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::
    acquireNonconstDetachedMultiVectorViewImpl(
        const Teuchos::Range1D &rowRng,
        const Teuchos::Range1D &colRng,
        RTOpPack::SubMultiVectorView<Scalar>* sub_mv
    )
    {
        multiVector_.getNonconstObj()->acquireDetachedView(rowRng, colRng, sub_mv);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::
    commitNonconstDetachedMultiVectorViewImpl(
        RTOpPack::SubMultiVectorView<Scalar>* sub_mv
    )
    {
        multiVector_.getNonconstObj()->commitDetachedView(sub_mv);
    }


    // Overridden protected members from SpmdMultiVectorBase


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<Scalar> >
    MultiVector<dim,degree,Scalar,Node>::spmdSpaceImpl() const
    {
        return vectorSpace_;
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::getNonconstLocalMultiVectorDataImpl(
        const Teuchos::Ptr<Teuchos::ArrayRCP<Scalar> > &localValues, const Teuchos::Ptr<Teuchos::Ordinal> &leadingDim
    )
    {
        multiVector_.getNonconstObj()->getNonconstLocalData(localValues, leadingDim);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::getLocalMultiVectorDataImpl(
        const Teuchos::Ptr<Teuchos::ArrayRCP<const Scalar> > &localValues, const Teuchos::Ptr<Teuchos::Ordinal> &leadingDim
    ) const
    {
        multiVector_.getConstObj()->getLocalData(localValues, leadingDim);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void MultiVector<dim,degree,Scalar,Node>::euclideanApply(
        const EOpTransp M_trans,
        const Thyra::MultiVectorBase<Scalar> &X,
        const Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > &Y,
        const Scalar alpha,
        const Scalar beta
    ) const
    {
        // Check whether one or both of the two vectors X,Y is a Bilayer::MultiVector.
        Teuchos::RCP<const tTMV >
            tX = this->getConstMultiVector(Teuchos::rcpFromRef(X));
        Teuchos::RCP<tTMV >
            tY = this->getMultiVector(Teuchos::rcpFromPtr(Y));

        if (Teuchos::nonnull(tX) && Teuchos::nonnull(tY))
            multiVector_.getConstObj()->apply(M_trans, * tX, tY.ptr(), alpha, beta);
        else if (Teuchos::nonnull(tX))
            multiVector_.getConstObj()->apply(M_trans, * tX, Y, alpha, beta);
        else if (Teuchos::nonnull(tY))
            multiVector_.getConstObj()->apply(M_trans, X, tY.ptr(), alpha, beta);
        else
            multiVector_.getConstObj()->apply(M_trans, X, Y, alpha, beta);
    }

    // private

    template <int dim, int degree, typename Scalar, class Node>
    MultiVector<dim,degree,Scalar,Node>::MultiVector(
                const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
                const Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
                const Teuchos::RCP<tTMV> &multiVector,
                const Teuchos::RCP<tTMV> &orbMultiVector)
    {
        vectorSpace_ = vectorSpace;
        domainSpace_ = domainSpace;
        multiVector_.initialize(multiVector);
        orbMultiVector_.initialize(orbMultiVector);
        this->updateSpmdSpace();
    }

    template <int dim, int degree, typename Scalar, class Node>
    MultiVector<dim,degree,Scalar,Node>::MultiVector(
                const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
                const Teuchos::RCP<const Thyra::ScalarProdVectorSpaceBase<Scalar> > &domainSpace,
                const Teuchos::RCP<const tTMV> &multiVector,
                const Teuchos::RCP<const tTMV> &orbMultiVector)
    {
        vectorSpace_ = vectorSpace;
        domainSpace_ = domainSpace;
        multiVector_.initialize(multiVector);
        orbMultiVector_.initialize(orbMultiVector);
        this->updateSpmdSpace();
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::
        getMultiVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& mv) const
    {
        typedef MultiVector<dim,degree,Scalar,Node> MVec;
        typedef Vector<dim,degree,Scalar,Node> Vec;

        Teuchos::RCP<MVec> vecs = Teuchos::rcp_dynamic_cast<MVec>(mv);
        if (Teuchos::nonnull(vecs)) 
            return vecs->getThyraMultiVector();

        return Teuchos::null;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::
    getConstMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const
    {
        typedef MultiVector<dim,degree,Scalar,Node> MVec;
        Teuchos::RCP<const MVec> vecs = Teuchos::rcp_dynamic_cast<const MVec>(mv);
        // if (Teuchos::nonnull(vecs))
        //     std::cout << "OK" << std::endl;
        // else
        // {
        //     std::cout << mv << std::endl;
        //     std::cout << vecs << std::endl;
        // }
        if (Teuchos::nonnull(vecs))
            return vecs->getConstThyraMultiVector();

        return Teuchos::null;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::
    getOrbMultiVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& mv) const
    {
        typedef MultiVector<dim,degree,Scalar,Node> MVec;
        typedef Vector<dim,degree,Scalar,Node> Vec;

        Teuchos::RCP<MVec> vecs = Teuchos::rcp_dynamic_cast<MVec>(mv);
        if (Teuchos::nonnull(vecs)) 
            return vecs->getThyraOrbMultiVector();

        Teuchos::RCP<Vec> vec = Teuchos::rcp_dynamic_cast<Vec>(mv);
        if (Teuchos::nonnull(vec))
            return vec->getThyraOrbVector();

        return Teuchos::null;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::
    getConstOrbMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const
    {
        typedef MultiVector<dim,degree,Scalar,Node> MVec;
        typedef Vector<dim,degree,Scalar,Node> Vec;

        Teuchos::RCP<const MVec> vecs = Teuchos::rcp_dynamic_cast<const MVec>(mv);
        if (Teuchos::nonnull(vecs))
            return vecs->getConstThyraOrbMultiVector();

        Teuchos::RCP<const Vec> vec = Teuchos::rcp_dynamic_cast<const Vec>(mv);
        if (Teuchos::nonnull(vec))
            return vec->getConstThyraOrbVector();

        return Teuchos::null;
    }


    /**
     * Explicit instantiations
     */
     template class MultiVector<1,1,double,types::DefaultNode>;
     template class MultiVector<1,2,double,types::DefaultNode>;
     template class MultiVector<1,3,double,types::DefaultNode>;
     template class MultiVector<2,1,double,types::DefaultNode>;
     template class MultiVector<2,2,double,types::DefaultNode>;
     template class MultiVector<2,3,double,types::DefaultNode>;

     template class MultiVector<1,1,std::complex<double>,types::DefaultNode>;
     template class MultiVector<1,2,std::complex<double>,types::DefaultNode>;
     template class MultiVector<1,3,std::complex<double>,types::DefaultNode>;
     template class MultiVector<2,1,std::complex<double>,types::DefaultNode>;
     template class MultiVector<2,2,std::complex<double>,types::DefaultNode>;
     template class MultiVector<2,3,std::complex<double>,types::DefaultNode>;

} // end namespace Bilayer
