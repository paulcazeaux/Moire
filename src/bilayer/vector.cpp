/* 
* File:   bilayer/vector.cpp
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on July 31, 2018, 10:00 PM
*/

#include <bilayer/vector.h>


namespace Bilayer {
    // Constructors/initializers/accessors
    template <int dim, int degree, typename Scalar, class Node>
    Vector<dim,degree,Scalar,Node>::Vector()
    {}


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::initialize(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<TMV> &inputVector
    )
    {
        #ifdef TEUCHOS_DEBUG
            TEUCHOS_ASSERT(Teuchos::nonnull(vectorSpace));
            TEUCHOS_ASSERT(Teuchos::nonnull(inputVector));
        #endif
        vectorSpace_ = vectorSpace;
        size_t 
        N = vectorSpace_->getNumOrbitals(),
        n_rows = vectorSpace_->getOrbVecMap()->getNodeNumElements();

        vector_.initialize(Teuchos::rcp<tTV>(new tTV()));
        orbVector_.initialize(Teuchos::rcp<tTMV>(new tTMV()));
        if (inputVector->getNumVectors() == 1 
                && inputVector->getMap()->isSameAs(* vectorSpace->getVecMap()))
        {
            typename TMV::dual_view_type::t_host
                localViewHost (inputVector->getLocalViewHost().data(), n_rows, N); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (inputVector->getLocalViewDevice().data(), n_rows, N);
            typename TMV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbVector_.getNonconstObj()->initialize(
                    vectorSpace_->getOrbTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(N),
                    Teuchos::rcp(new TMV (vectorSpace_->getOrbVecMap(), localDualView))
                                );
            
            vector_.getNonconstObj()->initialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    inputVector->getVectorNonConst(0));
        }
        else if (inputVector->getNumVectors() == vectorSpace_->getNumOrbitals() 
                && inputVector->getMap()->isSameAs(* vectorSpace->getOrbVecMap()))
        {
            typename TV::dual_view_type::t_host
                localViewHost (inputVector->getLocalViewHost().data(), n_rows*N, 1); 
            typename TV::dual_view_type::t_dev
                localViewDev  (inputVector->getLocalViewDevice().data(), n_rows*N, 1);
            typename TV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbVector_.getNonconstObj()->initialize(
                    vectorSpace_->getOrbTpetraVectorSpace(),
                    vectorSpace_->createDomainVectorSpace(N),
                    inputVector);
            vector_.getNonconstObj()->initialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    Teuchos::rcp(new TV (vectorSpace_->getVecMap(), localDualView))
                            );
        }
        else
            throw std::logic_error("Failed to initialize Bilayer::Vector!");
        this->updateSpmdSpace();
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::constInitialize(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<const TMV> &inputVector
    )
    {
        #ifdef TEUCHOS_DEBUG
            TEUCHOS_ASSERT(Teuchos::nonnull(vectorSpace));
            TEUCHOS_ASSERT(Teuchos::nonnull(inputVector));
        #endif
        vectorSpace_ = vectorSpace;
        size_t 
        N = vectorSpace_->getNumOrbitals(),
        n_rows = vectorSpace_->getOrbVecMap()->getNodeNumElements();

        Teuchos::RCP<tTV> vector (new tTV());
        Teuchos::RCP<tTMV> orbVector (new tTMV());
        if (inputVector->getNumVectors() == 1 
                && inputVector->getMap()->isSameAs(* vectorSpace->getVecMap()))
        {
            typename TMV::dual_view_type::t_host
                localViewHost (inputVector->getLocalViewHost().data(), n_rows, N); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (inputVector->getLocalViewDevice().data(), n_rows, N);
            typename TMV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbVector->constInitialize(
                    vectorSpace_->getOrbTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(N),
                    Teuchos::rcp(new TMV (vectorSpace_->getOrbVecMap(), localDualView))
                                );
            vector->constInitialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    inputVector->getVector(0));

            this->updateSpmdSpace();
        }
        else if (inputVector->getNumVectors() == vectorSpace_->getNumOrbitals() 
                && inputVector->getMap()->isSameAs(* vectorSpace->getOrbVecMap()))
        {
            typename TV::dual_view_type::t_host
                localViewHost (inputVector->getLocalViewHost().data(), n_rows*N, 1); 
            typename TV::dual_view_type::t_dev
                localViewDev  (inputVector->getLocalViewDevice().data(), n_rows*N, 1);
            typename TV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbVector->constInitialize(
                    vectorSpace_->getOrbTpetraVectorSpace(),
                    vectorSpace_->createDomainVectorSpace(N),
                    inputVector);
            vector->constInitialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    Teuchos::rcp(new TV (vectorSpace_->getVecMap(), localDualView))
                            );
        }
        else
            throw std::logic_error("Failed to constInitialize Bilayer::Vector!");

        vector_.initialize(vector);
        orbVector_.initialize(orbVector);
        this->updateSpmdSpace();
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getThyraVector()
    {
        return vector_.getNonconstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getConstThyraVector() const
    {
        return vector_.getConstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getThyraOrbVector()
    {
        return orbVector_.getNonconstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getConstThyraOrbVector() const
    {
        return orbVector_.getConstObj();
    }


    // Overridden from VectorDefaultBase
    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
    Vector<dim,degree,Scalar,Node>::domain() const
    {
        return vector_->domain();
    }


    // Overridden from SpmdMultiVectorBase


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::SpmdVectorSpaceBase<Scalar> >
    Vector<dim,degree,Scalar,Node>::spmdSpaceImpl() const
    {
        return vectorSpace_;
    }


    // Overridden from SpmdVectorBase


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::getNonconstLocalVectorDataImpl(
        const Teuchos::Ptr<Teuchos::ArrayRCP<Scalar> > &localValues )
    {
        vector_.getNonconstObj()->getNonconstLocalData(localValues);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::getLocalVectorDataImpl(
        const Teuchos::Ptr<Teuchos::ArrayRCP<const Scalar> > &localValues ) const
    {
        vector_.getConstObj()->getLocalData(localValues);
    }


    // Overridden from VectorBase


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::randomizeImpl(
        Scalar l,
        Scalar u
    )
    {
        vector_.getNonconstObj()->randomize(l,u);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::absImpl(
        const Thyra::VectorBase<Scalar>& x
    )
    {
        auto tx = this->getConstVector(Teuchos::rcpFromRef(x));
        if (Teuchos::nonnull(tx))
            vector_.getNonconstObj()->abs(* tx);
        else
            throw std::logic_error("Failed to cast Thyra::VectorBase argument as const Bilayer::Vector in Bilayer::Vector::absImpl!");
        
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::reciprocalImpl(
        const Thyra::VectorBase<Scalar>& x
    )
    {
        auto tx = this->getConstVector(Teuchos::rcpFromRef(x));
        if (Teuchos::nonnull(tx))
            vector_.getNonconstObj()->reciprocal(* tx);
        else
            throw std::logic_error("Failed to cast Thyra::VectorBase argument as const Bilayer::Vector in Bilayer::Vector::reciprocalImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::eleWiseScaleImpl(
        const Thyra::VectorBase<Scalar>& x
    )
    {
        auto tx = this->getConstVector(Teuchos::rcpFromRef(x));
        if (Teuchos::nonnull(tx))
            vector_.getNonconstObj()->ele_wise_scale(* tx);
        else
            throw std::logic_error("Failed to cast Thyra::VectorBase argument as const Bilayer::Vector in Bilayer::Vector::eleWiseScaleImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType
    Vector<dim,degree,Scalar,Node>::norm2WeightedImpl(
        const Thyra::VectorBase<Scalar>& x
    ) const
    {
        auto tx = this->getConstVector(Teuchos::rcpFromRef(x));
        if (Teuchos::nonnull(tx))
            return vector_->norm_2(*tx);
        else
            throw std::logic_error("Failed to cast Thyra::VectorBase argument as const Bilayer::Vector in Bilayer::Vector::norm2WeightedImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::applyOpImpl(
        const RTOpPack::RTOpT<Scalar> &op,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::VectorBase<Scalar> > > &vecs,
        const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::VectorBase<Scalar> > > &targ_vecs,
        const Teuchos::Ptr<RTOpPack::ReductTarget> &reduct_obj,
        const Teuchos::Ordinal global_offset
    ) const
    {
        Teuchos::Array<Teuchos::Ptr<const Thyra::VectorBase<Scalar> > > tvecs (vecs.size());
        Teuchos::Array<Teuchos::Ptr<Thyra::VectorBase<Scalar> > > ttarg_vecs (targ_vecs.size());
        

        for (int i = 0; i < vecs.size(); ++i) 
        {
            Teuchos::RCP<const tTV>
            tv = this->getConstVector(Teuchos::rcpFromPtr(vecs[i]));
            if (nonnull(tv))
                tvecs[i] = tv.ptr();
            else 
                throw std::logic_error("Failed to cast one or more Thyra::VectorBase arguments as const Bilayer::Vector in Bilayer::Vector::applyOpImpl!");
        }

        for (int i = 0; i < vecs.size(); ++i) 
        {
            Teuchos::RCP<tTV>
            tv = this->getVector(Teuchos::rcpFromPtr(targ_vecs[i]));
            if (nonnull(tv))
                ttarg_vecs[i] = tv.ptr();
            else 
                throw std::logic_error("Failed to cast one or more Thyra::VectorBase target arguments as Bilayer::Vector in Bilayer::Vector::applyOpImpl!");
        }

        vector_->applyOp(op, tvecs, ttarg_vecs, reduct_obj, global_offset);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::
    acquireDetachedVectorViewImpl(
        const Teuchos::Range1D& range,
        RTOpPack::ConstSubVectorView<Scalar>* sub_vec
    ) const
    {
        vector_.getConstObj()->acquireDetachedView(range, sub_vec);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::
    acquireNonconstDetachedVectorViewImpl(
        const Teuchos::Range1D& range,
        RTOpPack::SubVectorView<Scalar>* sub_vec 
    )
    {
        vector_.getNonconstObj()->acquireDetachedView(range, sub_vec);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::
        commitNonconstDetachedVectorViewImpl(
        RTOpPack::SubVectorView<Scalar>* sub_vec
    )
    {
        vector_.getNonconstObj()->commitDetachedView(sub_vec);
    }


    // Overridden protected functions from MultiVectorBase


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::assignImpl(Scalar alpha)
    {
        vector_.getNonconstObj()->assign(alpha);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::
    assignMultiVecImpl(const Thyra::MultiVectorBase<Scalar>& mv)
    {
        auto tmv = this->getConstMultiVector(Teuchos::rcpFromRef(mv));
        auto tv = this->getConstVector(Teuchos::rcpFromRef(mv));
        if (Teuchos::nonnull(tv))
            vector_.getNonconstObj()->assign(* tv);
        else if (Teuchos::nonnull(tmv))
            vector_.getNonconstObj()->assign(* tmv);
        else
            throw std::logic_error("Failed to cast Thyra::MultiVectorBase argument as const Bilayer::Vector or const Bilayer::MultiVector in Bilayer::Vector::assignMultiVecImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::scaleImpl(Scalar alpha)
    {
        vector_.getNonconstObj()->scale(alpha);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::updateImpl(
        Scalar alpha,
        const Thyra::MultiVectorBase<Scalar>& mv
    )
    {
        auto tmv = this->getConstMultiVector(Teuchos::rcpFromRef(mv));
        auto tv = this->getConstVector(Teuchos::rcpFromRef(mv));
        if (Teuchos::nonnull(tv))
            vector_.getNonconstObj()->update(alpha, * tv);
        else if (Teuchos::nonnull(tmv))
            vector_.getNonconstObj()->update(alpha, * tmv);
        else
            throw std::logic_error("Failed to cast Thyra::MultiVectorBase argument as const Bilayer::Vector or const Bilayer::MultiVector in Bilayer::Vector::updateImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::linearCombinationImpl(
        const Teuchos::ArrayView<const Scalar>& alpha,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > >& mv,
        const Scalar& beta
    )
    {
        Teuchos::Array<Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > > tmvs (mv.size());
        Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> > tmv, tv;
        bool allCastsSuccessful = true;
        for (int i = 0; i < mv.size(); ++i) 
        {
            tmv = this->getConstMultiVector(Teuchos::rcpFromPtr(mv[i]));
            tv = this->getConstVector(Teuchos::rcpFromPtr(mv[i]));
            if (Teuchos::nonnull(tv))
                tmvs[i] = tv.ptr();
            else if (Teuchos::nonnull(tmv))
                tmvs[i] = tmv.ptr();
            else 
            {
                allCastsSuccessful = false; 
                break;
            }
        }
        if (! allCastsSuccessful)
            throw std::logic_error("Failed to cast one or more Thyra::MultiVectorBase arguments as const Bilayer::MultiVector in Bilayer::Vector::linearCombinationImpl!");
        vector_.getNonconstObj()->linear_combination(alpha, tmvs, beta);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::dotsImpl(
        const Thyra::MultiVectorBase<Scalar>& mv,
        const Teuchos::ArrayView<Scalar>& prods
    ) const
    {
        auto tv = this->getConstVector(Teuchos::rcpFromRef(mv));
        auto tmv = this->getConstMultiVector(Teuchos::rcpFromRef(mv));
        if (Teuchos::nonnull(tv))
            vector_->dots(*tv, prods);
        else if (Teuchos::nonnull(tmv))
            vector_->dots(*tmv, prods);
        else
            throw std::logic_error("Failed to cast Thyra::MultiVectorBase argument as const Bilayer::Vector or Bilayer::MultiVector in Bilayer::Vector::dotsImpl!");
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::norms1Impl(
        const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
    ) const
    {
        vector_->norms_1(norms);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::norms2Impl(
        const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
    ) const
    {
        vector_->norms_2(norms);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::normsInfImpl(
        const Teuchos::ArrayView<typename Teuchos::ScalarTraits<Scalar>::magnitudeType>& norms
    ) const
    {
        vector_->norms_inf(norms);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::applyImpl(
        const EOpTransp M_trans,
        const Thyra::MultiVectorBase<Scalar> &X,
        const Teuchos::Ptr<Thyra::MultiVectorBase<Scalar> > &Y,
        const Scalar alpha,
        const Scalar beta
    ) const
    {
        Teuchos::RCP<const tTMV> 
        tX = this->getConstMultiVector(Teuchos::rcpFromRef(X));
        Teuchos::RCP<tTMV> 
        tY = this->getMultiVector(Teuchos::rcpFromPtr(Y));

        if (Teuchos::nonnull(tX) && Teuchos::nonnull(tY))
            vector_.getConstObj()->apply(M_trans, * tX, tY.ptr(), alpha, beta);
        else
            throw std::logic_error("Failed to cast one or more Thyra::MultiVectorBase arguments as Bilayer::MultiVector in Bilayer::Vector::applyImpl!");
    }


    // private

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::
    getVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& v) const
    {
        typedef Vector<dim,degree,Scalar,Node> Vec;

        Teuchos::RCP<Vec> vec = Teuchos::rcp_dynamic_cast<Vec>(v);
        if (Teuchos::nonnull(vec))
            return vec->getThyraVector();

        return Teuchos::null;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::
    getConstVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& v) const
    {
        typedef Vector<dim,degree,Scalar,Node> Vec;

        Teuchos::RCP<const Vec> vec = Teuchos::rcp_dynamic_cast<const Vec>(v);
        if (Teuchos::nonnull(vec))
            return vec->getConstThyraVector();

        return Teuchos::null;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::
        getMultiVector(const Teuchos::RCP<Thyra::MultiVectorBase<Scalar> >& v) const
    {
        typedef Vector<dim,degree,Scalar,Node> Vec;
        typedef MultiVector<dim,degree,Scalar,Node> MVec;

        Teuchos::RCP<MVec> mvec = Teuchos::rcp_dynamic_cast<MVec>(v);
        if (Teuchos::nonnull(mvec))
            return mvec->getThyraMultiVector();

        return Teuchos::null;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::
    getConstMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& v) const
    {
        typedef Vector<dim,degree,Scalar,Node> Vec;
        typedef MultiVector<dim,degree,Scalar,Node> MVec;

        Teuchos::RCP<const MVec> mvec = Teuchos::rcp_dynamic_cast<const MVec>(v);
        if (Teuchos::nonnull(mvec))
            return mvec->getConstThyraMultiVector();

        return Teuchos::null;
    }

    /**
     * Explicit instantiations
     */
     template class Vector<1,1,double,types::DefaultNode>;
     template class Vector<1,2,double,types::DefaultNode>;
     template class Vector<1,3,double,types::DefaultNode>;
     template class Vector<2,1,double,types::DefaultNode>;
     template class Vector<2,2,double,types::DefaultNode>;
     template class Vector<2,3,double,types::DefaultNode>;

     template class Vector<1,1,std::complex<double>,types::DefaultNode>;
     template class Vector<1,2,std::complex<double>,types::DefaultNode>;
     template class Vector<1,3,std::complex<double>,types::DefaultNode>;
     template class Vector<2,1,std::complex<double>,types::DefaultNode>;
     template class Vector<2,2,std::complex<double>,types::DefaultNode>;
     template class Vector<2,3,std::complex<double>,types::DefaultNode>;

} // end namespace Bilayer
