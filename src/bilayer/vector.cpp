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

        if (inputVector->getNumVectors() == 1 
                && inputVector->getMap()->isSameAs(* vectorSpace->getVecMap()))
        {
            typename TMV::dual_view_type
                localView = inputVector->getDualView();

            typename TMV::dual_view_type::t_host
                localViewHost (localView.h_view.data(), n_rows, N); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (localView.d_view.data(), n_rows, N);
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

            this->updateSpmdSpace();
        }
        else if (inputVector->getNumVectors() == vectorSpace_->getNumOrbitals() 
                && inputVector->getMap()->isSameAs(* vectorSpace->getOrbVecMap()))
        {
            typename TMV::dual_view_type
                localView = inputVector->getDualView();

            typename TV::dual_view_type::t_host
                localViewHost (localView.h_view.data(), n_rows*N, 1); 
            typename TV::dual_view_type::t_dev
                localViewDev  (localView.d_view.data(), n_rows*N, 1);
            typename TV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbVector_.getNonconstObj()->initialize(
                    vectorSpace_->getOrbTpetraVectorSpace(),
                    vectorSpace_->createDomainVectorSpace(N),
                    inputVector);
            vector_.getNonconstObj()->initialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    Teuchos::rcp(new TV (vectorSpace_->getOrbVecMap(), localDualView))
                            );
        }
        else
            throw dealii::ExcInternalError();

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

        if (inputVector->getNumVectors() == 1 
                && inputVector->getMap()->isSameAs(* vectorSpace->getVecMap()))
        {
            typename TMV::dual_view_type
                localView = inputVector->getDualView();

            typename TMV::dual_view_type::t_host
                localViewHost (localView.h_view.data(), n_rows, N); 
            typename TMV::dual_view_type::t_dev
                localViewDev  (localView.d_view.data(), n_rows, N);
            typename TMV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbVector_.getNonconstObj()->constInitialize(
                    vectorSpace_->getOrbTpetraVectorSpace(), 
                    vectorSpace_->createDomainVectorSpace(N),
                    Teuchos::rcp(new TMV (vectorSpace_->getOrbVecMap(), localDualView))
                                );
            vector_.getNonconstObj()->constInitialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    inputVector->getVector(0));

            this->updateSpmdSpace();
        }
        else if (inputVector->getNumVectors() == vectorSpace_->getNumOrbitals() 
                && inputVector->getMap()->isSameAs(* vectorSpace->getOrbVecMap()))
        {
            typename TMV::dual_view_type
                localView = inputVector->getDualView();

            typename TV::dual_view_type::t_host
                localViewHost (localView.h_view.data(), n_rows*N, 1); 
            typename TV::dual_view_type::t_dev
                localViewDev  (localView.d_view.data(), n_rows*N, 1);
            typename TV::dual_view_type
                localDualView (localViewHost, localViewDev);

            orbVector_.getNonconstObj()->constInitialize(
                    vectorSpace_->getOrbTpetraVectorSpace(),
                    vectorSpace_->createDomainVectorSpace(N),
                    inputVector);
            vector_.getNonconstObj()->constInitialize(
                    vectorSpace_->getTpetraVectorSpace(), 
                    Teuchos::rcp(new TV (vectorSpace_->getVecMap(), localDualView))
                            );
        }
        else
            throw dealii::ExcInternalError();

        this->updateSpmdSpace();
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getTpetraVector()
    {
        return vector_.getNonconstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getConstTpetraVector() const
    {
        return vector_.getConstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getTpetraOrbVector()
    {
        return orbVector_.getNonconstObj();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    Vector<dim,degree,Scalar,Node>::getConstTpetraOrbVector() const
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
            throw dealii::ExcInternalError();
        
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
            throw dealii::ExcInternalError();
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
            throw dealii::ExcInternalError();
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
            throw dealii::ExcInternalError();
        
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
                throw dealii::ExcInternalError();
        }

        for (int i = 0; i < vecs.size(); ++i) 
        {
            Teuchos::RCP<tTV>
            tv = this->getVector(Teuchos::rcpFromPtr(targ_vecs[i]));
            if (nonnull(tv))
                ttarg_vecs[i] = tv.ptr();
            else 
                throw dealii::ExcInternalError();
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
        if (Teuchos::nonnull(tmv))
            vector_.getNonconstObj()->assign(mv);
        else
            throw dealii::ExcInternalError();
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
        if (Teuchos::nonnull(tmv))
            vector_.getNonconstObj()->update(alpha, mv);
        else
            throw dealii::ExcInternalError();
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::linearCombinationImpl(
        const Teuchos::ArrayView<const Scalar>& alpha,
        const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > >& mv,
        const Scalar& beta
    )
    {
        Teuchos::Array<Teuchos::Ptr<const Thyra::MultiVectorBase<Scalar> > > tmvs (mv.size());
        Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> > tmv;
        for (int i = 0; i < mv.size(); ++i) 
        {
            tmv = this->getConstMultiVector(Teuchos::rcpFromPtr(mv[i]));
            if (nonnull(tmv))
                tmvs[i] = tmv.ptr();
            else 
                throw dealii::ExcInternalError();
        }

        vector_.getNonconstObj()->linear_combination(alpha, tmvs, beta);
    }


    template <int dim, int degree, typename Scalar, class Node>
    void Vector<dim,degree,Scalar,Node>::dotsImpl(
        const Thyra::MultiVectorBase<Scalar>& mv,
        const Teuchos::ArrayView<Scalar>& prods
    ) const
    {
        auto tmv = this->getConstVector(Teuchos::rcpFromRef(mv));
        if (Teuchos::nonnull(tmv))
            vector_->dots(*tmv, prods);
        else
            throw dealii::ExcInternalError();
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
            throw dealii::ExcInternalError();
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
            return vec->getTpetraVector();

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
            return vec->getConstTpetraVector();

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
            return mvec->getTpetraMultiVector();

        return Teuchos::null;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    MultiVector<dim,degree,Scalar,Node>::
    getConstMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& v) const
    {
        typedef Vector<dim,degree,Scalar,Node> Vec;
        typedef MultiVector<dim,degree,Scalar,Node> MVec;

        Teuchos::RCP<const MVec> mvec = Teuchos::rcp_dynamic_cast<const MVec>(v);
        if (Teuchos::nonnull(mvec))
            return mvec->getConstTpetraMultiVector();

        return Teuchos::null;
    }

    /**
     * Explicit instantiations
     */
     template class Vector<1,1,double>;
     template class Vector<1,2,double>;
     template class Vector<1,3,double>;
     template class Vector<2,1,double>;
     template class Vector<2,2,double>;
     template class Vector<2,3,double>;

     template class Vector<1,1,std::complex<double> >;
     template class Vector<1,2,std::complex<double> >;
     template class Vector<1,3,std::complex<double> >;
     template class Vector<2,1,std::complex<double> >;
     template class Vector<2,2,std::complex<double> >;
     template class Vector<2,3,std::complex<double> >;

} // end namespace Bilayer
