/* 
* File:   bilayer/vectorspace.cpp
* Author: Paul Cazeaux
*
* Created on July 31, 2018, 10:00 PM
*/



#include "bilayer/vectorspace.h"

namespace Bilayer {


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<VectorSpace<dim,degree,Scalar,Node> >
    VectorSpace<dim,degree,Scalar,Node>::create()
    {
        const Teuchos::RCP<this_t> vs(new this_t);
        vs->weakSelfPtr_ = vs.create_weak();
        return vs;
    }

    template <int dim, int degree, typename Scalar, class Node>
    void 
    VectorSpace<dim,degree,Scalar,Node>::initialize(
        const Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> >& baseAlgebra
    )
    {
        size_t N = baseAlgebra_->getNumOrbitals();
        baseAlgebra_ = baseAlgebra;
        orbVecMap_ = baseAlgebra_->getMap();
        orbTpetraVectorSpace_ = Thyra::tpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node>
            (orbVecMap_);

        vecMap_ = Tpetra::createContigMapWithNode<
                        typename map_type::local_ordinal_type, 
                        typename map_type::global_ordinal_type, Node>
                        (   N * orbVecMap_->getGlobalNumElements(), 
                            N * orbVecMap_->getNodeNumElements(), 
                            orbVecMap_ ->getComm() );
        tpetraVectorSpace_ = Thyra::tpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node>
            (vecMap_);
        comm_ = Thyra::convertTpetraToThyraComm(baseAlgebra_->getComm());

        this->updateState(vecMap_->getGlobalNumElements(),
            !baseAlgebra_->getMap()->isDistributed());
        this->setScalarProd(euclideanScalarProd<dim,degree,Scalar,Node>(baseAlgebra_));
    }


    // Overridden from VectorSpace


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<Thyra::VectorBase<Scalar> >
    VectorSpace<dim,degree,Scalar,Node>::createMember() const
    {
        return createVector<dim,degree,Scalar,Node>(
            weakSelfPtr_.create_strong().getConst(),
            Teuchos::rcp(new 
                Tpetra::Vector<Scalar,types::loc_t,types::glob_t,Node>
                    (vecMap_)));
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP< Thyra::MultiVectorBase<Scalar> >
    VectorSpace<dim,degree,Scalar,Node>::createMembers(int numMembers) const
    {
        return createMultiVector<dim,degree,Scalar,Node>(
            weakSelfPtr_.create_strong().getConst(),
            createDomainVectorSpace(numMembers), 
            Teuchos::rcp(new 
                Tpetra::MultiVector<Scalar,types::loc_t,types::glob_t,Node>
                    (vecMap_, numMembers)));
    }


    template <int dim, int degree, typename Scalar, class Node>
    bool VectorSpace<dim,degree,Scalar,Node>::hasInCoreView(
        const Teuchos::Range1D& range_in, const EViewType viewType, const EStrideType strideType
    ) const
    {
        return tpetraVectorSpace_->hasInCoreView(range_in, viewType, strideType);
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP< const Thyra::VectorSpaceBase<Scalar> >
    VectorSpace<dim,degree,Scalar,Node>::clone() const
    {
        return vectorSpace<dim,degree,Scalar,Node>(baseAlgebra_);
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> >
    VectorSpace<dim,degree,Scalar,Node>::getAlgebra() const
    {
        return baseAlgebra_;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const typename BaseAlgebra<dim,degree,Scalar,Node>::Vector::map_type>
    VectorSpace<dim,degree,Scalar,Node>::getOrbVecMap() const
    {
        return orbVecMap_;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const typename BaseAlgebra<dim,degree,Scalar,Node>::Vector::map_type>
    VectorSpace<dim,degree,Scalar,Node>::getVecMap() const
    {
        return vecMap_;
    }

    template <int dim, int degree, typename Scalar, class Node>
    size_t
    VectorSpace<dim,degree,Scalar,Node>::getNumOrbitals() const
    {
        return baseAlgebra_->getNumOrbitals();
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node> >
    VectorSpace<dim,degree,Scalar,Node>::getOrbTpetraVectorSpace() const
    {
        return orbTpetraVectorSpace_;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node> >
    VectorSpace<dim,degree,Scalar,Node>::getTpetraVectorSpace() const
    {
        return tpetraVectorSpace_;
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node> >
    VectorSpace<dim,degree,Scalar,Node>::createDomainVectorSpace(size_t numElements) const
    {
        return Thyra::tpetraVectorSpace<Scalar,types::loc_t,types::glob_t,Node>(
                Tpetra::createLocalMapWithNode<types::loc_t,types::glob_t,Node>(
                    numElements, 
                    vecMap_->getComm(), 
                    vecMap_->getNode() ));
    }

    // Overridden from SpmdVectorSpaceDefaultBase


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Teuchos::Comm<Teuchos::Ordinal> >
    VectorSpace<dim,degree,Scalar,Node>::getComm() const
    {
        return tpetraVectorSpace_->getComm();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::Ordinal VectorSpace<dim,degree,Scalar,Node>::localSubDim() const
    {
        return tpetraVectorSpace_->localSubDim();
    }

    // private


    template <int dim, int degree, typename Scalar, class Node>
    VectorSpace<dim,degree,Scalar,Node>::VectorSpace()
    {
        // The base classes should automatically default initialize to a safe
        // uninitialized state.
    }

    /**
     * Explicit instantiations
     */
     template class VectorSpace<1,1,double>;
     template class VectorSpace<1,2,double>;
     template class VectorSpace<1,3,double>;
     template class VectorSpace<2,1,double>;
     template class VectorSpace<2,2,double>;
     template class VectorSpace<2,3,double>;

     template class VectorSpace<1,1,std::complex<double> >;
     template class VectorSpace<1,2,std::complex<double> >;
     template class VectorSpace<1,3,std::complex<double> >;
     template class VectorSpace<2,1,std::complex<double> >;
     template class VectorSpace<2,2,std::complex<double> >;
     template class VectorSpace<2,3,std::complex<double> >;

} // end namespace Bilayer
