/* 
* File:   bilayer/scalarprod.cpp
* Author: Paul Cazeaux
*
* Created on July 31, 2018, 10:00 PM
*/



#include "scalarprod.h"



namespace Bilayer {


    template <int dim, int degree, typename Scalar, class Node>
    void EuclideanScalarProd<dim,degree,Scalar,Node>::scalarProdsImpl(
        const Thyra::MultiVectorBase<Scalar>& X,
        const Thyra::MultiVectorBase<Scalar>& Y,
        const Teuchos::ArrayView<Scalar>& scalarProds_out
    ) const
    {
        typedef Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> TMV;
        Teuchos::RCP<const TMV> tX = this->getConstThyraMultiVector(Teuchos::rcpFromRef(X));
        Teuchos::RCP<const TMV> tY = this->getConstThyraMultiVector(Teuchos::rcpFromRef(Y));

        if (Teuchos::nonnull(tX) && Teuchos::nonnull(tY)) 
            baseAlgebra->dot(* tX->getConstTpetraMultiVector(), * tY->getConstTpetraMultiVector(), scalarProds_out );
        else // Throw an internal error
            throw dealii::ExcInternalError();
    }


    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
    EuclideanScalarProd<dim,degree,Scalar,Node>::
    getConstThyraMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const
    {
        using Teuchos::rcp_dynamic_cast;
        typedef MultiVector<dim,degree,Scalar,Node> MVec;
        typedef Vector<dim,degree,Scalar,Node> Vec;

        Teuchos::RCP<const MVec> mvec = rcp_dynamic_cast<const MVec>(mv);
        if (Teuchos::nonnull(mvec)) 
            return mvec->getConstThyraOrbMultiVector();

        Teuchos::RCP<const Vec> vec = rcp_dynamic_cast<const Vec>(mv);
        if (Teuchos::nonnull(vec)) 
            return vec->getConstThyraOrbVector();

        return Teuchos::null;
    }

    /**
     * Explicit instantiations
     */
     template class EuclideanScalarProd<1,1,double,types::DefaultNode>;
     template class EuclideanScalarProd<1,2,double,types::DefaultNode>;
     template class EuclideanScalarProd<1,3,double,types::DefaultNode>;
     template class EuclideanScalarProd<2,1,double,types::DefaultNode>;
     template class EuclideanScalarProd<2,2,double,types::DefaultNode>;
     template class EuclideanScalarProd<2,3,double,types::DefaultNode>;

     template class EuclideanScalarProd<1,1,std::complex<double>,types::DefaultNode>;
     template class EuclideanScalarProd<1,2,std::complex<double>,types::DefaultNode>;
     template class EuclideanScalarProd<1,3,std::complex<double>,types::DefaultNode>;
     template class EuclideanScalarProd<2,1,std::complex<double>,types::DefaultNode>;
     template class EuclideanScalarProd<2,2,std::complex<double>,types::DefaultNode>;
     template class EuclideanScalarProd<2,3,std::complex<double>,types::DefaultNode>;
} // end namespace Bilayer