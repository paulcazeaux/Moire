/* 
* File:   bilayer/operator.cpp
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on February 24, 2019, 9:00AM
*/

#include <bilayer/operator.h>


namespace Bilayer {

    template <int dim, int degree, typename Scalar, class Node>
    Operator<dim,degree,Scalar,Node>::Operator()
    {}

    template <int dim, int degree, typename Scalar, class Node>
    void
    Operator<dim,degree,Scalar,Node>::initialize(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<op_type> &Op
        )
    {
        vectorSpace_ = vectorSpace;
        operator_.initialize(Op);
    }

    template <int dim, int degree, typename Scalar, class Node>
    void
    Operator<dim,degree,Scalar,Node>::constInitialize(
        const Teuchos::RCP<const VectorSpace<dim,degree,Scalar,Node> > &vectorSpace,
        const Teuchos::RCP<const op_type> &Op
        )
    {
        #ifdef TEUCHOS_DEBUG
            TEUCHOS_ASSERT(Teuchos::nonnull(vectorSpace));
            TEUCHOS_ASSERT(Teuchos::nonnull(Op));
        #endif
        vectorSpace_ = vectorSpace;
        operator_.initialize(Op);
    }

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
    Operator<dim,degree,Scalar,Node>::domain() const 
    {return vectorSpace_;}

    template <int dim, int degree, typename Scalar, class Node>
    Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> >
    Operator<dim,degree,Scalar,Node>::range() const 
    {return vectorSpace_;}


    template <int dim, int degree, typename Scalar, class Node>
    bool
    Operator<dim,degree,Scalar,Node>::opSupportedImpl (Thyra::EOpTransp M_trans) const
    {
        if (is_null(operator_))
            return false;

        if (M_trans == Thyra::NOTRANS)
            return true;

        if (M_trans == Thyra::CONJ)
            return !Teuchos::ScalarTraits<Scalar>::isComplex;

        return operator_->hasTransposeApply();
    }

    template <int dim, int degree, typename Scalar, class Node>
    void
    Operator<dim,degree,Scalar,Node>::applyImpl ( const Thyra::EOpTransp M_trans, 
                    const Thyra::MultiVectorBase< Scalar > &X_in, 
                    const Teuchos::Ptr< Thyra::MultiVectorBase< Scalar > > &Y_inout, 
                    const Scalar alpha, const Scalar beta) const
    {
        typedef Vector<dim,degree,Scalar,Node> Vec;
        typedef MultiVector<dim,degree,Scalar,Node> MVec;
        const Teuchos::RCP<const MVec> X = Teuchos::rcp_dynamic_cast<const MVec>
                    (Teuchos::rcpFromRef(X_in));
        Teuchos::RCP<MVec> Y = Teuchos::rcp_dynamic_cast<MVec>
                    (Teuchos::rcpFromPtr(Y_inout));
        const Teuchos::RCP<const Vec> vX = Teuchos::rcp_dynamic_cast<const Vec>
                    (Teuchos::rcpFromRef(X_in));
        Teuchos::RCP<Vec> vY = Teuchos::rcp_dynamic_cast<Vec>
                    (Teuchos::rcpFromPtr(Y_inout));

        const Teuchos::ETransp mode = Thyra::convertToTeuchosTransMode<Scalar>(M_trans);
        if (Teuchos::nonnull(vX) && Teuchos::nonnull(vY))
            operator_.getConstObj()->apply(
                        * vX->getConstThyraOrbVector()->getConstTpetraMultiVector(),
                        * vY->getThyraOrbVector()->getTpetraMultiVector(),
                        mode,
                        alpha, beta);
        else if (Teuchos::nonnull(X) && Teuchos::nonnull(Y))
            operator_.getConstObj()->apply(
                    * X->getConstThyraOrbMultiVector()->getConstTpetraMultiVector(),
                    * Y->getThyraOrbMultiVector()->getTpetraMultiVector(),
                    mode,
                    alpha, beta);
        else if (Teuchos::nonnull(vX) && Teuchos::nonnull(Y))
            operator_.getConstObj()->apply(
                        * vX->getConstThyraOrbVector()->getConstTpetraMultiVector(),
                        * Y->getThyraOrbMultiVector()->getTpetraMultiVector(),
                        mode,
                        alpha, beta);
        else
            throw std::logic_error("Failed to cast one or more Thyra::MultiVectorBase arguments to either Bilayer::Vector or Bilayer::MultiVector in Bilayer::Operator::applyImpl!");
    }

    /**
     * Explicit instantiations
     */
     template class Operator<1,1,double,types::DefaultNode>;
     template class Operator<1,2,double,types::DefaultNode>;
     template class Operator<1,3,double,types::DefaultNode>;
     template class Operator<2,1,double,types::DefaultNode>;
     template class Operator<2,2,double,types::DefaultNode>;
     template class Operator<2,3,double,types::DefaultNode>;

     template class Operator<1,1,std::complex<double>,types::DefaultNode>;
     template class Operator<1,2,std::complex<double>,types::DefaultNode>;
     template class Operator<1,3,std::complex<double>,types::DefaultNode>;
     template class Operator<2,1,std::complex<double>,types::DefaultNode>;
     template class Operator<2,2,std::complex<double>,types::DefaultNode>;
     template class Operator<2,3,std::complex<double>,types::DefaultNode>;
}