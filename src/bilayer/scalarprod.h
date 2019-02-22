/* 
* File:   bilayer/scalarprod.h
* Author: Paul Cazeaux
*
* Reimplemented from Thyra Tpetra adapters templates
*
* Created on July 31, 2018, 10:00 PM
*/

#ifndef moire__bilayer_scalar_prod_h
#define moire__bilayer_scalar_prod_h

#include <memory>
#include <vector>
#include <array>
#include <utility>
#include <algorithm>
#include <fstream>

#include <complex>
#include "RTOpPack_Types.hpp"

#include <Thyra_EuclideanScalarProd.hpp>

#include <Tpetra_DefaultPlatform.hpp>
#include <Kokkos_Core.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Range1D.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ConstNonconstObjectContainer.hpp>
#include <Tpetra_MultiVector_decl.hpp>
#include <Tpetra_CrsMatrix_decl.hpp>
#include <Tpetra_Operator.hpp>


#include "tools/types.h"
#include "tools/numbers.h"
#include "bilayer/base_algebra.h"
#include "bilayer/vector.h"
#include "bilayer/multivector.h"



namespace Bilayer {

    /** Extends concrete implementation of a Euclidean scalar product for
    * bilayer C* algebra element vectors/multivectors.
    */
    template <int dim, int degree, typename Scalar, class Node=Kokkos::Compat::KokkosSerialWrapperNode>
    class EuclideanScalarProd : public Thyra::EuclideanScalarProd<Scalar> 
    {
    public:
        EuclideanScalarProd(Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> >& baseAlgebra):
            baseAlgebra(baseAlgebra)
            {};

    protected:

        /** Overridden from EuclideanScalarProd */

        /** If X and Y are both bilayer elements, computes the pair-wise
        * scalar products directly. Otherwise, this throws an error.
        */
        virtual void scalarProdsImpl(
            const Thyra::MultiVectorBase<Scalar>& X,
            const Thyra::MultiVectorBase<Scalar>& Y,
            const Teuchos::ArrayView<Scalar>& scalarProds
        ) const;

    private:
        //////////////////////
        // Private data member
        Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> > baseAlgebra;

        //////////////////////////
        // Private member function
        Teuchos::RCP<const Thyra::TpetraMultiVector<Scalar,types::loc_t,types::glob_t,Node> >
        getConstTpetraMultiVector(const Teuchos::RCP<const Thyra::MultiVectorBase<Scalar> >& mv) const;

    };


    // Nonmember constructor for EuclideanScalarProd.
    template <int dim, int degree, typename Scalar, class Node>
    inline
    Teuchos::RCP<const EuclideanScalarProd<dim,degree,Scalar,Node> >
    euclideanScalarProd(Teuchos::RCP<const BaseAlgebra<dim,degree,Scalar,Node> >& baseAlgebra)
    {
        return  Teuchos::rcp(new EuclideanScalarProd<dim,degree,Scalar,Node>(baseAlgebra));
    }


} // end namespace Bilayer
#endif