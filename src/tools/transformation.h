/* 
 * File:   transformation.h
 * Author: Paul Cazeaux
 *
 * Created on May 4, 2017, 12:58AM
 */


#ifndef moire__tools_transformation_h
#define moire__tools_transformation_h

#include "deal.II/base/tensor.h"


template<int dim>
struct Transformation {};

template<>
struct Transformation<1> {
    static double       matrix(const double scaling, const double angle) 
                        { return scaling; }; // in 1D, the angle is ignored
};

template<>
struct Transformation<2>{
    static dealii::Tensor<2,2>      matrix(const double scaling, const double angle)
    {
        const double rotation[2][2]
        = {{
            scaling * std::cos(angle), scaling * std::sin(angle)
            },
            {
            -scaling * std::sin(angle), scaling * std::cos(angle)
            }
        };
  return dealii::Tensor<2,2> (rotation);
    }
};

#endif
