/* 
 * File:   transformation.h
 * Author: Paul Cazeaux
 *
 * Created on May 4, 2017, 12:58AM
 */


#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H


#include "deal.II/base/tensor.h"
#include "deal.II/physics/transformations.h"


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
                        { return scaling * dealii::Physics::Transformations::Rotations::rotation_matrix_2d<double>(angle); };
};




#endif /* TRANSFORMATION_H */