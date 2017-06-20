
/* 
 * File:   1d_model.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#include "1d_model.h"

double Coupling::Intralayer::one_d_model(const int vector)
{
    if (vector == 1 || vector == -1)
        return 1.;
    else 
        return 0.;
}

double Coupling::Interlayer::one_d_model(const double vector)
{
    double r = std::abs(vector)/Toy1D::r0;
    return Toy1D::W * std::exp(-(r*r));
}