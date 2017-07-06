
/* 
 * File:   1d_model.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#include "materials/1d_model.h"

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
    if (std::abs(vector) < Toy1D::inter_search_radius )
        return Toy1D::W * std::exp(-(r*r));
    else
        return 0.;
}

bool IsNonZero::Intralayer::one_d_model(const int vector)
{
    /* Make sure that the diagonal element (vector = 0) is in the parsity pattern */
    if (vector == 1 || vector == 0 || vector == -1)
        return true;
    else 
        return false;
}

bool IsNonZero::Interlayer::one_d_model(const double vector)
{
    return (std::abs(vector) < Toy1D::inter_search_radius ); 
}