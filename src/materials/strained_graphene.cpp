
/* 
 * File:   strained_graphene.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#include "materials/strained_graphene.h"

using Graphene::Atom;
using Graphene::Orbital;

double Coupling::Intralayer::strained_graphene(
        const Orbital orbit_row, const Orbital orbit_col, 
        const std::array<int, 2>& vector)
{
    /* Compute hexagonal homogeneous coordinates from grid coordinates.
     * We use the following system:
     *
     *                A                   A                 |             (-1,2,-1)            (1,1,-2)              
     *                             (1/2, sqrt(3)/2)         |                                                       
     *                          .                           |                        (0,1,-1)                        
     *                                                      |                                                       
     *                B                   B                 |              (-1,1,0)            (1,0,-1)             
     *                                                      |                                                       
     *                                                      |                                                       
     *      A                   A                    A      |   (-2,1,1)             (0,0,0)              (2,-1,-1)   
     *                        (0,0)                (1,0)    |                                                       
     *                                                      |                                                       
     *                .                   .                 |              (-1,0,1)            (1,-1,0)              
     *                                                      |                                                       
     *                          B                           |                        (0,-1,1)                       
     *                                                      |                                                       
     *                A                   A                 |             (-1,-1,2)            (1,-2,1)             
     *                                                                    
     * In this system, the distance between a point and the center is still the euclidean distance, on all three coordinates.
     */
    std::array<int, 2> hom_vec  {{   2 * vector[0] +     vector[1],
                                    -1 * vector[0] +     vector[1] }};
       // redundant 3rd coordinate: -1 * vector[0] - 2 * vector[1]

    /* Shift the arrow vector by the orbital coordinates */
    if (atom(orbit_col) == Atom::A && atom(orbit_row) == Atom::B)
    {
        hom_vec[0] -=  1;
        // hom_vec[2] -= -1;
    }
    else if (atom(orbit_col) == Atom::B && atom(orbit_row) == Atom::A)
    {
        hom_vec[0] +=  1;
        // hom_vec[2] += -1;
    }
    /* Compute the distance to the origin: 
     * we use the identity r = x^2 + y^2 + z^2 = x^2 + y^2 + (x+y)^2 = 2 * (x * (x+y) + y^2) 
     */
    int r = hom_vec[0] * (hom_vec[0] + hom_vec[1]) + hom_vec[1]*hom_vec[1];


    const double t_arr[9] = {0.3504, -2.8922, 0.2425, -0.2656, 0.0235, 0.0524, -0.0209, -0.0148, -0.0211};
    switch (r)
    {
        case 0:
            return t_arr[0];
        case 1:
            return t_arr[1];
        case 3:
            return t_arr[2];
        case 4:
            return t_arr[3];
        case 7:
            return t_arr[4];
        case 9:
            return t_arr[5];
        case 12:
            return t_arr[6];
        case 13:
            return t_arr[7];
        case 16:
            return t_arr[8];
        default:
            return 0.;
    }
}

bool IsNonZero::Intralayer::strained_graphene(
        const Orbital orbit_row, const Orbital orbit_col, 
        const std::array<int, 2>& vector)
{
    /* Compute hexagonal homogeneous coordinates from grid coordinates.
     * We use the following system:
     *
     *                A                   A                 |             (-1,2,-1)            (1,1,-2)              
     *                             (1/2, sqrt(3)/2)         |                                                       
     *                          .                           |                        (0,1,-1)                        
     *                                                      |                                                       
     *                B                   B                 |              (-1,1,0)            (1,0,-1)             
     *                                                      |                                                       
     *                                                      |                                                       
     *      A                   A                    A      |   (-2,1,1)             (0,0,0)              (2,-1,-1)   
     *                        (0,0)                (1,0)    |                                                       
     *                                                      |                                                       
     *                .                   .                 |              (-1,0,1)            (1,-1,0)              
     *                                                      |                                                       
     *                          B                           |                        (0,-1,1)                       
     *                                                      |                                                       
     *                A                   A                 |             (-1,-1,2)            (1,-2,1)             
     *                                                                    
     * In this system, the distance between a point and the center is still the euclidean distance, on all three coordinates.
     */
    std::array<int, 2> hom_vec  {{   2 * vector[0] +     vector[1],
                                    -1 * vector[0] +     vector[1] }};
       // redundant 3rd coordinate: -1 * vector[0] - 2 * vector[1]

    /* Shift the arrow vector by the orbital coordinates */
    if (atom(orbit_col) == Atom::A && atom(orbit_row) == Atom::B)
    {
        hom_vec[0] -=  1;
        // hom_vec[2] -= -1;
    }
    else if (atom(orbit_col) == Atom::B && atom(orbit_row) == Atom::A)
    {
        hom_vec[0] +=  1;
        // hom_vec[2] += -1;
    }
    /* Compute the distance to the origin: 
     * we use the identity r = x^2 + y^2 + z^2 = x^2 + y^2 + (x+y)^2 = 2 * (x * (x+y) + y^2) 
     */
    int r = hom_vec[0] * (hom_vec[0] + hom_vec[1]) + hom_vec[1]*hom_vec[1];

    switch (r)
    {
        case 0:
        case 1:
        case 3:
        case 4:
        case 7:
        case 9:
        case 12:
        case 13:
        case 16:
            return true;
        default:
            return false;
    }
}