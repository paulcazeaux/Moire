/* 
 * File:   1d_model.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#ifndef TOY_MODEL_1D_H
#define TOY_MODEL_1D_H

#include <array>
#include <cmath>

namespace Toy1D {
    /* Parameters library */
    const double            alpha           = 1.;
    const std::array<std::array<double, 1>, 1> // For compatibility with 2D case...
                            lattice = {{ {{1.}} }};
    const double            atom_pos = 0.;

    const double            intra_search_radius = 1.1;
    const double            inter_search_radius = 6.;

    const int               n_orbitals      = 1;

    const double            W = .5;
    const double            r0 = .25;
};

namespace Coupling {

namespace Intralayer {
    double one_d_model(const int vector);

}   /* End namespace Intralayer */

namespace Interlayer {
    double one_d_model(const double vector);

}   /* End namespace Interlayer */

}   /* End namespace Coupling */
#endif