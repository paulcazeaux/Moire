/* 
 * File:   strained_graphene.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#ifndef moire__materials_strained_graphene_h
#define moire__materials_strained_graphene_h


#include <cmath>
#include "materials/graphene.h"

namespace Coupling {

    namespace Intralayer {
        double strained_graphene(const Graphene::Orbital orbit_row, const Graphene::Orbital orbit_col, 
                            const std::array<int, 2>& vector);

    }   /* End namespace Intralayer */

}   /* End namespace Coupling */

namespace IsNonZero {

    namespace Intralayer {
        bool strained_graphene(const Graphene::Orbital orbit_row, const Graphene::Orbital orbit_col, 
                            const std::array<int, 2>& vector);
    }   /* End namespace Intralayer */

}   /* End namespace IsNonZero */

namespace StrainedGraphene {
    
    /* Geometry library */
    const double alpha           = 2.4768;
    const std::array<std::array<double, 2>, 2>
    lattice   = {{ {{1. * alpha, 0.}}, {{.5 * alpha, 0.86602540378 * alpha}} }};
    const std::map<Graphene::Atom, std::array<double, 3> > atom_pos = {
        {Graphene::Atom::A, {{0., 0., 0.}} },
        {Graphene::Atom::B, {{0.5 * alpha, 0.28867513459 * alpha, 0}} } };

    /* Useful constants */
    const int               n_orbitals      = 2;
    const double            intra_cutoff_radius = 5.7;
    const double            inter_cutoff_radius = 8.;

    const double            intra_search_radius = intra_cutoff_radius + Graphene::a/3.;
    const double            inter_search_radius = inter_cutoff_radius + Graphene::a/3.;
}



#endif
