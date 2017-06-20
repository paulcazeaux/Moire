/* 
 * File:   graphene.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#ifndef moire__materials_graphene_h
#define moire__materials_graphene_h

#include <array>
#include <algorithm>
#include <map>
#include <cmath>
#include <cassert>
#include "tools/numbers.h"

namespace Graphene {
    /* Syntactic sugar to identify the orbital characters we are working with */
    enum class Atom : int { A, B};
    enum class Orbital : int {A_pz = 0, B_pz = 1};

}   /* End namespace Graphene */

/* Common namespaces where to define the tight-binding hopping coefficient functions  */
namespace Coupling {

    namespace Intralayer {
        double graphene(const Graphene::Orbital orbit_row, const Graphene::Orbital orbit_col, 
                            const std::array<int, 2>& vector);
    }   /* End namespace Intralayer */

    namespace Interlayer {
        double C_to_C(const Graphene::Orbital orbit_row, const Graphene::Orbital orbit_col,
                        std::array<double, 3> vector, 
                        const double theta_row, const double theta_col);
    }   /* End namespace Interlayer */

}   /* End namespace Coupling */



namespace Graphene {
    /* Geometry library */
    const double a = 2.4768;
    const std::array<std::array<double, 2>, 2>
    lattice = {{ {{1. * a, 0.}}, {{.5 * a, numbers::SQRT3_2 * a}} }};

    const std::map<Atom, std::array<double, 3> > 
    atom_pos = {
        {Atom::A, {{0., 0., 0.}} },
        {Atom::B, {{0.5 * a, numbers::SQRT3_6 * a, 0}} } };

    /* Useful constants */
    const int               n_orbitals      = 2;
    const double            intra_cutoff_radius = 4 * numbers::SQRT3_3 * a + 1e-5;
    const double            inter_cutoff_radius = 8.;

    const double            intra_search_radius = intra_cutoff_radius + a/3.;
    const double            inter_search_radius = inter_cutoff_radius + a/3.;


    /* Utility functions to go from Orbital type to its integer index */
    inline 
    Orbital 
    orbital(const int idx)
    {
        assert(idx >= 0 && idx < n_orbitals);
        return static_cast<Orbital>(idx);
    }

    inline 
    int 
    index(const Orbital O)
    {   
        return static_cast<int>(O); 
    }

    inline 
    Atom 
    atom(const Orbital O)
    {   
        switch (O)
        {
            case Orbital::A_pz:
                return Atom::A;

            case Orbital::B_pz:
                return Atom::B;
        }   
    }
}   /* End namespace Graphene */
#endif
