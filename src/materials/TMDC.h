/* 
 * File:   TMDC.h
 * Author:  Paul Cazeaux
 * 
 * Created on June 12, 2017, 9:00 AM
 */

#ifndef moire__materials_tmdc_h
#define moire__materials_tmdc_h

#include <algorithm>
#include <array>
#include <map>
#include <cmath>
#include <cassert>
#include "tools/numbers.h"

namespace TMDC {
    /* Syntactic sugar to identify the orbital characters we are working with */
    enum class Atom : size_t { M, X_A, X_B};
    enum class Orbital : size_t  {
                M_dz2 = 0, M_dxy = 1, M_dx2_y2 = 2, M_dxz = 3, M_dyz = 4, 
                    X_A_px = 5, X_A_py = 6, X_A_pz = 7, 
                    X_B_px = 8, X_B_py = 9, X_B_pz = 10};

    const size_t    n_orbitals          = 11;
    const double    inter_cutoff_radius = 4.;
    const double    intra_cutoff_radius = 3.;
}   /* End namespace TMDC */

namespace Coupling {

    namespace Intralayer {
        double MoS2(    const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, std::array<int, 2> vector);
        double WSe2(    const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, std::array<int, 2> vector);
        double MoSe2(   const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, std::array<int, 2> vector);
        double WS2(     const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, std::array<int, 2> vector);
    }   /* End namespace Intralayer */

    namespace Interlayer {
        double S_to_S(  const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, 
                            std::array<double, 3> vector, 
                            const double theta_row, const double theta_col);
        double Se_to_Se(const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, 
                            std::array<double, 3> vector, 
                            const double theta_row, const double theta_col);
    }   /* End namespace Interlayer */

}   /* End namespace Coupling */

namespace IsNonZero {

    namespace Intralayer {
        bool TMDC(    const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, std::array<int, 2> vector);
    }   /* End namespace Intralayer */

    namespace Interlayer {
        bool S_to_S(  const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, 
                            std::array<double, 3> vector, 
                            const double theta_row, const double theta_col);
        bool Se_to_Se(  const TMDC::Orbital orbit_row,  const TMDC::Orbital orbit_col, 
                            std::array<double, 3> vector, 
                            const double theta_row, const double theta_col);
    }   /* End namespace Interlayer */

}   /* End namespace IsNonZero */

/* We define here a library of geometric constants for each TMDC material category (M = Mo / W) */
namespace TMDC {
    namespace MS2 {
        const double  a = 3.18;
        const std::array<std::array<double, 2>, 2>
        lattice = {{ {{1. * a, 0.}}, {{-.5 * a, numbers::SQRT3_2 * a}} }};

        const std::map<Atom, std::array<double, 3> >
        atom_pos   = { 
                { Atom::M,  {{0., 0., 0.}} },
                { Atom::X_A,    {{0.5 * a, numbers::SQRT3_6 * a, -0.492138365 * a}} },
                { Atom::X_B,    {{0.5 * a, numbers::SQRT3_6 * a,  0.492138365 * a}} }  };
        /* Useful constants */
        using TMDC::n_orbitals;
        using TMDC::intra_cutoff_radius;
        using TMDC::inter_cutoff_radius;

        const double  intra_search_radius = intra_cutoff_radius + a * numbers::SQRT3_3;
        const double  inter_search_radius = inter_cutoff_radius + a * numbers::SQRT3_3;

    }

    namespace MSe2 {
        const double  a = 3.28;
        const std::array<std::array<double, 2>, 2>
        lattice = {{ {{1. * a, 0.}}, {{-.5 * a, numbers::SQRT3_2 * a}} }};

        const std::map<Atom, std::array<double, 3> > atom_pos   = 
            { 
                { Atom::M,  {{0., 0., 0.}} },
                { Atom::X_A,    {{0.5 * a, numbers::SQRT3_6 * a, -0.50451807228 * a}} },
                { Atom::X_B,    {{0.5 * a, numbers::SQRT3_6 * a, 0.50451807228 * a}} } 
            };
        /* Useful constants */
        using TMDC::n_orbitals;
        using TMDC::intra_cutoff_radius;
        using TMDC::inter_cutoff_radius;

        const double  intra_search_radius = intra_cutoff_radius + a * numbers::SQRT3_3;
        const double  inter_search_radius = inter_cutoff_radius + a * numbers::SQRT3_3;
    }



    /* Implementation of utility functions to go from Orbital type to underlying index */
    inline 
    Orbital 
    orbital(const size_t idx)
    {
        assert(idx < n_orbitals);
        return static_cast<Orbital>(idx);
    }

    inline 
    size_t 
    index(const Orbital O)
    {   
        return static_cast<size_t>(O); 
    }

    inline 
    Atom 
    atom(const Orbital O)
    {   
        switch (O)
        {
            case Orbital::M_dz2:
            case Orbital::M_dxy:
            case Orbital::M_dx2_y2:
            case Orbital::M_dxz:
            case Orbital::M_dyz:
                return Atom::M;

            case Orbital::X_A_px:
            case Orbital::X_A_py:
            case Orbital::X_A_pz:
                return Atom::X_A;

            case Orbital::X_B_px:
            case Orbital::X_B_py:
            case Orbital::X_B_pz:
                return Atom::X_B;
        }   
    }


} /* End namespace TMDC */


/* Export namespace aliases that will contain the relevant geometric data */
namespace MoS2  = TMDC::MS2;
namespace WS2   = TMDC::MS2;
namespace MoSe2 = TMDC::MSe2;
namespace WSe2  = TMDC::MSe2;

#endif
