/* 
* File:   types.h
* Author: Paul Cazeaux
*
* Created on May 4, 2017, 5:00 PM
*/

#ifndef moire__tools_types_h
#define moire__tools_types_h

#include <cstddef>
#include <cstdint>
#include <Tpetra_Core.hpp>

namespace types 
{

    typedef long long   glob_t;
    typedef int32_t     loc_t;
    typedef int8_t      block_t;

    typedef uint8_t subdomain_id;

    /* Definition of an invalid global index value used throughout the project */
    const   glob_t      invalid_global_index    = static_cast< glob_t>(-1);
    const   loc_t       invalid_local_index     = static_cast<loc_t>(-1);
    const   subdomain_id invalid_id             = static_cast<subdomain_id>(-1);

    typedef struct MemUsage {
        size_t Vectors;
        size_t Matrices;
        size_t InitArrays;
        size_t Static;
    } MemUsage;

    typedef typename Kokkos::Compat::KokkosSerialWrapperNode DefaultNode ;
}


#endif
