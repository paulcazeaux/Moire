/* 
* File:   types.h
* Author: Paul Cazeaux
*
* Created on May 4, 2017, 5:00 PM
*/



#ifndef TYPES_H
#define TYPES_H

namespace types 
{

    typedef unsigned int global_index;
    typedef signed int   grid_index;

    typedef unsigned int subdomain_id;

    /* Definition of an invalid global index value used throughout the project */
    const global_index      invalid_global_index    = static_cast<global_index>(-1);
    const unsigned int      invalid_lattice_index   = static_cast<unsigned int>(-1);

}


#endif /* TYPES_H */