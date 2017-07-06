/* 
 * File:   numbers.h
 * Author: Paul Cazeaux
 *
 * Created on April 22, 2017, 12:28AM
 */


#ifndef moire__tools_numbers_h
#define moire__tools_numbers_h

#include <complex>
 
namespace numbers
{
    /**
    * e
    */
    static const double E        = 2.71828182845904523536;
    /**
     * log_2 e
     */
    static const double  LOG2E   = 1.44269504088896340736;

    /**
     * log_10 e
     */
    static const double  LOG10E  = 0.43429448190325182765;

    /**
     * log_e 2
     */
    static const double  LN2     = 0.693147180559945309417;

    /**
     * log_e 10
     */
    static const double  LN10    = 2.30258509299404568402;

    /**
     * pi
     */
    static const double  PI      = 3.14159265358979323846;

    /**
     * pi/2
     */
    static const double  PI_2    = 1.57079632679489661923;

    /**
     * pi/3
     */
    static const double  PI_3    = 1.04719755119659774615;

    /**
     * pi/4
     */
    static const double  PI_4    = 0.785398163397448309616;

    /**
     * pi/6
     */
    static const double  PI_6    = 0.523598775598298873077;

    /**
     * sqrt(2)
     */
    static const double  SQRT2   = 1.41421356237309504880;

    /**
     * sqrt(2)/2
     */
    static const double  SQRT2_2 = 0.707106781186547524400;

    /**
     * sqrt(3)
     */
    static const double  SQRT3   = 1.73205080756887729353;

    /**
     * sqrt(3)/2
     */
    static const double  SQRT3_2 = 0.866025403784438646764;

    /**
     * sqrt(3)/3
     */
    static const double  SQRT3_3 = 0.577350269189625764509;

    /**
     * sqrt(3)/6
     */
    static const double  SQRT3_6 = 0.288675134594812882255;



    /**
     * Auxiliary function for complex conjugation,
     * working for non-complex scalar types also 
     */
    template<typename Scalar>
    Scalar
    conjugate(const Scalar x) {return x;}
    
    template<class T>
    std::complex<T>
    conjugate(const std::complex<T> x) {return std::conj(x); }
}

#endif
