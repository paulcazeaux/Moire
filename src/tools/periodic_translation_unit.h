/* 
 * File:   periodic_translation_unit.h
 * Author: Paul Cazeaux
 *
 * Created on June 29, 2017, 15:02AM
 */


#ifndef moire__tools_periodic_translation_unit_h
#define moire__tools_periodic_translation_unit_h

#include <string>
#include <array>
#include <numeric>
#include <complex>

#include "fftw3.h"
#include <Kokkos_Core.hpp>
#include "deal.II/base/tensor.h"

#include "tools/numbers.h"

template<int dim, typename Scalar>
class PeriodicTranslationUnit
{
    static_assert(std::is_same<Scalar, double>::value || std::is_same<Scalar, std::complex<double>>::value ,
     "Periodic Translation Unit object construction failed: Scalar type must be double or std::complex<double>.");
};


template<int dim>
class PeriodicTranslationUnit<dim, double>
{
public:
    typedef Kokkos::View<double *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> > view_t;
    typedef Kokkos::View<typename std::conditional<dim == 1, double **, double *** >::type,
        Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged> > view_nd_t;

    PeriodicTranslationUnit(const int n_inner, const std::array<int, dim> n_nodes);
    ~PeriodicTranslationUnit();

    view_t view() const;

    view_nd_t view_nd() const;

    void
    translate(dealii::Tensor<1, dim> vector);

    friend std::ostream& operator<<( std::ostream& os, const PeriodicTranslationUnit<dim, double>& torus)
    {
        for (size_t i = 0; i < torus.n; ++i)
            std::cout << torus.data[i] << (i + 1 < torus.n ? ",\t" : "");
        return os;
    };

private:
    int n;
    int n_inner;
    std::array<int, dim> n_nodes;


    fftw_plan       fplan, bplan;
    double *        data;
    std::complex<double> *  fft_data;
};



template<int dim>
class PeriodicTranslationUnit<dim, std::complex<double>>
{
public:
    typedef Kokkos::View<std::complex<double> *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> > view_t;
    typedef Kokkos::View<typename std::conditional<dim == 1, std::complex<double> **, std::complex<double> *** >::type,
        Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged> > view_nd_t;

    PeriodicTranslationUnit(const int n_inner, const std::array<int, dim> n_nodes);
    ~PeriodicTranslationUnit();

    view_t view() const;

    view_nd_t view_nd() const;

    void
    translate(dealii::Tensor<1, dim> vector);

    friend std::ostream& operator<<( std::ostream& os, const PeriodicTranslationUnit<dim, std::complex<double>>& torus)
    {
        for (size_t i = 0; i < torus.n; ++i)
            os << numbers::real(torus.data[i]) << " + " << numbers::imag(torus.data[i]) << (i + 1 < torus.n ? "i\t" : "i");
        return os;
    };

private:
    int n;
    int n_inner;
    std::array<int, dim> n_nodes;


    fftw_plan       fplan, bplan;
    std::complex<double> *  data, * fft_data;
};


/**
 * Declaration of explicit specializations
 */

template<>
void
PeriodicTranslationUnit<1, double>::translate(dealii::Tensor<1, 1> vector);
template<>
void
PeriodicTranslationUnit<2, double>::translate(dealii::Tensor<1, 2> vector);

template<>
void
PeriodicTranslationUnit<1, std::complex<double>>::translate(dealii::Tensor<1, 1> vector);
template<>
void
PeriodicTranslationUnit<2, std::complex<double>>::translate(dealii::Tensor<1, 2> vector);


 #endif