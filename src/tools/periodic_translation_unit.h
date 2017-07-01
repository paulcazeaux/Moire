/* 
 * File:   periodic_translation_unit.h
 * Author: Paul Cazeaux
 *
 * Created on June 29, 2017, 15:02AM
 */


#include <string>
#include <array>
#include <numeric>
#include <complex>

#include "fftw3.h"
#include <Kokkos_Core.hpp>
#include "deal.II/base/tensor.h"

#include "tools/numbers.h"

#ifndef moire__tools_periodic_translation_unit_h
#define moire__tools_periodic_translation_unit_h

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
    PeriodicTranslationUnit(const int n_inner, const std::array<int, dim> n_nodes);
    ~PeriodicTranslationUnit();

    Kokkos::View<double *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
    view() const;

    void
    translate(dealii::Tensor<1, dim> vector);

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
    PeriodicTranslationUnit(const int n_inner, const std::array<int, dim> n_nodes);
    ~PeriodicTranslationUnit();

    Kokkos::View<std::complex<double> *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
    view() const;

    void
    translate(dealii::Tensor<1, dim> vector);

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
Kokkos::View<double *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
PeriodicTranslationUnit<1, double>::view() const;
template<>
void
PeriodicTranslationUnit<1, double>::translate(dealii::Tensor<1, 1> vector);


template<>
Kokkos::View<double *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
PeriodicTranslationUnit<2, double>::view() const;
template<>
void
PeriodicTranslationUnit<2, double>::translate(dealii::Tensor<1, 2> vector);

template<>
Kokkos::View<std::complex<double> *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
PeriodicTranslationUnit<1, std::complex<double>>::view() const;
template<>
void
PeriodicTranslationUnit<1, std::complex<double>>::translate(dealii::Tensor<1, 1> vector);


template<>
Kokkos::View<std::complex<double> *, Kokkos::Serial, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
PeriodicTranslationUnit<2, std::complex<double>>::view() const;
template<>
void
PeriodicTranslationUnit<2, std::complex<double>>::translate(dealii::Tensor<1, 2> vector);


 #endif