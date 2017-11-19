/* 
 * File:   periodic_translation_unit.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 29, 2017, 15:02AM
 */

#include "periodic_translation_unit.h"


/********************************************************/
/*          Definition for Scalar = double              */
/********************************************************/

template<int dim>
PeriodicTranslationUnit<dim, double>::PeriodicTranslationUnit(const int n_inner, const std::array<int, dim> n_nodes)
    :
    n_inner ( n_inner ),
    n_nodes ( n_nodes)
{
    n = n_inner * std::accumulate(n_nodes.begin(), n_nodes.end(), 1., 
                                [] (int a, int b) {return a*b; }) ;
    int 
    n_c  = n_inner  * std::accumulate(n_nodes.begin(), n_nodes.end() - 1, 1., 
                                [] (int a, int b) {return a*b; })  
                    * (n_nodes.back()/2 + 1);

    data   = (double *) fftw_malloc(sizeof(double) * n);
    fft_data  = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * n_c);

    fplan =  fftw_plan_many_dft_r2c(dim, n_nodes.data(), n_inner, 
                                            data,  NULL, n_inner, 1,
                                            reinterpret_cast<fftw_complex *>(fft_data), NULL, n_inner, 1, 
                                            FFTW_MEASURE);
    bplan =  fftw_plan_many_dft_c2r(dim, n_nodes.data(), n_inner, 
                                            reinterpret_cast<fftw_complex *>(fft_data),  NULL, n_inner, 1,
                                            data, NULL, n_inner, 1, 
                                            FFTW_MEASURE);
}


template<int dim>
PeriodicTranslationUnit<dim, double>::~PeriodicTranslationUnit()
{
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);

    fftw_free(data);
    fftw_free(fft_data);
}


template<int dim>
typename PeriodicTranslationUnit<dim, double>::view_t
PeriodicTranslationUnit<dim, double>::view() const
{
    return view_t(data, n);
}

template<>
typename PeriodicTranslationUnit<1, double>::view_nd_t
PeriodicTranslationUnit<1, double>::view_nd() const
{
    view_nd_t View (data, n_nodes[0], n_inner);
    return View;
}

template<>
typename PeriodicTranslationUnit<2, double>::view_nd_t
PeriodicTranslationUnit<2, double>::view_nd() const
{
    Kokkos::View<double ***, Kokkos::LayoutRight, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
    View (data, n_nodes[1], n_nodes[0], n_inner);
    return View;
}

template<>
void
PeriodicTranslationUnit<1, double>::translate(dealii::Tensor<1, 1> vector)
{
    /* Forward FFT */
    fftw_execute(fplan);

    /* Phase shift */
    for (int index = 0; index < n_nodes[0]/2 + 1; ++index)
    {
        std::complex<double> phase = std::polar(1./static_cast<double>(n_nodes[0]), 
                                    -2 * numbers::PI * vector[0] * index);
        for (int j=0; j<n_inner; ++j)
            fft_data[n_inner * index + j] *= phase;
    }

    /* Backward FFT */
    fftw_execute(bplan);
}

template<>
void
PeriodicTranslationUnit<2, double>::translate(dealii::Tensor<1, 2> vector)
{
    /* Forward FFT */
    fftw_execute(fplan);

    /* Phase shift */
    int stride_y = n_nodes[0]/2 + 1;
    for (int index_x = 0; index_x < stride_y; ++index_x)
    {
        std::complex<double> phase = std::polar(1./static_cast<double>(n_nodes[0] * n_nodes[1]), 
                                -2 * numbers::PI * vector[0] * index_x);
        for (int index_y = 0; index_y < n_nodes[1]; ++index_y)
            for (int j=0; j<n_inner; ++j)
                fft_data[ n_inner * (stride_y * index_y + index_x) + j] *= phase;

    }
    if (n_nodes[1] % 2 == 0) // Even number of nodes
    {
        for (int index_y = 0; index_y < n_nodes[1]/2; ++index_y)
        {
            std::complex<double>
            phase = std::polar(1., -2 * numbers::PI * vector[1] * index_y);
            for (int i=0; i<n_inner * stride_y; ++i)
                fft_data[ n_inner * stride_y * index_y + i] *= phase;
        }
        {
            int index_y = n_nodes[1] / 2; // Nyquist frequency
            std::complex<double>
            phase = std::cos(2 * numbers::PI * vector[1] * index_y);
            for (int i=0; i<n_inner * stride_y; ++i)
                fft_data[ n_inner * stride_y * index_y + i] *= phase;
        }
        for (int index_y = n_nodes[1]/2 + 1; index_y < n_nodes[1]; ++index_y)
        {
            std::complex<double>
            phase = std::polar(1., -2 * numbers::PI * vector[1] * (index_y - n_nodes[1]));
            for (int i=0; i<n_inner * stride_y; ++i)
                fft_data[ n_inner * stride_y * index_y + i] *= phase;
        }
    }
    else // Odd number of nodes
    {
        for (int index_y = 1; index_y <= n_nodes[1]/2; ++index_y)
        {
            std::complex<double>
            phase = std::polar(1., -2 * numbers::PI * vector[1] * index_y);
            for (int i=0; i<n_inner * stride_y; ++i)
                fft_data[ n_inner * stride_y * index_y + i] *= phase;
        }
        for (int index_y = n_nodes[1]/2 + 1; index_y < n_nodes[1]; ++index_y)
        {
            std::complex<double>
            phase = std::polar(1., -2 * numbers::PI * vector[1] * (index_y - n_nodes[1]));
            for (int i=0; i<n_inner * stride_y; ++i)
                fft_data[ n_inner * stride_y * index_y + i] *= phase;
        }
    }

    /* Backward FFT */
    fftw_execute(bplan);
}

/********************************************************/
/*      Definition for Scalar = std::complex<double>    */
/********************************************************/

template<int dim>
PeriodicTranslationUnit<dim, std::complex<double>>::PeriodicTranslationUnit(const int n_inner, const std::array<int, dim> n_nodes)
    :
    n_inner ( n_inner ),
    n_nodes ( n_nodes)
{
    n = n_inner * std::accumulate(n_nodes.begin(), n_nodes.end(), 1., 
                                [] (int a, int b) {return a*b; }) ;

    data   = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * n);
    fft_data  = (std::complex<double> *) fftw_malloc(sizeof(std::complex<double>) * n);

    fplan =  fftw_plan_many_dft(dim, n_nodes.data(), n_inner, 
                                            reinterpret_cast<fftw_complex *>(data),  NULL, n_inner, 1,
                                            reinterpret_cast<fftw_complex *>(fft_data), NULL, n_inner, 1, 
                                            FFTW_FORWARD, FFTW_MEASURE);
    bplan =  fftw_plan_many_dft(dim, n_nodes.data(), n_inner, 
                                            reinterpret_cast<fftw_complex *>(fft_data),  NULL, n_inner, 1,
                                            reinterpret_cast<fftw_complex *>(data), NULL, n_inner, 1, 
                                            FFTW_BACKWARD, FFTW_MEASURE);
}

template<int dim>
PeriodicTranslationUnit<dim, std::complex<double>>::~PeriodicTranslationUnit()
{
    fftw_destroy_plan(fplan);
    fftw_destroy_plan(bplan);

    fftw_free(fft_data);
    fftw_free(data);
}


template<int dim>
typename PeriodicTranslationUnit<dim, std::complex<double>>::view_t
PeriodicTranslationUnit<dim, std::complex<double>>::view() const
{
    return view_t(data, n);
}


template<>
typename PeriodicTranslationUnit<1, std::complex<double>>::view_nd_t
PeriodicTranslationUnit<1, std::complex<double>>::view_nd() const
{
    return view_nd_t(data, n_nodes[0], n_inner);
}

template<>
typename PeriodicTranslationUnit<2, std::complex<double>>::view_nd_t
PeriodicTranslationUnit<2, std::complex<double>>::view_nd() const
{
    return view_nd_t(data, n_nodes[1], n_nodes[0], n_inner);
}


template<>
void
PeriodicTranslationUnit<1, std::complex<double>>::translate(dealii::Tensor<1, 1> vector)
{
    /* Forward FFT */
    fftw_execute(fplan);

    /* Apply phase shift */
    if (n_nodes[0] % 2 == 0) // Even number of nodes
    {
        for (int index = 0; index < n_nodes[0]/2; ++index)
        {
            std::complex<double>
            phase = std::polar(1./static_cast<double>(n_nodes[0]), -2 * numbers::PI * vector[0] * index);
            for (int j=0; j<n_inner; ++j)
                fft_data[n_inner * index + j] *= phase;
        }
        {
            int index = n_nodes[0] / 2; // Nyquist frequency
            std::complex<double>
            phase = std::cos(2 * numbers::PI * vector[0] * index) / static_cast<double>(n_nodes[0]);
            for (int j=0; j<n_inner; ++j)
                fft_data[n_inner * index + j] *= phase;
        }
        for (int index = n_nodes[0]/2 + 1; index < n_nodes[0]; ++index)
        {
            std::complex<double>
            phase = std::polar(1./static_cast<double>(n_nodes[0]), -2 * numbers::PI * vector[0] * (index - n_nodes[0]));
            for (int j=0; j<n_inner; ++j)
                fft_data[n_inner * index + j] *= phase;
        }
    }
    else // Odd number of nodes
    {
        for (int index = 0; index <= n_nodes[0]/2; ++index)
        {
            std::complex<double>
            phase = std::polar(1./static_cast<double>(n_nodes[0]), -2 * numbers::PI * vector[0] * index);
            for (int j=0; j<n_inner; ++j)
                fft_data[n_inner * index + j] *= phase;
        }
        for (int index = n_nodes[0]/2 + 1; index < n_nodes[0]; ++index)
        {
            std::complex<double>
            phase = std::polar(1./static_cast<double>(n_nodes[0]), -2 * numbers::PI * vector[0] * (index - n_nodes[0]));
            for (int j=0; j<n_inner; ++j)
                fft_data[n_inner * index + j] *= phase;
        }
    }

    /* Backward FFT */
    fftw_execute(bplan);
}

template<>
void
PeriodicTranslationUnit<2, std::complex<double>>::translate(dealii::Tensor<1, 2> vector)
{
    /* Forward FFT */
    fftw_execute(fplan);

    /* Apply phase shift */
    double C = 1./static_cast<double>(n_nodes[0]);
    if (n_nodes[0] % 2 == 0) // Even number of nodes
    {
        for (int index_x = 0; index_x < n_nodes[0]/2; ++index_x)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[0] * index_x);
            for (int index_y = 0; index_y < n_nodes[1]; ++index_y)
                for (int j=0; j<n_inner; ++j)
                    fft_data[ n_inner * (n_nodes[0] * index_y + index_x) + j] *= phase;
        }
        {
            int index_x = n_nodes[0] / 2; // Nyquist frequency
            std::complex<double>
            phase = C * std::cos(2 * numbers::PI * vector[0] * index_x) ;
            for (int index_y = 0; index_y < n_nodes[1]; ++index_y)
                for (int j=0; j<n_inner; ++j)
                    fft_data[ n_inner * (n_nodes[0] * index_y + index_x) + j] *= phase;
        }
        for (int index_x = n_nodes[0]/2 + 1; index_x < n_nodes[0]; ++index_x)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[0] * (index_x - n_nodes[0]));
            for (int index_y = 0; index_y < n_nodes[1]; ++index_y)
                for (int j=0; j<n_inner; ++j)
                    fft_data[ n_inner * (n_nodes[0] * index_y + index_x) + j] *= phase;
        }
    }
    else // Odd number of nodes
    {
        for (int index_x = 0; index_x <= n_nodes[0]/2; ++index_x)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[0] * index_x);
            for (int index_y = 0; index_y < n_nodes[1]; ++index_y)
                for (int j=0; j<n_inner; ++j)
                    fft_data[ n_inner * (n_nodes[0] * index_y + index_x) + j] *= phase;
        }
        for (int index_x = n_nodes[0]/2 + 1; index_x < n_nodes[0]; ++index_x)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[0] * (index_x - n_nodes[0]));
            for (int index_y = 0; index_y < n_nodes[1]; ++index_y)
                for (int j=0; j<n_inner; ++j)
                    fft_data[ n_inner * (n_nodes[0] * index_y + index_x) + j] *= phase;
        }
    }


    C = 1./static_cast<double>(n_nodes[1]);
    if (n_nodes[1] % 2 == 0) // Even number of nodes
    {
        for (int index_y = 0; index_y < n_nodes[1]/2; ++index_y)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[1] * index_y);
            for (int i=0; i<n_inner * n_nodes[0]; ++i)
                fft_data[ n_inner * n_nodes[0] * index_y + i] *= phase;
        }
        {
            int index_y = n_nodes[1] / 2; // Nyquist frequency
            std::complex<double>
            phase = C * std::cos(2 * numbers::PI * vector[1] * index_y);
            for (int i=0; i<n_inner * n_nodes[0]; ++i)
                fft_data[ n_inner * n_nodes[0] * index_y + i] *= phase;
        }
        for (int index_y = n_nodes[1]/2 + 1; index_y < n_nodes[1]; ++index_y)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[1] * (index_y - n_nodes[1]));
            for (int i=0; i<n_inner * n_nodes[0]; ++i)
                fft_data[ n_inner * n_nodes[0] * index_y + i] *= phase;
        }
    }
    else // Odd number of nodes
    {
        for (int index_y = 0; index_y <= n_nodes[1]/2; ++index_y)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[1] * index_y);
            for (int i=0; i<n_inner * n_nodes[0]; ++i)
                fft_data[ n_inner * n_nodes[0] * index_y + i] *= phase;
        }
        for (int index_y = n_nodes[1]/2 + 1; index_y < n_nodes[1]; ++index_y)
        {
            std::complex<double>
            phase = std::polar(C, -2 * numbers::PI * vector[1] * (index_y - n_nodes[1]));
            for (int i=0; i<n_inner * n_nodes[0]; ++i)
                fft_data[ n_inner * n_nodes[0] * index_y + i] *= phase;
        }
    }
    
    /* Backward FFT */
    fftw_execute(bplan);
}




/**
 * Explicit instantiations
 */


template
class PeriodicTranslationUnit<1, double>;
template
class PeriodicTranslationUnit<2, double>;

template
class PeriodicTranslationUnit<1, std::complex<double>>;
template
class PeriodicTranslationUnit<2, std::complex<double>>;