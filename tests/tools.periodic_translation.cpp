/* 
 * Test file:   tools.periodic_translation.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 16, 2017, 9:00 AM
 */

#include "tests.h"
#include "tools/periodic_translation_unit.h"

template<typename Scalar>
void do_test_1(int n)
{
    PeriodicTranslationUnit<1,Scalar> torus (1, {{n}});

    for (size_t i = 0; i < n; ++i)
        torus.view()[i] = std::exp(std::sin(2 * numbers::PI * i/static_cast<double>(n)));
        
    for (size_t i = 0; i < 4*n; ++i)
        torus.translate({{1./(4*n)}});

    for (size_t i = 0; i < n; ++i)
        AssertThrow( std::abs(torus.view()[i] - std::exp(std::sin(2 * numbers::PI * i/static_cast<double>(n))) ) < 1e-10,
                        dealii::ExcInternalError() );
    
    std::cout << "Periodic, 1D translation OK" << std::endl;
}

template<typename Scalar>
void do_test_2(int n)
{
    PeriodicTranslationUnit<2,Scalar> torus (2, {{n, n}});

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t o = 0 ; o < 2; ++o)
                torus.view_nd()(i,j,o) = (std::exp(std::sin(2 * numbers::PI * i/static_cast<double>(n))) - 1.)
                                            * (std::exp(std::sin(2 * numbers::PI * j/static_cast<double>(n))) - 1.)
                                            * (o == 0 ? 1 : -1);
    
    for (size_t i = 0; i < 4*n; ++i)
        torus.translate({{1./(4.*n), 3./(4.*n)}});

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t o = 0 ; o < 2; ++o)
                AssertThrow( std::abs(torus.view_nd()(i,j,o) - (std::exp(std::sin(2 * numbers::PI * i/static_cast<double>(n))) - 1.)
                                            * (std::exp(std::sin(2 * numbers::PI * j/static_cast<double>(n))) - 1.)
                                            * (o == 0 ? 1 : -1) ) < 1e-10,
                        dealii::ExcInternalError() );

    std::cout << "Periodic, 2D translation OK" << std::endl;
}

 int main(int argc, char** argv) {
        /*********************************************************/
        /*                  Initialize MPI                       */
        /*********************************************************/
    try
    {
        /* Test 1D and 2D, odd and even, doubles and complex numbers */
        do_test_1<double>(10);
        do_test_1<double>(11);
        do_test_1<std::complex<double>>(10);
        do_test_1<std::complex<double>>(11);
        do_test_2<double>(10);
        do_test_2<double>(11);
        do_test_2<std::complex<double>>(10);
        do_test_2<std::complex<double>>(11);
    }
  catch (std::exception &exc)
    {
      std::cout << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cout << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return -1;
    }
  catch (...)
    {
      std::cout << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cout << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return -1;
    };

  return 0;
}