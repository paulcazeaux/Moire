/* 
 * Test file:   materials.1d_toy_model.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 16, 2017, 9:00 AM
 */

#include "tests.h"
#include "materials/1d_model.h"

void do_test_intralayer()
{
    size_t n = 7;
    std::vector<int>    v   {-3, -2, -1, 0, 1, 2, 3};
    std::vector<double> t   { 0,  0,  1., 0, 1., 0, 0};
    std::vector<bool>   s   {false, false, true, true, true, false, false};

    for (size_t i=0; i < n; ++i)
    {
        AssertThrow(Coupling::Intralayer::one_d_model( v.at(i) ) ==   t.at(i),
                        std::logic_error("Error setting up intralayer elements of 1D toy model") );
        AssertThrow(IsNonZero::Intralayer::one_d_model( v.at(i) ) ==  s.at(i),
                        std::logic_error("Error setting up intralayer elements of 1D toy model") );
    }

    std::cout << "Intralayer OK" << std::endl;
}

void do_test_interlayer()
{
    size_t n = 7;
    std::vector<double> v       {0.,  .25,        .5,         .75,         1.,          1.25,        2.};
    std::vector<double> t       {0.5, 1.83943e-1, 9.15782e-3, 6.17049e-05, 5.62676e-08, 6.94397e-12, 0.      };
    std::vector<double> t_b     {true, true, true, true, true, true, false };

    for (size_t i=0; i < n; ++i)
    {
        AssertThrow( std::fabs( Coupling::Interlayer::one_d_model( v.at(i) ) - t.at(i) ) < 1e-5, 
                        std::logic_error("Error setting up interlayer elements of 1D toy model") );
        AssertThrow( IsNonZero::Interlayer::one_d_model(v.at(i) ) == t_b.at(i), 
                        std::logic_error("Error setting up interlayer elements of 1D toy model") );
    }


    std::cout << "Interlayer OK" << std::endl;
}

int main(int argc, char** argv) {
    try
    {
        do_test_intralayer();
        do_test_interlayer();
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