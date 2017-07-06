/* 
 * Test file:   materials.graphene.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 16, 2017, 9:00 AM
 */

#include "tests.h"
#include "materials/graphene.h"

#include <fstream>
#include <iostream>

static const int dim = 1;

void do_test_intralayer()
{
    using Graphene::Orbital;
    size_t n = 7;
    std::vector<Orbital>            o1     {{  Orbital::A_pz,  Orbital::A_pz,  Orbital::A_pz,  Orbital::A_pz, Orbital::A_pz, Orbital::A_pz, Orbital::A_pz}};
    std::vector<Orbital>            o2     {{  Orbital::A_pz,  Orbital::B_pz,  Orbital::A_pz,  Orbital::B_pz, Orbital::B_pz, Orbital::B_pz, Orbital::B_pz}};
    std::vector<std::array<int, 2>> v      {{  {{0,0}},        {{0,0}},        {{1,0}},        {{1, -1}},     {{1,0}},       {{1,1}},       {{2,0}}   }};

    std::vector<double> t   { 0.3504,  -2.8922,  0.2425, -0.2656, 0.0235, -0.0211, 0.};
    std::vector<bool>   s   {true, true, true, true, true, true, false};

    for (size_t i=0; i < n; ++i)
    {
        //std::cout << Coupling::Intralayer::graphene( o1.at(i), o2.at(i), v.at(i) ) << std::endl;
        AssertThrow(Coupling::Intralayer::graphene( o1.at(i), o2.at(i), v.at(i) ) ==   t.at(i),
                        dealii::ExcInternalError() );
        AssertThrow(IsNonZero::Intralayer::graphene(o1.at(i), o2.at(i),  v.at(i) ) ==  s.at(i),
                        dealii::ExcInternalError() );
    }

    //std::cout << "Intralayer OK" << std::endl;
}

void do_test_interlayer()
{
    using Graphene::Orbital;
    size_t n = 100;
    Orbital o1 = Orbital::B_pz;
    Orbital o2 = Orbital::B_pz;

    std::vector<std::array<double, 3>>  v;
    std::vector<bool>   s;

    for (size_t i = 0; i<n; ++i)
    for (size_t j = 0; j<n; ++j)
    {
        double x = (2*static_cast<double>(i)-n) * 9. / static_cast<double>(n);
        double y = (2*static_cast<double>(j)-n) * 9. / static_cast<double>(n);
        v.push_back({{x,y,0}});
        s.push_back((std::sqrt(x*x + y*y) < Graphene::inter_cutoff_radius ? true : false));
    }

    std::cout << "x = ["; 
    for (size_t i = 0; i<n; ++i)
    for (size_t j = 0; j<n; ++j)
        std::cout << v[n*i+j][0] << (j < n-1 ? "," : (i < n-1 ? ";\t" : "];\n") );

    std::cout << "y = ["; 
    for (size_t i = 0; i<n; ++i)
    for (size_t j = 0; j<n; ++j)
        std::cout << v[n*i+j][1] << (j < n-1 ? "," : (i < n-1 ? ";\t" : "];\n") );
    

    double theta_row = 0.5235987756;
    double theta_col = -0.5235987756;

    std::vector<double> t   {{ .3155}};

    std::cout << "t = ["; 
    for (size_t i=0; i<n; ++i)
    for (size_t j=0; j<n; ++j)
    {
        std::cout   << Coupling::Interlayer::C_to_C( o1, o2, v.at(n*i+j), theta_row, theta_col) 
                    << (j < n-1 ? "," : (i < n-1 ? ";\t" : "];\n") );
        // AssertThrow(Coupling::Interlayer::C_to_C( o1, o2, v.at(i), theta_row, theta_col ) ==   t.at(i),
        //                 dealii::ExcInternalError() );
        //AssertThrow(IsNonZero::Interlayer::C_to_C(o1, o2,  v.at(n*i+j), theta_row, theta_col ) ==  s.at(n*i+j),
        //                dealii::ExcInternalError() );
    }

    //std::cout << "Interlayer OK" << std::endl;
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