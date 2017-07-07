/* 
 * Test file:   materials.graphene.cpp
 * Author: Paul Cazeaux
 *
 * Created on June 16, 2017, 9:00 AM
 */

#include "tests.h"
#include "materials/graphene.h"
#include "tools/numbers.h"

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

    std::cout << "Intralayer OK" << std::endl;
}

void do_test_interlayer()
{
    using Graphene::Orbital;
    size_t n = 9;
    std::vector<Orbital>               o1     {{  Orbital::A_pz,  Orbital::B_pz,    Orbital::B_pz,     Orbital::B_pz,    Orbital::B_pz,    Orbital::A_pz,    Orbital::A_pz,    Orbital::A_pz,    Orbital::A_pz}};
    std::vector<Orbital>               o2     {{  Orbital::A_pz,  Orbital::A_pz,    Orbital::A_pz,     Orbital::A_pz,    Orbital::A_pz,    Orbital::A_pz,    Orbital::A_pz,    Orbital::A_pz,    Orbital::B_pz}};
    std::vector<std::array<double, 3>> v      {{  {{0,0}},        {{1.2384,0.715}}, {{1.2384,0.715}},  {{1.2384,0.715}}, {{1.2384,0.715}}, {{1.2384,0.715}}, {{1.2384,0.715}}, {{7,0}},          {{7,0}}      }};
    std::vector<double>                theta1 {{  0.,             0.,               0.5*numbers::PI_6, numbers::PI_6,    numbers::PI_3,    numbers::PI_2,    numbers::PI_4,    0.,               0.           }};
    std::vector<double>                theta2 {{  0.,             0.,               0.5*numbers::PI_6, numbers::PI_6,    -numbers::PI_3,   numbers::PI_3,    -numbers::PI_4,   0.,               0.           }};

    std::vector<double>                t      {{  .3155,          .3155,            0.290245,          0.23025,          0.11532,          0.04828,          0.070967,         -0.0001187,       0.           }};
    std::vector<bool>                  s      {{  true,           true,             true,              true,             true,             true,             true,             true,             false         }};

    for (size_t i=0; i<n; ++i)
    {
        AssertThrow(std::fabs(Coupling::Interlayer::C_to_C( o1[i], o2[i], v[i], theta1[i], theta2[i]) - t[i]) < 1e-5,
                         dealii::ExcInternalError() );
        AssertThrow(IsNonZero::Interlayer::C_to_C(o1[i], o2[i], v[i], theta1[i], theta2[i]) ==  s[i],
                        dealii::ExcInternalError() );
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