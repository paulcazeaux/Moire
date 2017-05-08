/* 
 * File:   intralayer_coupling.h
 * Author: Stephen
 * 
 * Created on February 5, 2016, 3:33 PM
 */

#include <cmath>
#include <vector>
#include <iostream>

#ifndef INTRALAYER_COUPLING_H
#define INTRALAYER_COUPLING_H

double intralayer_term(double, double, double, double, double, double, int, int, int);
double intralayer_graphene(double, double, double, double, double, double);
double intralayer_strain_graphene(double, double, double, double, double, double);
double intralayer_tmdc(double, double, double, double, double, double, int, int, int);

#endif /* INTRALAYER_COUPLING_H */
