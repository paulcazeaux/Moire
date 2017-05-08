#ifndef INTERLAYER_COUPLINGS_HEADER
#define INTERLAYER_COUPLINGS_HEADER

#include <cmath>
#include <fftw3.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

double interlayer_term(double,double,double,double,double,double,int,int,double,double,int,int);

void load_fftw_complex(std::vector< std::vector< fftw_complex* > >&, std::string);
double interp_4point(double, double, double, double, double, double);

double inter_graphene(double,double,double,double,int,int,double,double);
double inter_tmdc(double,double,double,double,double,double,int,int,int,int);

class Interlayer_coupling{

	private:
	
		// fftw global info, to prevent having to reload *fftw.dat file for every call...
		std::vector<std::vector< fftw_complex* > > fftw_data;
		int L_x;
		int L_y;
		int length_x;
		int length_y;
	
	public:

		Interlayer_coupling();
		~Interlayer_coupling();
		
		double interp_fft(double, double, int, int, int);
		double interp_fft_v(double, double, int, int, int);

		void fft_setup(int, int, int, int, std::string);

};

#endif
